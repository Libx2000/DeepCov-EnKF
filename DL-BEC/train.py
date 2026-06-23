import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--gpus", type=str, default="0")
    return parser.parse_args()

class H5PerturbationDataset(Dataset):
    def __init__(self, file_path, m):
        self.file = h5py.File(file_path, 'r')
        self.data = self.file['perturbations']
        self.m = m
        self.num_samples = self.data.shape[0]
        
        sample_shape = self.data[0].shape
        assert all(self.data[i].shape == sample_shape for i in range(self.num_samples)), "样本形状不一致"
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()
        return sample
def collate_fn(batch):
    return torch.stack(batch)


class BlockLowRankCov(nn.Module):
    def __init__(self, m, block_size, rank):
        super().__init__()
        self.m = m
        self.block_size = block_size
        self.num_blocks = m // block_size
        self.rank = rank
        
        self.block_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(block_size, 64),
                nn.ReLU(),
                nn.Linear(64, block_size * (rank + 1))
            ) for _ in range(self.num_blocks)
        ])
        
    def forward(self, X):
        batch_size = X.size(0)
        X_centered = X - X.mean(dim=1, keepdim=True)
        
        blocks = []
        for i in range(self.num_blocks):
            X_block = X_centered[:, :, i*self.block_size:(i+1)*self.block_size]
            
            params = self.block_nets[i](X_block.mean(dim=1))
            L_flat = params[..., :self.block_size*self.rank]
            diag_log = params[..., self.block_size*self.rank:]
            
            L = L_flat.view(batch_size, self.block_size, self.rank)
            D = torch.exp(diag_log)
            cov_block = L @ L.transpose(-1, -2) + torch.diag_embed(D)
            
            blocks.append(cov_block)
            
        return blocks
    

class BlockDiagMatrix:
    def __init__(self, blocks,m):
        self.blocks = blocks
        self.m = m
    def matmul(self, X):
        results = []
        start = 0
        for block in self.blocks:
            end = start + block.size(-1)
            X_block = X[..., start:end]
            results.append(X_block @ block)
            start = end
        return torch.cat(results, dim=-1)
    
    def to_dense(self):
        matrix = torch.zeros(self.blocks[0].size(0), self.m, self.m)
        start = 0
        for block in self.blocks:
            end = start + block.size(-1)
            matrix[:, start:end, start:end] = block
            start = end
        return matrix

class CovarianceLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred_blocks, target_blocks):
        mse_loss = 0.0
        for p_block, t_block in zip(pred_blocks, target_blocks):
            mse_loss += F.mse_loss(p_block, t_block)
        mse_loss /= len(pred_blocks)
        
        eig_loss = 0.0
        for block in pred_blocks:
            eigvals = torch.linalg.eigvalsh(block)
            eig_loss += F.relu(1e-6 - eigvals[:, 0]).mean()
            
        return mse_loss + self.alpha * eig_loss

def train(args, criterion):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = len(args.gpus.split(','))
    
    model = BlockLowRankCov(args.m, args.block_size, args.rank)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler()
    
    dataset = H5PerturbationDataset(args.data_path, args.m)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, X in enumerate(loader):
            X = X.to(device)
            
            with autocast():
                pred_blocks = model(X)
                
                batch_size, n, m = X.shape
                X_centered = X - X.mean(dim=1, keepdim=True)
                
                target_blocks = []
                for i in range(model.module.num_blocks if num_gpus>1 else model.num_blocks):
                    start = i * args.block_size
                    end = start + args.block_size
                    X_block = X_centered[:, :, start:end]
                    target = torch.einsum('bni,bnj->bij', X_block, X_block) / (n - 1)
                    target_blocks.append(target)
                
                loss = criterion(pred_blocks, target_blocks)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx+1)
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx} | Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            'model': model.module.state_dict() if num_gpus>1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    criterion = CovarianceLoss(alpha=0.1)
    
    train(args, criterion)