import torch
import numpy as np
import time
import os
import glob
import xarray as xr
import torch
from src.utils.weighted_acc_rmse import weighted_acc_torch, weighted_rmse_torch, weighted_mae_torch, activity_torch
from src.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR
from scipy.stats import multivariate_normal
from scipy.linalg import solve
import torch.nn as nn

class BlockLowRankCov(nn.Module):
    def __init__(self, m, block_size, rank):
        super().__init__()
        self.m = m
        self.block_size = block_size
        self.num_blocks = m // block_size
        self.rank = rank
        
        # 每个子块的参数生成器
        self.block_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(block_size, 64),
                nn.ReLU(),
                nn.Linear(64, block_size * (rank + 1))  # 正确输出维度
            ) for _ in range(self.num_blocks)
        ])
        
    def forward(self, X):
        """ 
        Args:
            X: 输入扰动矩阵 (batch, n, m)
        Returns:
            BlockDiagMatrix对象，包含分块协方差矩阵
        """
        batch_size = X.size(0)
        X_centered = X - X.mean(dim=1, keepdim=True)
        
        blocks = []
        for i in range(self.num_blocks):
            X_block = X_centered[:, :, i*self.block_size:(i+1)*self.block_size]
            
            # 生成子块参数
            params = self.block_nets[i](X_block.mean(dim=1))
            L_flat = params[..., :self.block_size*self.rank]
            diag_log = params[..., self.block_size*self.rank:]
            
            # 构建子协方差矩阵
            L = L_flat.view(batch_size, self.block_size, self.rank)
            D = torch.exp(diag_log)
            cov_block = L @ L.transpose(-1, -2) + torch.diag_embed(D)
            
            blocks.append(cov_block)
            
        return blocks

def read_data(path, raw_variables, variables, resolution, years, mean, std, ic, prediction_length, dt):
    out_vars = []
    times = []
    for year in years:
        np_vars = {}
        for var in raw_variables:
            ps = glob.glob(os.path.join(f"{path}", f"{var}_{resolution}deg", f"*{year}*.nc"))
            # ps = glob.glob(os.path.join(f"{path}/{resolution}deg", f"{var}_{resolution}deg", f"*{year}*.nc"))
            ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                np_vars[var] = ds[code].to_numpy()[ic:ic+prediction_length*dt+1:dt].astype(np.float32)

            else:  # multiple-level variables, only use a subset
                assert len(ds[code].shape) == 4
                all_levels = ds["level"][:].to_numpy().astype(np.float32)
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = ds.sel(level=[level])
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[ic:ic+prediction_length*dt+1:dt].astype(np.float32)
    
        out_vars.append(np.concatenate([np_vars[k] for k in variables], axis=1))
    
    out_vars = np.concatenate(out_vars, axis=0).astype(np.float32)
    
    return (out_vars - mean) / std, ds.time.values, ds.lon.values, ds.lat.values

def reindex(global_idx, n_samples_per_shards):
    total_idx = 0
    for i in range(12):
        if (global_idx >= total_idx) and (global_idx < total_idx + n_samples_per_shards[i]):
            break
        else:
            total_idx += n_samples_per_shards[i]
    shard_idx = i #which month we are on
    local_idx = ((global_idx - total_idx) % n_samples_per_shards[i]) #which sample in that month we are on - determines indices for centering

    return shard_idx, local_idx




def particle_filter_assimilation(
    particles: np.ndarray,      # 粒子集合 [n, m]
    weights: np.ndarray,        # 粒子权重 [n]
    observation: np.ndarray,    # 观测数据 [m]
    obs_noise_cov: np.ndarray,  # 观测噪声协方差 [m, m]
    resample_method: str = 'systematic',  # 重采样方法 ['systematic', 'residual', 'stratified']
    resample_threshold: float = 0.5       # 重采样阈值
):
    """
    简化的粒子滤波同化函数
    输入输出维度：
    输入粒子形状: (n, m)
    输入权重形状: (n,)
    输入观测形状: (m,)
    输出粒子形状: (n, m)
    输出权重形状: (n,)
    估计结果形状: (m,)
    """
    n, m = particles.shape
    
    # ===== 观测更新 =====
    # 假设观测模型为恒等函数 (H = I)
    innovations = observation - particles  # 计算观测残差 [n, m]
    
    # 计算马氏距离 (向量化实现)
    inv_obs_cov = np.linalg.inv(obs_noise_cov)
    log_likelihood = -0.5 * np.einsum('ni,ij,nj->n', innovations, inv_obs_cov, innovations)
    
    # 更新权重（防止数值下溢）
    max_log = np.max(log_likelihood)
    new_weights = np.exp(log_likelihood - max_log) * weights
    new_weights /= new_weights.sum()
    
    # ===== 重采样决策 =====
    neff = 1.0 / np.sum(new_weights**2)  # 有效粒子数
    if neff < resample_threshold * n:
        indices = resample(new_weights, n, method=resample_method)
        particles = particles[indices]
        new_weights = np.full(n, 1.0/n)  # 重置权重
    
    # ===== 状态估计 =====
    state_est = np.average(particles, weights=new_weights, axis=0)
    
    return particles,new_weights,state_est


def resample(weights: np.ndarray, n: int, method: str) -> np.ndarray:
    """多方法重采样器"""
    if method == 'systematic':
        cumsum = np.cumsum(weights)
        return np.searchsorted(cumsum, (np.arange(n) + np.random.random()) / n)
    elif method == 'residual':
        copies = (n * weights).astype(int)
        residual = weights - copies / n
        residual /= residual.sum()
        return np.concatenate([np.repeat(np.arange(n), copies),
                               np.random.choice(n, n - copies.sum(), p=residual)])
    elif method == 'stratified':
        return _stratified_resample(weights, n)
    else:
        raise ValueError(f"未知重采样方法: {method}")

def _stratified_resample(weights: np.ndarray, n: int) -> np.ndarray:
    """分层重采样实现"""
    positions = (np.random.rand(n) + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    return np.searchsorted(cumsum, positions)

def load_modelP(checkpoint_path, m, block_size, rank):
    model = BlockLowRankCov(m, block_size, rank)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为评估模式
    return model

# 使用示例
modelP = load_modelP(
    checkpoint_path=f"/hpcfs/fhome/yangjh16/CoastNet/p_train/models/epoch_100.pth",
    m=28800,
    block_size=600,
    rank=10
).cuda()
modelP = modelP.float()  

def enkf(seq_pred, real_obs, R, model):
    
    """
    集合卡尔曼滤波数据同化
    
    参数:
        seq_pred (np.ndarray): 预测集合，形状为 (n, m)
        real_obs (np.ndarray): 观测数据，形状为 (m,)
        R (np.ndarray): 观测误差协方差矩阵，形状为 (m, m)
    
    返回:
        np.ndarray: 分析后的集合，形状为 (n, m)
    """
    # if np.any(np.isnan(seq_pred)) or np.any(np.isinf(seq_pred)):
    #     print("输入矩阵1包含 NaN 或 Inf！")
    # if np.any(np.isnan(real_obs)) or np.any(np.isinf(real_obs)):
    #     print("输入矩阵2包含 NaN 或 Inf！")
    n, m = seq_pred.shape
    
    # H = np.eye(m)  # 观测算子（假设全观测）.

    
    # 计算集合平均和扰动
    ensemble_mean = np.mean(seq_pred, axis=0)

    X_prime = seq_pred - ensemble_mean  # 扰动矩阵
    # X = torch.from_numpy(X_prime).unsqueeze(0).to(dtype=torch.float32).cuda()
    
    # with torch.no_grad():
    #     blocks = model(X)
    # P = torch.block_diag(*[b.squeeze(0) for b in blocks])
    # 计算集合协方差和卡尔曼增益
    # start = time.perf_counter()  # 开始计时
    # P = np.zeros([28800,28800])
    # for i in range(28800):
    #     for j in range(28800):
    #         P[i,j] = np.dot(X_prime[:,i],X_prime[:,j])
    # end = time.perf_counter()  # 结束计时
    # print(f"代码执行耗时: {end - start:.6f} 秒")
            
    P = (X_prime.T @ X_prime) / (n - 1) + 1e-6 * np.eye(m) # 协方差矩阵
    # P = P.cpu().detach().to(torch.float32).numpy()
    np.save(f"/hpcfs/fhome/yangjh16/CoastNet/B.npy",P)
    P1 = np.linalg.inv(P)
    np.save(f"/hpcfs/fhome/yangjh16/CoastNet/B_inv.npy",P1)
    # K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    K = P @ np.linalg.inv(P + R)
    
   

    K = K.astype(np.float32)# 卡尔曼增益

    
    # 生成扰动观测集合

    obs_perturbations = np.random.normal(0, 0.2, [n,m])

    perturbed_obs = real_obs + obs_perturbations

    
    # 计算分析集合
    innovation = perturbed_obs - seq_pred  # 观测空间中的创新
    
    analysis_ensemble = seq_pred + (innovation @ K.T)

    
    return analysis_ensemble





def autoregressive_inference(ic, 
                             resolution, 
                             years,
                             in_mean,
                             in_std,
                             out_mean,
                             out_std,
                             module, 
                             dt, 
                             prediction_length,
                             h5gt,
                             h5gt1,
                             h5clim, 
                             mult, 
                             in_variables,
                             out_variables, 
                             device):
    
    
    
    ic = int(ic)
    n_samples_per_shards = [h5gt[i][in_variables[0]].shape[0] for i in range(len(h5gt))]
    prediction_length = int(prediction_length) // dt
    idx_targets = [ic + int(dt * i) for i in range(1 + prediction_length)]
    gts = []
    for i in range(len(idx_targets)):
        shard_idx, local_idx = reindex(idx_targets[i], n_samples_per_shards)
        gt = np.concatenate([np.expand_dims(h5gt[shard_idx % len(h5gt)][k][local_idx], axis=0)
                                        for k in in_variables], axis=1).astype(np.float32)
        gts.append(torch.from_numpy(np.nan_to_num((gt - in_mean) / in_std)))
        
    gts1 = []
    for i in range(len(idx_targets)):
        shard_idx, local_idx = reindex(idx_targets[i], n_samples_per_shards)
        gt1 = np.concatenate([np.expand_dims(h5gt1[shard_idx % len(h5gt1)][k][local_idx], axis=0)
                                        for k in ['sst']], axis=1).astype(np.float32)
        gts1.append(np.nan_to_num(gt1))

        
    
    clim = np.concatenate([np.expand_dims(h5clim[0][k], axis=0) for k in out_variables], axis=1).astype(np.float32)
    mask1 = ~np.isnan(clim) * 1
    clim = torch.from_numpy((clim - out_mean) / out_std)
    mask = ~torch.isnan(clim) * 1
    clim = torch.nan_to_num(clim)
    
    mult = mult
    shape = gts[0].shape
    number = 5000
    eninit_data = torch.zeros((number,1, shape[-3], shape[-2], shape[-1]))
    seq_pred = torch.zeros((number,1 + prediction_length, len(out_variables), shape[-2], shape[-1]))
    seq_da = torch.zeros((number,1 + prediction_length, len(out_variables), shape[-2], shape[-1]))
    seq_real = torch.zeros((1 + prediction_length, shape[-3], shape[-2], shape[-1]))
    seq_rmse = torch.zeros((1 + prediction_length, len(out_variables)))
    seq_acc = torch.zeros((1 + prediction_length, len(out_variables)))
    seq_activity = torch.zeros((1 + prediction_length, len(out_variables)))
    # standardize
    

    for n in range(number): 
        eninit_data[n] = torch.as_tensor(gts[0])
        
#        print((init_data[:,len(in_variables)-len(out_variables):]).shape)
        if n % 2==0:
            x = torch.as_tensor(np.random.normal(0, 0.1, [1,1,120,240]))

            eninit_data[n,:,len(in_variables)-len(out_variables):] = eninit_data[n,:,len(in_variables)-len(out_variables):] + x
        else:
            eninit_data[n,:,len(in_variables)-len(out_variables):] = eninit_data[n,:,len(in_variables)-len(out_variables):] - x
 
    R = np.eye(28800) * 0.2
    # K = np.load(f"/hpcfs/fhome/yangjh16/CoastNet/test/K.npy")
    with torch.no_grad():
        for i in range(1 + prediction_length):
            for n in range(number):
            # 从ic开始
                if i == 0:  # start of sequence
                    seq_real[i:i + 1] = eninit_data[n,:]
                    ocean = eninit_data[n,:,len(in_variables)-len(out_variables):]
                    seq_pred[n,i:i + 1] = ocean
                    seq_da[n,i:i + 1] = ocean
                    
                else:
                    seq_real[i:i + 1] = torch.as_tensor(gts[i])
                    # switch to the 24-hour model if the forecast time is 24 hours, 48 hours, ..., 24*N hours
                    if ((10 // dt)  > 0) and (i % (10 // dt)) == 0:
                        # Call the model pretrained for 24 hours forecast
                        atmos = seq_real[i-10//dt:i-10//dt+1, :len(in_variables)-len(out_variables)]
                        ocean = seq_real[i-10//dt:i-10//dt+1, len(in_variables)-len(out_variables):]
                        init_data = torch.concat([atmos, seq_da[n,i-10//dt:i-10//dt+1]], dim=1)
                        seq_pred[n,i:i+1] = module(init_data.to(device, dtype=torch.float32),
                                                 mask.to(device, dtype=torch.float32),
                                                 torch.from_numpy(10 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                                 in_variables,
                                                 out_variables).cpu().detach()
                    # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
                    elif ((5 // dt)  > 0) and (i % (5 // dt)) == 0:
                        # Switch the input back to the stored input
                        atmos = seq_real[i-5//dt:i-5//dt+1, :len(in_variables)-len(out_variables)]
                        ocean = seq_real[i-5//dt:i-5//dt+1, len(in_variables)-len(out_variables):]
                        init_data = torch.concat([atmos, seq_da[n,i-5//dt:i-5//dt+1]], dim=1)
                        seq_pred[n,i:i+1] = module(init_data.to(device, dtype=torch.float32),
                                                 mask.to(device, dtype=torch.float32),
                                                 torch.from_numpy(5 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                                 in_variables,
                                                 out_variables).cpu().detach()
                    # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
                    elif ((3 // dt)  > 0) and (i % (3 // dt)) == 0:
                        # Switch the input back to the stored input
                        atmos = seq_real[i-3//dt:i-3//dt+1, :len(in_variables)-len(out_variables)]
                        ocean = seq_real[i-3//dt:i-3//dt+1, len(in_variables)-len(out_variables):]
                        init_data = torch.concat([atmos, seq_da[n,i-3//dt:i-3//dt+1]], dim=1)
                        seq_pred[n,i:i+1] = module(init_data.to(device, dtype=torch.float32),
                                                 mask.to(device, dtype=torch.float32),
                                                 torch.from_numpy(3 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                                 in_variables,
                                                 out_variables).cpu().detach()
                    # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
                    elif ((2 // dt)  > 0) and (i % (2 // dt)) == 0:
                        # Switch the input back to the stored input
                        atmos = seq_real[i-2//dt:i-2//dt+1, :len(in_variables)-len(out_variables)]
                        ocean = seq_real[i-2//dt:i-2//dt+1, len(in_variables)-len(out_variables):]
                        init_data = torch.concat([atmos, seq_da[n,i-2//dt:i-2//dt+1]], dim=1)
                        seq_pred[n,i:i+1] = module(init_data.to(device, dtype=torch.float32),
                                                 mask.to(device, dtype=torch.float32),
                                                 torch.from_numpy(2 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                                 in_variables,
                                                 out_variables).cpu().detach()
                    # switch to the 1-hour model
                    elif ((1 // dt)  > 0) and (i % (1 // dt)) == 0:
                        # Switch the input back to the stored input
                        atmos = seq_real[i-1//dt:i-1//dt+1, :len(in_variables)-len(out_variables)]
                        ocean = seq_real[i-1//dt:i-1//dt+1, len(in_variables)-len(out_variables):]
                        init_data = torch.concat([atmos, seq_da[n,i-1//dt:i-1//dt+1]], dim=1)
    
                        seq_pred[n,i:i+1] = module(init_data.to(device, dtype=torch.float32),
                                                 mask.to(device, dtype=torch.float32),
                                                 torch.from_numpy(1 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                                 in_variables,
                                                 out_variables).cpu().detach()
                     
                g = torch.mean(seq_pred[:,i:i+1], 0)       
                seq_rmse[i:i + 1] = mult * weighted_rmse_torch(mask * ocean, mask * g, mask)
                seq_acc[i:i + 1] = weighted_acc_torch(mask * (ocean - clim), 
                                                      mask * (g - clim))
                seq_activity[i:i + 1] = activity_torch(g, clim, mult)
            # 加入da
            if i!=0:
                pred_en = seq_pred[:,i,0]
                pred_en = (pred_en*out_std+out_mean).cpu().detach().to(torch.float32).numpy()
                pred_en = pred_en[0]
                
                pred_en1 = np.zeros([number,120*240])
                for h in range(number):
                    pred_en[h] = pred_en[h]*mask1
                    pred_en1[h] = pred_en[h].flatten()
                # realdata = gts1[i]
                # realdata = realdata[0,0,:,:].flatten()
                realdata = seq_real[i, len(in_variables)-len(out_variables):]
                realdata = (realdata*out_std+out_mean).cpu().detach().to(torch.float32).numpy()
                realdata = realdata[0,0,:,:].flatten()
                da_en = enkf(pred_en1,realdata,R,modelP)

                da_en1 = np.zeros([number,120,240])
                for h in range(number):
                    da_en1[h] = da_en[h].reshape(120,240)
    
                
                da_en = torch.from_numpy((da_en1 - out_mean) / out_std)
                
                seq_da[:,i,0,:,:] = da_en
            else:
                seq_da[:,i,0,:,:] = seq_pred[:,i,0,:,:]
            
            
            
    seq_pred = seq_pred.cpu().detach().to(torch.float32).numpy()
    seq_real = seq_real.cpu().detach().to(torch.float32).numpy()
    seq_rmse = seq_rmse.cpu().detach().to(torch.float32).numpy()
    seq_acc = seq_acc.cpu().detach().to(torch.float32).numpy()
    seq_activity = seq_activity.cpu().detach().to(torch.float32).numpy()
    # sp = np.expand_dims(np.mean(seq_pred, 0),0) 
    # sp_save = sp[:, len(in_variables)-len(out_variables):]*out_std+out_mean
    # np.save(f"/hpcfs/fhome/yangjh16/CoastNet/adata/results/pred_mean.npy", sp_save)
    # sr = np.expand_dims(np.mean(seq_real, 0),0) 
    # sr_save = sr[:, len(in_variables)-len(out_variables):]*out_std+out_mean
    # np.save(f"/hpcfs/fhome/yangjh16/CoastNet/adata/results/real_mean.npy", sr_save)
    # np.save(f"/hpcfs/fhome/yangjh16/CoastNet/adata/results/pred_en.npy", seq_pred)
    # np.save(f"/hpcfs/fhome/yangjh16/CoastNet/adata/results/real_en.npy", seq_real)    
    # np.save(f"/hpcfs/fhome/yangjh16/CoastNet/adata/results/rmse_en.npy", seq_rmse)
    # np.save(f"/hpcfs/fhome/yangjh16/CoastNet/adata/results/acc_en.npy", seq_acc)
    # np.save(f"/hpcfs/fhome/yangjh16/CoastNet/adata/results/activity_en.npy", seq_activity)
        # return  np.expand_dims(seq_real, 0), \
        #         np.expand_dims(seq_pred, 0), \
        #         np.expand_dims(seq_rmse, 0), \
        #         np.expand_dims(seq_acc, 0), \
        #         np.expand_dims(seq_activity, 0)

    return  np.expand_dims(seq_real,0), \
            np.expand_dims(seq_pred,0), \
            np.expand_dims(seq_rmse,0), \
            np.expand_dims(seq_acc,0), \
            np.expand_dims(seq_activity,0)
                
def cycle_medium_inference(ic, 
                            mean,
                            std,
                            module,
                            dt, 
                            prediction_length,
                            analysis_np,
                            h5era5,
                            h5clim, 
                            mult, 
                            variables, 
                            device):
    ic = int(ic)
    n_samples_per_shards = [h5era5[i][variables[0]].shape[0] for i in range(len(h5era5))]
    prediction_length = int(prediction_length) // dt
    idx_targets = [ic + int(dt * i) for i in range(1 + prediction_length)]
    era5s, clims = [], []
    for i in range(len(idx_targets)):
        shard_idx, local_idx = reindex(idx_targets[i], n_samples_per_shards)
        era5 = np.concatenate([h5era5[shard_idx % len(h5era5)][k][local_idx] 
                                        for k in variables], axis=0).astype(np.float32)
        era5s.append(torch.from_numpy((era5 - mean) /std))
        clim = np.concatenate([h5clim[shard_idx % len(h5clim)][k][local_idx] 
                                        for k in variables], axis=0).astype(np.float32)
        clims.append(torch.from_numpy((clim - mean) /std))
     
    mult = mult
    # valid_data_all, times, lon, lat = read_data(path, raw_variables, variables, resolution, years, mean, std, ic, prediction_length, dt)
    shape = era5s[0].shape
    seq_pred = torch.zeros((1 + prediction_length, shape[-3], shape[-2], shape[-1]))
    seq_real = torch.zeros((1 + prediction_length, shape[-3], shape[-2], shape[-1]))
    seq_rmse = torch.zeros((1 + prediction_length, shape[-3]))
    seq_acc = torch.zeros((1 + prediction_length, shape[-3]))
    seq_activity = torch.zeros((1 + prediction_length, shape[-3]))
    # standardize
    init_data = torch.as_tensor(analysis_np[ic // dt])
    
    with torch.no_grad():
        for i in range(1 + prediction_length):
            # 从ic开始
            if i == 0:  # start of sequence
                seq_real[i:i + 1] = torch.as_tensor(era5s[i])
                seq_pred[i:i + 1] = init_data
            else:
                seq_real[i:i + 1] = torch.as_tensor(era5s[i])
                # switch to the 24-hour model if the forecast time is 24 hours, 48 hours, ..., 24*N hours
                if ((24 // dt)  > 0) and (i % (24 // dt)) == 0:
                    # Call the model pretrained for 24 hours forecast
                    seq_pred[i:i+1] = module(seq_pred[i-24//dt:i-24//dt+1].to(device, dtype=torch.float32),
                                            torch.from_numpy(24 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                            variables,
                                            variables).cpu().detach()
                # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
                elif ((12 // dt)  > 0) and (i % (12 // dt)) == 0:
                    # Switch the input back to the stored input
                    seq_pred[i:i+1] = module(seq_pred[i-12//dt:i-12//dt+1].to(device, dtype=torch.float32),
                                            torch.from_numpy(12 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                            variables,
                                            variables).cpu().detach()
                # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
                elif ((6 // dt)  > 0) and (i % (6 // dt)) == 0:
                    # Switch the input back to the stored input
                    seq_pred[i:i+1] = module(seq_pred[i-6//dt:i-6//dt+1].to(device, dtype=torch.float32),
                                            torch.from_numpy(6 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                            variables,
                                            variables).cpu().detach()
                # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
                elif ((3 // dt)  > 0) and (i % (3 // dt)) == 0:
                    # Switch the input back to the stored input
                    seq_pred[i:i+1] = module(seq_pred[i-3//dt:i-2//dt+1].to(device, dtype=torch.float32),
                                            torch.from_numpy(3 * np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                            variables,
                                            variables).cpu().detach()
                # switch to the 1-hour model
                elif ((1 // dt)  > 0) and (i % (1 // dt)) == 0:
                    # Switch the input back to the stored input
                    seq_pred[i:i+1] = module(seq_pred[i-1:i].to(device, dtype=torch.float32),
                                            torch.from_numpy(np.ones((1, 1))).to(device, dtype=torch.float32) / 100,
                                            variables,
                                            variables).cpu().detach()
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clims[i], 
                                                  seq_pred[i:i+1] - clims[i])
            seq_activity[i:i + 1] = activity_torch(seq_pred[i:i+1], clims[i], mult)

        seq_pred = seq_pred.cpu().detach().to(torch.float32).numpy()
        seq_real = seq_real.cpu().detach().to(torch.float32).numpy()
        seq_rmse = seq_rmse.cpu().detach().to(torch.float32).numpy()
        seq_acc = seq_acc.cpu().detach().to(torch.float32).numpy()
        seq_activity = seq_activity.cpu().detach().to(torch.float32).numpy()

        return  np.expand_dims(seq_real, 0), \
                np.expand_dims(seq_pred, 0), \
                np.expand_dims(seq_rmse, 0), \
                np.expand_dims(seq_acc, 0), \
                np.expand_dims(seq_activity, 0)
