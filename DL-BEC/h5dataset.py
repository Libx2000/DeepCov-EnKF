import h5py
import numpy as np
import os

def generate_h5_data(output_path='./data/train_data.h5', m=28800, n=500, num_samples=100, mask_path='./maskP.npy'):
    mask = np.load(mask_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        dset = f.create_dataset("perturbations", 
                                shape=(num_samples, n, m),
                                dtype='float32')
        
        for i in range(num_samples):
            data = np.zeros([n, m])
            for j in range(n):
                data[j] = np.random.randn(m).astype(np.float32)
                data[j] = np.multiply(data[j], mask)
            data -= data.mean(axis=0) 
            dset[i] = data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成训练数据")
    parser.add_argument('--output_path', type=str, default='./data/train_data.h5')
    parser.add_argument('--m', type=int, default=28800)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--mask_path', type=str, default='./maskP.npy')
    args = parser.parse_args()
    generate_h5_data(args.output_path, args.m, args.n, args.num_samples, args.mask_path)
