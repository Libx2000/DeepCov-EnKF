import argparse
import os

def generate_data(args):
    from h5dataset import generate_h5_data
    generate_h5_data(
        output_path=args.output_path,
        m=args.m,
        n=args.n,
        num_samples=args.num_samples,
        mask_path=args.mask_path
    )

def train_model(args):
    from train import train, CovarianceLoss
    train_args = argparse.Namespace(
        data_path=args.data_path,
        m=args.m,
        batch_size=args.batch_size,
        block_size=args.block_size,
        rank=args.rank,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        gpus=args.gpus
    )
    os.makedirs(train_args.save_dir, exist_ok=True)
    criterion = CovarianceLoss(alpha=0.1)
    train(train_args, criterion)

def main():
    parser = argparse.ArgumentParser(description="EnKF协方差矩阵生成模型")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    data_parser = subparsers.add_parser('generate', help='生成训练数据')
    data_parser.add_argument('--output_path', type=str, default='./data/train_data.h5', help='输出HDF5文件路径')
    data_parser.add_argument('--m', type=int, default=28800, help='数据维度')
    data_parser.add_argument('--n', type=int, default=500, help='集合成员数')
    data_parser.add_argument('--num_samples', type=int, default=100, help='样本数量')
    data_parser.add_argument('--mask_path', type=str, default='./maskP.npy', help='掩码文件路径')

    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_path', type=str, required=True, help='HDF5数据文件路径')
    train_parser.add_argument('--m', type=int, required=True, help='数据维度')
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--block_size', type=int, default=512)
    train_parser.add_argument('--rank', type=int, default=10)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--save_dir', type=str, default='./models')
    train_parser.add_argument('--gpus', type=str, default='0', help='使用的GPU编号')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_data(args)
        print(f"数据生成完成，保存至: {args.output_path}")
    elif args.command == 'train':
        train_model(args)
        print(f"训练完成，模型保存至: {args.save_dir}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
