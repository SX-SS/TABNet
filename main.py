import os
import argparse
import datetime
import random
import time
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import util.misc as utils
from data import build
# from engines import train_one_epoch
from inference import infer, evaluate
from models import build_model


def get_args_parser():#获取网络参数
    tasks = {
        'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 4}
    }
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)    #用于优化器更新权重时的步长大小
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)   #权重衰减系数，用于防止过拟合的正则化项
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=500, type=int)   #学习率下降的周期，表示经过多少个轮次，学习率需要衰减
    parser.add_argument('--tasks', default=tasks, type=dict)
    parser.add_argument('--model', default='MSCMR', required=False)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,   #指定预训练模型权重的路径，如果设置了此参数，只有模型中的掩码头（mask head）部分会被训练，其余部分的参数将保持不变（被冻结），这通常用于微调（fine-tuning）已有的预训练模型
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--in_channels', default=1, type=int)

    # Puzzle Mix
    parser.add_argument('--in_batch', type=str2bool, default=False, help='whether to use different lambdas in batch')   #是否在批处理中使用不同的 lambda 参数，类型为布尔值（str2bool），默认值为 False。作用：决定在一个批次中是否使用不同的混合权重（lambda）
    parser.add_argument('--mixup_alpha', type=float, default=0.5, help='alpha parameter for mixup')   #控制 Mixup 的混合比例
    parser.add_argument('--box', type=str2bool, default=False, help='true for CutMix')   #决定是否启用 CutMix
    parser.add_argument('--graph', type=str2bool, default=True, help='true for PuzzleMix')   #决定是否启用 PuzzleMix
    parser.add_argument('--neigh_size', type=int, default=4,   #计算图像区域之间距离的邻域大小，控制在图像分块操作中用于计算邻近块之间距离的大小
                        help='neighbor size for computing distance beteeen image regions')
    parser.add_argument('--n_labels', type=int, default=3, help='label space size')   #用于指定在数据增强操作中所考虑的标签数量


    parser.add_argument('--transport', type=str2bool, default=True, help='whether to use transport')   #是否使用传输
    parser.add_argument('--t_eps', type=float, default=0.8, help='transport cost coefficient')   #传输成本系数
    parser.add_argument('--t_size', type=int, default=4,    #传输分辨率
                        help='transport resolution. -1 for using the same resolution with graphcut')

    parser.add_argument('--adv_eps', type=float, default=10.0, help='adversarial training ball')    #对抗性训练球大小
    parser.add_argument('--adv_p', type=float, default=0.0, help='adversarial training probability')   #对抗性训练概率

    parser.add_argument('--clean_lam', type=float, default=0.0, help='clean input regularization')   #控制数据清洁度的正则化力度
    parser.add_argument('--mp', type=int, default=8, help='multi-process for graphcut (CPU)')   #用于图割计算的多进程数

    # * Loss coefficients
    parser.add_argument('--multiDice_loss_coef', default=0, type=float)   #控制多Dice损失项在总损失中的权重。Dice损失通常用于衡量预测和真实标签之间的重叠情况，常用于分割任务中
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)   #控制交叉熵损失项在总损失中的权重。交叉熵损失是分类问题中常用的损失函数，衡量预测类别分布与真实类别分布之间的差异
    parser.add_argument('--Rv', default=1, type=float)
    parser.add_argument('--Lv', default=1, type=float)
    parser.add_argument('--Myo', default=1, type=float)
    parser.add_argument('--Avg', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset', default='MSCMR_dataset', type=str, help='multi-sequence CMR segmentation dataset')   #指定的数据集，用于多序列心脏磁共振（CMR）分割任务
    # set your outputdir 
    parser.add_argument('--output_dir', default='output/exp2-test/', help='path where to save, empty for no saving')  #指定模型训练过程中输出文件的保存路径，若默认为空字符串，则不输出
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')   #指定训练或测试的设备
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')    #指定要使用的GPU ID,默认为0表示使用第一块GPU，多个GPU可以使用逗号分隔的字符串表示，如''0,1,2''
    parser.add_argument('--seed', default=42, type=int)    #指定随机种子，默认为 42。设置随机种子有助于实验的可重复性,相同种子生成的随机数序列是确定性的
    parser.add_argument('--resume', default='', help='resume from checkpoint')   #指定从某个检查点恢复训练的路径。默认为空字符串，表示从头开始训练
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')   #指定训练开始的轮次
    parser.add_argument('--eval', default=True, action='store_true')   #指定是否进入评估模式。action='store_true' 表示该参数为布尔值，存在时为 True，用于切换模型为评估模式而非训练模式
    parser.add_argument('--num_workers', default=4, type=int)   #指定数据加载器在数据预处理时使用的线程数，默认为 4。更多的线程数可以加快数据加载速度，但会占用更多的系统资源。

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')   #指定分布式训练的进程总数，默认为 1。表示单节点或单GPU训练。如果要在多个节点或多个GPU上进行训练，需要设置为大于 1 的值，这个数值通常等于参与训练的所有设备（如GPU）的总数。
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')   #用于设置分布式训练的初始化URL，默认为 'env://'。
    return parser

def str2bool(v):
    if isinstance(v, bool):   #将输入参数转换为布尔类型
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    if not args.eval:    #当程序不是处于评估模式（args.eval 为 False）时，创建一个 SummaryWriter 对象，用于 TensorBoard 的日志记录，将结果输出到指定的目录
        writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    # ------------------------------
    # [[[[0.5000]]]], [[[[0.5000]]]]
    # ------------------------------
    args.mean = torch.tensor([0.5], dtype=torch.float32).reshape(1, 1, 1, 1).cuda()   #创建均值和标准差的张量，用于对输入数据进行归一化，均值和标准差都设为 0.5，并通过 reshape 方法调整张量形状为 (1, 1, 1, 1)。
    args.std = torch.tensor([0.5], dtype=torch.float32).reshape(1, 1, 1, 1).cuda()   #这两个张量被发送到 GPU（通过 .cuda()），以确保它们与数据处理的设备一致

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True   #cuda设置
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    # fix the seed for reproducibility    确保实验的可复现性，通过固定随机数种子来减少由于随机性导致的结果差异
    seed = args.seed + utils.get_rank()   #获取初始种子值 ，utils.get_rank() 通常用于获取当前进程在分布式训练中的排名（进程 ID）。这样在分布式训练时，不同进程使用不同的种子，避免进程间的随机性冲突。
    torch.manual_seed(seed)   #用 torch.manual_seed(seed) 设置 PyTorch 的随机种子，以确保所有的 PyTorch 随机数生成器（如模型初始化、数据打乱等）都基于相同的种子值，从而使随机操作可复现
    np.random.seed(seed)   #通过 np.random.seed(seed) 设置 NumPy 的随机种子，使得所有 NumPy 的随机数生成（如数据预处理或其他需要随机性的 NumPy 操作）也具有可复现性
    random.seed(seed)   #使用 random.seed(seed) 设置 Python 标准库中 random 模块的随机种子，用于确保所有由 random 模块生成的随机数（如随机选择、打乱等）是可复现的

    # losses = ['CrossEntropy', 'Rv', 'Lv', 'Myo', 'Avg']
    losses = ['CrossEntropy']   # 只使用了交叉熵损失（CrossEntropy）作为训练过程中使用的损失函数
    model, criterion, postprocessors, visualizer = build_model(args, losses)    #调用 build_model 函数，根据传入的参数 args 和指定的损失函数 losses 构建模型
    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)   #计算模型中所有可训练参数的总数并打印，if p.requires_grad 通常用于检查张量 p 是否需要梯度计算
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]    #构建优化器参数字典 param_dicts，包含所有需要更新的参数
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)   #使用 Adam 优化器，并根据传入的学习率 (args.lr) 和权重衰减 (args.weight_decay) 初始化
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)   #使用 StepLR 学习率调度器，每隔一定步数（args.lr_drop）降低学习率

    print('Building training dataset...')
    dataset_train_dict = build(image_set='train', args=args)   #通过调用 build 函数构建训练数据集
    num_train = [len(v) for v in dataset_train_dict.values()]
    print('Number of training images: {}'.format(sum(num_train)))   #计算并打印训练数据的总图像数量

    print('Building validation dataset...')
    dataset_val_dict = build(image_set='val', args=args)   #构建验证集
    num_val = [len(v) for v in dataset_val_dict.values()]
    print('Number of validation images: {}'.format(sum(num_val)))

    sampler_train_dict = {k: torch.utils.data.RandomSampler(v) for k, v in dataset_train_dict.items()}   #sampler_train_dict: 为每个训练数据集创建一个 RandomSampler，这会随机打乱数据，以确保每个训练周期的数据顺序是不同的
    sampler_val_dict = {k: torch.utils.data.SequentialSampler(v) for k, v in dataset_val_dict.items()}

    batch_sampler_train = {
        k: torch.utils.data.BatchSampler(v, args.batch_size, drop_last=True) for k, v in sampler_train_dict.items()
    }   #batch_sampler_train: 为每个训练采样器创建一个 BatchSampler，按指定的批大小（args.batch_size）将数据分批处理。设置 drop_last=True 意味着如果最后一个批次的大小小于指定的批大小，则会丢弃这个批次，这样可以保证每个批次的大小一致
    dataloader_train_dict = { #dataloader_train_dict: 为每个训练数据集和批采样器创建数据加载器 DataLoader
        k: DataLoader(v1, batch_sampler=v2, collate_fn=utils.collate_fn, num_workers=args.num_workers)  #使用 batch_sampler 来指定如何批量采样数据，并使用 collate_fn=utils.collate_fn 自定义数据的整理方式（例如，处理不同大小的图像、填充等）
        for (k, v1), v2 in zip(dataset_train_dict.items(), batch_sampler_train.values())#num_workers=args.num_workers 指定了用于数据加载的子进程数量，多进程可以加速数据加载
    }
    dataloader_val_dict = {
        k: DataLoader(v1, args.batch_size, sampler=v2, drop_last=False, collate_fn=utils.collate_fn,
                      num_workers=args.num_workers)
        for (k, v1), v2 in zip(dataset_val_dict.items(), sampler_val_dict.values())
    }

    if args.frozen_weights is not None:   #加载冻结权重（预训练模型）
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.whst.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:   #加载检查点（用于恢复训练）
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:   #评估模式
        infer(model, criterion, dataloader_train_dict, device)
        return

    print("Start training")
    start_time = time.time()
    best_dice = None
    for epoch in range(args.start_epoch, args.epochs):
        print('-' * 40)
        train_stats = train_one_epoch(model, criterion, dataloader_train_dict, optimizer, device, epoch, args, writer)

        # lr_scheduler
        lr_scheduler.step()   #每轮训练结束后，调用学习率调度器 lr_scheduler.step() 来调整学习率

        # evaluate
        losses = ['CrossEntropy', 'Rv', 'Lv', 'Myo', 'Avg']    #根据预定义的损失项（如 CrossEntropy, Rv, Lv, Myo, Avg），重新构建损失函数 criterion
        _, criterion, _, _ = build_model(args, losses)

        test_stats = evaluate(   #调用 evaluate 函数对验证集进行评估，得到测试统计信息 test_stats
            model, criterion, postprocessors, dataloader_val_dict, device, args.output_dir, visualizer, epoch, writer
        )

        # save checkpoint for high dice score
        dice_score = test_stats["Avg"]   #读取验证结果中的 dice score（Avg ），用于判断模型的性能
        print("dice score:", dice_score)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']   #初始化检查点路径 checkpoint_paths，包含默认的 checkpoint.pth。
            if best_dice is None or dice_score > best_dice:
                best_dice = dice_score
                print("Update best model!")
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')

            # You can change the threshold
            if dice_score > 0.80:
                print("Update high dice score model!")
                file_name = str(dice_score)[0:6] + '_' + str(epoch) + '_checkpoint.pth'
                checkpoint_paths.append(output_dir / file_name)

            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:   #在学习率下降前或每 100 个轮次保存一次检查点
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time   #计算并打印整个训练过程的总时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
