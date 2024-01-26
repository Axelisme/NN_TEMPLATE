import os
import sys
sys.path.append(os.getcwd())

import argparse

from modules.runner import Runner


def main():
    """Main function."""
    # parse arguments
    parser = argparse.ArgumentParser(description='Training or evaluating a model.')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'evaluate'], required=True, help='mode of this run')

    parser.add_argument('-e', '--ckpt', type=str, help='load existing ckpt')
    parser.add_argument('-n', '--name', type=str, help='name of this run')
    parser.add_argument('-M', '--modelrc', type=str, help='modelrc file, use to train a new model, default to configs/modelrc.yaml')
    parser.add_argument('-T', '--taskrc', type=str, help='taskrc file, default to configs/taskrc.yaml')
    parser.add_argument('-a', '--resume', action='store_true', help='resume training from existing ckpt, default to False')

    parser.add_argument('-o', '--ckpt_dir', type=str, help='checkpoint directory, default to checkpoints parent dir or RESULTS_DIR/{name}')
    parser.add_argument('-t', '--train_loader', type=str, help='train loader want to use, default to "train_loader"')
    parser.add_argument('-v', '--valid_loader', type=str, help='valid loader want to use, default to "devel_loader"')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use, default to "cuda:0"')
    parser.add_argument('--slient', action='store_true', help='slient or not')

    parser.add_argument('--WandB', action='store_true', help='use W&B to log')
    parser.add_argument('--disable_save', action='store_true', help='disable auto save ckpt')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing ckpt or not')
    args = parser.parse_args()

    # check arguments
    if args.ckpt:
        if args.modelrc:
            parser.error('While loading existing ckpt, you cannot specify --modelrc')
        if args.resume:
            if args.taskrc:
                parser.error('While resuming training, you cannot specify --taskrc')
            if args.name:
                parser.error('While resuming training, you cannot specify --name')
    else:
        if args.modelrc is None:
            args.modelrc = 'configs/modelrc.yaml'
        if args.resume:
            parser.error('While resuming training, you must specify --ckpt')

    if args.mode == 'train':
        if args.train_loader is None:
            args.train_loader = 'train_loader'
        if args.valid_loader is None:
            args.valid_loader = 'devel_loader'
        if args.taskrc is None:
            if args.ckpt and not args.resume:
                parser.error('While loading existing ckpt but not resuming training, you must specify --taskrc')
            else:
                args.taskrc = 'configs/taskrc.yaml'
        if args.ckpt is None and args.name is None:
            parser.error('While training a new model, you must specify --name')
    elif args.mode == 'evaluate':
        invalid_flags = ['name', 'modelrc', 'resume', 'ckpt_dir', 'train_loader', 'WandB', 'disable_save', 'overwrite']
        for flag in invalid_flags:
            if getattr(args, flag):
                parser.error(f'While evaluating, you cannot specify --{flag}')
        if args.valid_loader is None:
            parser.error('While evaluating, you must specify --valid_loader')

    # run
    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()