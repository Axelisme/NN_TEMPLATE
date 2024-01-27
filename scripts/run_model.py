import os
import sys
sys.path.append(os.getcwd())

import argparse

from modules.runner import Runner
from global_vars import RESULT_DIR

def main():
    """Main function."""
    # parse arguments
    parser = argparse.ArgumentParser(description='Training or evaluating a model.')
    # common arguments
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'evaluate'], required=True, help='mode of this run')
    parser.add_argument('-e', '--ckpt', type=str, help='load existing checkpoint')
    parser.add_argument('-T', '--taskrc', type=str, help='taskrc file, default to configs/taskrc.yaml')
    parser.add_argument('-o', '--override', type=str, help='override config, like "modelrc.A.B=1,modelrc.C=2"')
    parser.add_argument('--device', type=str, default='cuda', help='device to use, default to "cuda"')
    parser.add_argument('--silent', action='store_true', help='silent or not')
    # training arguments
    parser.add_argument('-n', '--name', type=str, help='name of this run')
    parser.add_argument('-M', '--modelrc', type=str, help='modelrc file, use to train a new model, default to configs/modelrc.yaml')
    parser.add_argument('-a', '--resume', action='store_true', help='resume training from existing ckpt, default to False')
    parser.add_argument('-d', '--ckpt_dir', type=str, help='checkpoint directory, default to checkpoints parent dir or RESULTS_DIR/{name}')
    parser.add_argument('--WandB', action='store_true', help='use W&B to log')
    parser.add_argument('--disable_save', action='store_true', help='disable auto save ckpt')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing ckpt or not')
    # evaluating arguments
    # accept list of loaders
    parser.add_argument('-t', '--valid_loaders', nargs='+', type=str, help='validation loaders, required while evaluating')
    args = parser.parse_args()

    # check arguments
    if args.mode == 'train':
        if args.valid_loaders:
            parser.error('While training, you cannot specify --valid_loader')
        if args.ckpt:
            if args.modelrc:
                parser.error('While loading existing ckpt, you cannot specify --modelrc')
            if args.resume:
                if args.taskrc:
                    parser.error('While resuming training, you cannot specify --taskrc')
                if args.name:
                    parser.error('While resuming training, you cannot specify --name')
            else:
                if args.taskrc is None:
                    parser.error('While loading existing ckpt but not resuming training, you must specify --taskrc')
            if not args.ckpt_dir:
                args.ckpt_dir = os.path.dirname(args.ckpt)
        else:
            if args.resume:
                parser.error('While resuming training, you must specify --ckpt')
            if args.name is None:
                parser.error('While training a new model, you must specify --name')
            if args.modelrc is None:
                args.modelrc = 'configs/modelrc.yaml'
            if args.taskrc is None:
                args.taskrc = 'configs/taskrc.yaml'
            if args.ckpt_dir is None:
                args.ckpt_dir = os.path.join(RESULT_DIR, args.name)
    elif args.mode == 'evaluate':
        must_specify = ['ckpt', 'valid_loaders']
        invalid_flags = ['name', 'modelrc', 'resume', 'WandB', 'disable_save', 'overwrite']
        for flag in must_specify:
            if getattr(args, flag) is None:
                parser.error(f'While evaluating, you must specify --{flag}')
        for flag in invalid_flags:
            if getattr(args, flag):
                parser.error(f'While evaluating, you cannot specify --{flag}')
        if len(args.valid_loaders) != len(set(args.valid_loaders)):
            parser.error('Duplicate loaders in --valid_loader')
        if not args.ckpt_dir:
            args.ckpt_dir = os.path.dirname(args.ckpt)
    else:
        raise ValueError(f'Unknown mode {args.mode}')

    # create override dict
    override_dict = {}
    if args.override:
        for statement in args.override.split(','):
            keys, value = statement.split('=')
            value = value.strip()
            keys = [key.strip() for key in keys.split('.')]
            cur_dict = override_dict
            for key in keys[:-1]:
                if key not in cur_dict:
                    cur_dict[key] = {}
                cur_dict = cur_dict[key]
            cur_dict[keys[-1]] = eval(value)

    # run
    runner = Runner(args, override_dict)
    eval(f'runner.{args.mode}()')


if __name__ == '__main__':
    main()