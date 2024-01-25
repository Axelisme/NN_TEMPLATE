import os
import yaml
import time
import wandb
from typing import Dict, Any
from argparse import Namespace
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm.auto import tqdm

from util.io import show
from util.utility import init, create_instance
from modules.trainer import Trainer
from modules.valider import Valider
from modules.ckpt import CheckPointManager
from custom.evaluator.Loss2evaluator import LossScore
from global_vars import RESULT_DIR


def recursive_overwrite(conf: Dict[str,Dict|Any], new_conf: Dict[str,Dict|Any]):
    for key, value in new_conf.items():
        if isinstance(value, dict):
            if key not in conf:
                print(f"something wrong with {key}")
                raise ValueError
            recursive_overwrite(conf[key], value)
        elif conf[key] == value:
            show(f'[Runner] Keep {key} as {value}.')
        else:
            print(f'[Runner] overwrite {key} from {conf[key]} to {value}')
            conf[key] = value


class Runner:
    def __init__(self, args: Namespace, overwrite_rc: Dict = {}, start_method:str = 'forkserver'):
        self.args = args

        # set slient or not
        show.set_slient(args.slient)

        # check mode
        assert args.mode in ['train', 'evaluate'], f'Unknown mode: {args.mode}'
        show(f'[RUNNER] Initialize runner with mode: {args.mode}.')

        # create checkpoint manager
        if args.ckpt_dir is not None: # use specified ckpt_dir
            ckpt_dir = args.ckpt_dir
        elif args.ckpt is not None:   # use ckpt_dir of ckpt
            ckpt_dir = os.path.dirname(args.ckpt)
        else:                         # use default ckpt_dir
            assert args.name is not None, 'Please specify the name of this run.'
            ckpt_dir = os.path.join(RESULT_DIR, args.name)
        self.ckpt_manager = CheckPointManager(ckpt_dir)

        # load modelrc
        if args.ckpt:  # provide ckpt
            self.modelrc = self.ckpt_manager.load_config(args.ckpt, 'modelrc')
        else:  # load modelrc from file
            show(f'[RUNNER] Load modelrc from {args.modelrc}.')
            with open(args.modelrc, 'r') as f:
                self.modelrc = yaml.load(f, Loader=yaml.FullLoader)

        # load taskrc
        if args.resume or args.mode == 'evaluate':
            self.taskrc = self.ckpt_manager.load_config(args.ckpt, 'taskrc')
        else:
            show(f'[RUNNER] Load taskrc from {args.taskrc}.')
            with open(args.taskrc, 'r') as f:
                self.taskrc = yaml.load(f, Loader=yaml.FullLoader)

        # override modelrc and taskrc
        recursive_overwrite(self.modelrc, overwrite_rc.get('modelrc', {}))
        recursive_overwrite(self.taskrc, overwrite_rc.get('taskrc', {}))

        # dump config
        self.ckpt_manager.dump_config(
            {
                'modelrc': self.modelrc,
                'taskrc': self.taskrc,
                'args': vars(args)
            },
            file_name=f"train_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
        )

        # set ckpt conf, such as save_mod, check_metrics, etc.
        self.ckpt_manager.set(self.taskrc['ckpt'])

        # set random seed
        runner_conf = self.taskrc['runner']
        init(runner_conf['seed'], start_method=start_method)
        show(f'[RUNNER] Random seed: {runner_conf["seed"]}.')


    def _get_model(self):
        model_select = self.modelrc['select']
        show(f"[RUNNER] Model: {model_select}.")
        return create_instance(self.modelrc[model_select])


    def _get_criterion(self):
        loss_select = self.taskrc['arch']['loss']['select']
        show(f"[RUNNER] Loss function: {loss_select}.")
        return create_instance(self.taskrc['arch']['loss'][loss_select])


    def _get_modules_for_train(self, parameters):
        arch_conf = self.taskrc['arch']

        # select optimizer
        optim_select = arch_conf['optimizer']['select']
        show(f"[RUNNER] Optimizer: {optim_select}.")
        optimizer = create_instance(arch_conf['optimizer'][optim_select], params=parameters)

        # select scheduler
        sched_select = arch_conf['scheduler']['select']
        show(f"[RUNNER] Scheduler: {sched_select}.")
        scheduler = create_instance(arch_conf['scheduler'][sched_select], optimizer=optimizer)

        return optimizer, scheduler


    def _get_modules_for_eval(self, criterion):
        arch_conf = self.taskrc['arch']

        # select evaluator
        metrics = {}
        for i, name in enumerate(arch_conf['metric']['select'], start=1):
            show(f"[RUNNER] Metric {i}: {name}.")
            # use importlib to avoid weird bug of 'BinnedAveragePrecision' not found
            metrics[name] = create_instance(arch_conf['metric'][name])
        if arch_conf['metric']['use_loss']:
            loss_name = arch_conf['metric']['loss_name']
            show(f"[RUNNER] Metric {i+1}: {loss_name}.")
            metrics[loss_name] = LossScore(criterion)

        return MetricCollection(metrics)

    @staticmethod
    def _get_dataloader(datasets_conf:Dict, loader_conf:Dict, mode:str):
        dataset = loader_conf['dataset']
        show(f"[RUNNER] {mode.capitalize()} dataset: {dataset}.")
        dataset = create_instance(datasets_conf[dataset])
        return DataLoader(dataset, **loader_conf['kwargs'])


    def _show_train_result(self, step: int, lr: float, train_result: Dict[str,float]):
        """Print result of training."""
        # print result
        show(f'Step: ({step} / {self.taskrc["runner"]["total_steps"]})')
        show(f'Learning rate: {lr:0.3e}')
        show("Train result:")
        for name, evaluator in train_result.items():
            show(f'\t{name}: {evaluator:0.4f}')


    def _show_valid_result(self, step: int, valid_result: Dict[str,float]):
        """Print result of validation."""
        # print result
        show(f'Step: ({step} / {self.taskrc["runner"]["total_steps"]})')
        show("Valid result:")
        for name, evaluator in valid_result.items():
            show(f'\t{name}: {evaluator:0.4f}')


    def train(self):
        # initialize W&B
        if self.args.WandB:
            show(f'[RUNNER] Initialize W&B to record training.')
            if self.args.resume:
                wandb.init(project=self.taskrc['WandB']['project'], resume=True)
            else:
                wandb_conf = self.taskrc['WandB']
                assert self.args.name is not None, 'Please specify the name of this run.'
                wandb.init(
                    project=wandb_conf['project'],
                    name=self.args.name,
                    reinit=True,
                    **wandb_conf['init_kwargs']
                )

        # load models
        model = self._get_model()
        criterion = self._get_criterion()
        optimizer, scheduler = self._get_modules_for_train(model.parameters())
        metrics = self._get_modules_for_eval(criterion)

        # load model from checkpoint if have
        if self.args.ckpt is not None:
            # first move model to device to avoid bug of 'device mismatch'
            model.to(self.args.device)
            param = self.ckpt_manager.load_param(self.args.ckpt, 'model')
            model.load_state_dict(param)
            if self.args.resume:
                optimizer.load_state_dict(self.ckpt_manager.load_param(self.args.ckpt, 'optimizer'))
                scheduler.load_state_dict(self.ckpt_manager.load_param(self.args.ckpt, 'scheduler'))

        # watch model params in wandb if needed
        if self.args.WandB and self.taskrc['WandB']['track_params']:
            wandb.watch(models=model, criterion=criterion, **self.taskrc['WandB']['watch_kwargs'])

        # prepare dataset and dataloader
        datasets_conf = self.taskrc['data']['datasets']
        loader_conf = self.taskrc['data']['dataloaders']
        train_loader = self._get_dataloader(datasets_conf, loader_conf[self.args.train_loader], 'train')
        valid_loader = self._get_dataloader(datasets_conf, loader_conf[self.args.valid_loader], 'valid')

        # create trainer and valider
        runner_conf = self.taskrc['runner']
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.args.device,
            slient=self.args.slient,
            **runner_conf['trainer_kwargs']
        )
        valider = Valider(
            model=model,
            valid_loader=valid_loader,
            metrics=metrics,
            device=self.args.device,
            slient=self.args.slient,
            **runner_conf['valider_kwargs']
        )

        # setup progress bar and other things
        pbar = tqdm(total=runner_conf['total_steps'], desc='Overall', dynamic_ncols=True, disable=self.args.slient)
        if self.args.resume:
            meta_conf = self.ckpt_manager.load_config(self.args.ckpt, 'meta')
            start_step = meta_conf['step'] + 1
            best_step = meta_conf['best_step']
            best_result = meta_conf['best_result']
            show(f'[RUNNER] Resume training from step {start_step}.')
            pbar.update(start_step)
        else:
            start_step = 1
            best_step = 0
            best_result = {}
            show(f'[RUNNER] Start training from scratch.')

        # start training
        for step in range(start_step, runner_conf['total_steps'] + 1):
            # train one step or one epoch
            if runner_conf['step_mod'] == 'grad':
                trainer.one_step()
            elif runner_conf['step_mod'] == 'epoch':
                trainer.one_epoch()
            else:
                raise ValueError(f'Unknown step_mod: {runner_conf["step_mod"]}')

            pbar.update()

            # show result per log_freq steps
            if step % runner_conf['log_freq'] == 0 or step == runner_conf['total_steps']:
                train_result = trainer.pop_result()
                pbar.refresh() # refresh tqdm bar to show the latest result
                self._show_train_result(step, trainer.lr, train_result)
                show('-' * 50)
                if self.args.WandB:
                    wandb.log(train_result, step=step, commit=False)

            save_params = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_confs = {
                'modelrc': self.modelrc,
                'taskrc': self.taskrc,
                'meta': {'step': step, 'best_step': best_step, 'best_result': best_result}
            }
            print_slash = False


            # validate a epoch per dev_freq steps
            if step % runner_conf['valid_freq'] == 0 or step == runner_conf['total_steps']:
                valider.one_epoch()
                current_result = valider.pop_result()
                pbar.refresh() # refresh tqdm bar to show the latest result
                self._show_valid_result(step, current_result)
                print_slash = True
                if self.args.WandB:
                    wandb.log(current_result, step=step, commit=False)
                # save model if better
                if not self.args.disable_save:
                    if self.ckpt_manager.check_better(current_result, best_result):
                        best_step = step
                        best_result = current_result
                        self.ckpt_manager.save_model(
                            ckpt_name='valid-best.pth',
                            params=save_params,
                            confs=save_confs,
                            overwrite=True
                        )


            if step % runner_conf['save_freq'] == 0 and not self.args.disable_save:
                self.ckpt_manager.save_model(
                    ckpt_name=f'step-{step}.pth',
                    params=save_params,
                    confs=save_confs,
                    overwrite=self.args.overwrite
                )
                self.ckpt_manager.clean_old_ckpt('step-*.pth')
                print_slash = True

            if print_slash:
                show('-' * 50)

            # log result to wandb
            if self.args.WandB:
                wandb.log({'lr':trainer.lr}, step=step, commit=True)
        show(f'[RUNNER] Training finished.')
        show(f'[RUNNER] Best result:')
        self._show_valid_result(best_step, best_result)
        show('-' * 50)

        pbar.close()
        trainer.close()
        valider.close()


    def evaluate(self):
        # load models
        model = self._get_model()
        criterion = self._get_criterion()
        metrics = self._get_modules_for_eval(criterion)

        # load model from checkpoint if have
        param = self.ckpt_manager.load_param(self.args.ckpt, 'model')
        model.load_state_dict(param)

        # prepare dataset and dataloader
        loader_conf = self.taskrc['data']['dataloaders']
        valid_loader = self._get_dataloader(self.taskrc['data']['datasets'], loader_conf[self.args.valid_loader], 'valid')

        # create valider
        runner_conf = self.taskrc['runner']
        valider = Valider(
            model=model,
            valid_loader=valid_loader,
            metrics=metrics,
            device=self.args.device,
            slient=self.args.slient,
            **runner_conf['valider_kwargs']
        )

        # start validating
        valider.one_epoch()
        current_result = valider.pop_result()

        # show result
        show(f'[RUNNER] Evaluation finished.')
        self._show_valid_result(0, current_result)

        valider.close()


    def run(self):
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'evaluate':
            assert self.args.ckpt is not None, 'Please specify the checkpoint to evaluate.'
            self.evaluate()
        else: # should not happen
            raise ValueError(f'Unknown mode: {self.args.mode}')