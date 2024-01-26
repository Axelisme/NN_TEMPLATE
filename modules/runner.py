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
from modules.Loss2evaluator import LossScore


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

        # set silent or not
        show.set_silent(args.silent)

        # load modelrc
        if args.ckpt:  # provide ckpt
            self.modelrc = CheckPointManager.load_config(args.ckpt, 'modelrc')
        else:  # load modelrc from file
            show(f'[RUNNER] Load modelrc from {args.modelrc}.')
            with open(args.modelrc, 'r') as f:
                self.modelrc = yaml.load(f, Loader=yaml.FullLoader)

        # load taskrc
        if args.taskrc is None:
            self.taskrc = CheckPointManager.load_config(args.ckpt, 'taskrc')
        else:
            show(f'[RUNNER] Load taskrc from {args.taskrc}.')
            with open(args.taskrc, 'r') as f:
                self.taskrc = yaml.load(f, Loader=yaml.FullLoader)

        # override modelrc and taskrc
        recursive_overwrite(self.modelrc, overwrite_rc.get('modelrc', {}))
        recursive_overwrite(self.taskrc, overwrite_rc.get('taskrc', {}))

        # dump config
        os.makedirs(args.ckpt_dir, exist_ok=True)
        dump_path = os.path.join(args.ckpt_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
        CheckPointManager.dump_config(
            save_conf = {
                'modelrc': self.modelrc,
                'taskrc': self.taskrc,
                'args': vars(self.args)
            },
            path = dump_path
        )

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


    def _get_modules_for_eval(self, criterion = None):
        arch_conf = self.taskrc['arch']

        # select evaluator
        metrics = {}
        for name in arch_conf['metric']['select']:
            metrics[name] = create_instance(arch_conf['metric'][name])
        if criterion is not None:
            loss_name = arch_conf['metric']['loss_name']
            assert loss_name not in metrics, f'Loss name {loss_name} has been used.'
            metrics[loss_name] = LossScore(criterion)

        # show metrics
        if len(metrics) >= 0:
            show("[RUNNER] Metrics:")
            for i, name in enumerate(metrics.keys(), start=1):
                show(f"\t{i}. {name}")
        else:
            show("[RUNNER] No metrics.")

        return MetricCollection(metrics)

    @staticmethod
    def _get_dataloader(datasets_conf:Dict, loader_conf:Dict, mode:str):
        dataset = loader_conf['dataset']
        show(f"\t{mode.capitalize()} dataset: {dataset}.")
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


    def _show_valid_results(self, step: int, valid_results: Dict[str,Dict[str, float]]):
        """Print result of validation."""
        # print result
        show(f'Step: ({step} / {self.taskrc["runner"]["total_steps"]})')
        show("Valid result:")
        for name, valid_result in valid_results.items():
            show(f'{name.capitalize()} :')
            for name, evaluator in valid_result.items():
                show(f'\t{name}: {evaluator:0.4f}')


    def train(self):
        show(f'[RUNNER] Mod: TRAIN')

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
        if self.args.ckpt:
            # first move model to device to avoid bug of 'device mismatch'
            model.to(self.args.device)
            model.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'model'))
            if self.args.resume:
                optimizer.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'optimizer'))
                scheduler.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'scheduler'))

        # watch model params in wandb if needed
        if self.args.WandB and self.taskrc['WandB']['track_params']:
            wandb.watch(models=model, criterion=criterion, **self.taskrc['WandB']['watch_kwargs'])

        # prepare dataset and dataloader
        datasets_conf = self.taskrc['data']['datasets']
        loader_conf = self.taskrc['data']['dataloaders']
        if len(loader_conf['valid_selects']) != len(set(loader_conf['valid_selects'])):
            raise ValueError('Duplicate loaders in taskrc["data"]["loader"]["valid_selects"]')
        show("[RUNNER] Train dataset:")
        train_loader = self._get_dataloader(datasets_conf, loader_conf[loader_conf['train_select']], 'train')
        show("[RUNNER] Valid dataset:")
        valid_loaders = [self._get_dataloader(datasets_conf, loader_conf[name], name) for name in loader_conf['valid_selects']]

        # create trainer and valider
        runner_conf = self.taskrc['runner']
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.args.device,
            silent=self.args.silent,
            **runner_conf['trainer_kwargs']
        )
        valider = Valider(
            model=model,
            metrics=metrics,
            device=self.args.device,
            silent=self.args.silent,
            **runner_conf['valider_kwargs']
        )

        # create checkpoint manager if needed
        if not self.args.disable_save:
            ckpt_manager = CheckPointManager(self.args.ckpt_dir, **self.taskrc['ckpt'])

        # setup progress bar and other things
        pbar = tqdm(total=runner_conf['total_steps'], desc='Overall', dynamic_ncols=True, disable=self.args.silent)
        if self.args.resume: # resume from checkpoint
            meta_conf = CheckPointManager.load_config(self.args.ckpt, 'meta')
            start_step = meta_conf['step'] + 1
            best_step = meta_conf['best_step']
            best_results = meta_conf['best_results']
            pbar.update(start_step)
            show(f'[RUNNER] Resume training from step {start_step}.')
            show('-' * 50)
            show(f'[RUNNER] Last Best result:')
            self._show_valid_results(best_step, best_results)
        else: # start from scratch
            start_step = 1
            best_step = 0
            best_results = {}
            show(f'[RUNNER] Start training from scratch.')

        # start training
        show('-' * 50)
        for step in range(start_step, runner_conf['total_steps'] + 1):
            # train one gradient step or one epoch
            if runner_conf['step_mod'] == 'grad':
                trainer.one_step()
            elif runner_conf['step_mod'] == 'epoch':
                trainer.one_epoch()
            else:
                raise ValueError(f'Unknown step_mod: {runner_conf["step_mod"]}')

            # update one step
            pbar.update()

            # should print slash in the end of this step
            print_slash = False

            # show result per log_freq steps
            if step % runner_conf['log_freq'] == 0 or step == runner_conf['total_steps']:
                train_result = trainer.pop_result()
                pbar.refresh() # refresh tqdm bar to show the latest result
                self._show_train_result(step, trainer.lr, train_result)
                print_slash = True
                if self.args.WandB: # log train result to wandb
                    wandb.log(train_result, step=step, commit=False)

            # prepare for saving
            if not self.args.disable_save:
                save_params = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                save_confs = {
                    'modelrc': self.modelrc,
                    'taskrc': self.taskrc,
                    'meta': {'step': step, 'best_step': best_step, 'best_results': best_results}
                }

            if step % runner_conf['save_freq'] == 0 and not self.args.disable_save:
                ckpt_manager.save_model(
                    ckpt_name=f'step-{step}.pth',
                    params=save_params,
                    confs=save_confs,
                    overwrite=self.args.overwrite
                )
                ckpt_manager.clean_old_ckpt('step-*.pth')
                print_slash = True

            # validate a epoch per dev_freq steps
            if step % runner_conf['valid_freq'] == 0 or step == runner_conf['total_steps']:
                if print_slash:
                    show('-' * 50)
                current_results = {}
                for name, valid_loader in zip(loader_conf['valid_selects'], valid_loaders):
                    valider.one_epoch(valid_loader, name)
                    current_result = valider.pop_result()
                    current_results[name] = current_result
                    if self.args.WandB: # log valid result to wandb
                        log_result = {f'{name}-{k}':v for k,v in current_result.items()}
                        wandb.log(log_result, step=step, commit=False)

                pbar.refresh() # refresh tqdm bar to show the latest result
                self._show_valid_results(step, current_results)
                print_slash = True

                # save model if better
                if not self.args.disable_save:
                    if ckpt_manager.check_better(current_results, best_results):
                        best_step = step
                        best_results = current_results
                        ckpt_manager.save_model(
                            ckpt_name='valid-best.pth',
                            params=save_params,
                            confs=save_confs,
                            overwrite=True
                        )

            if print_slash:
                show('-' * 50)

            # log lr to wandb, and commit this step
            if self.args.WandB:
                wandb.log({'lr':trainer.lr}, step=step, commit=True)

        show(f'[RUNNER] Training finished.')
        show(f'[RUNNER] Best result:')
        self._show_valid_results(best_step, best_results)
        show('-' * 50)

        pbar.close()
        trainer.close()
        valider.close()


    def evaluate(self):
        show(f'[RUNNER] Mod: EVALUATE')

        # load models
        model = self._get_model()
        if self.taskrc['arch']['metric']['use_loss']:
            metrics = self._get_modules_for_eval(self._get_criterion())
        else:
            metrics = self._get_modules_for_eval()

        # load model from checkpoint if have
        model.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'model'))

        # prepare dataset and dataloader
        loader_conf = self.taskrc['data']['dataloaders']
        show("[RUNNER] Valid dataset:")
        valid_loaders = [self._get_dataloader(self.taskrc['data']['datasets'], loader_conf[name], name) for name in self.args.valid_loaders]

        # create valider
        runner_conf = self.taskrc['runner']
        valider = Valider(
            model=model,
            metrics=metrics,
            device=self.args.device,
            silent=self.args.silent,
            **runner_conf['valider_kwargs']
        )

        # start validating
        show("[RUNNER] Start evaluating.")
        show('-' * 50)
        current_results = {}
        for name, valid_loader in zip(self.args.valid_loaders, valid_loaders):
            # validate a epoch
            valider.one_epoch(valid_loader, name)
            # show result
            current_results[name] = valider.pop_result()
        self._show_valid_results(0, current_results)

        valider.close()
