import os
import yaml
import time
import wandb
import torch
from importlib import import_module
from typing import Dict, Any, Callable
from argparse import Namespace
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm.auto import tqdm
from functools import partial

from util.io import show
from util.utility import init, import_instance, create_instance
from modules.trainer import Trainer
from modules.valider import Valider
from modules.ckpt import CheckPointManager
from modules.Loss2evaluator import LossScore


def recursive_overwrite(conf: Dict[str,Dict|Any], new_conf: Dict[str,Dict|Any]):
    for key, value in new_conf.items():
        if isinstance(value, dict):
            if key not in conf:
                raise ValueError(f'Cannot find {key} in config.')
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
            show(f'[Runner] Load modelrc from {args.modelrc}.')
            with open(args.modelrc, 'r') as f:
                self.modelrc = yaml.load(f, Loader=yaml.FullLoader)

        # load taskrc
        if args.taskrc is None:
            self.taskrc = CheckPointManager.load_config(args.ckpt, 'taskrc')
        else:
            show(f'[Runner] Load taskrc from {args.taskrc}.')
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
        show(f'[Runner] Random seed: {runner_conf["seed"]}.')


    def _get_model(self):
        model_select = self.modelrc['select']
        show(f"[Runner] Model: {model_select}.")
        return create_instance(self.modelrc[model_select])


    def _get_criterion(self):
        loss_select = self.taskrc['loss']['select']
        show(f"[Runner] Loss function: {loss_select}.")
        return create_instance(self.taskrc['loss'][loss_select])


    def _get_modules_for_train(self, parameters):
        # select optimizer
        optim_select = self.taskrc['optimizer']['select']
        show(f"[Runner] Optimizer: {optim_select}.")
        optimizer = create_instance(self.taskrc['optimizer'][optim_select], params=parameters)

        # select scheduler
        sched_select = self.taskrc['scheduler']['select']
        show(f"[Runner] Scheduler: {sched_select}.")
        scheduler = create_instance(self.taskrc['scheduler'][sched_select], optimizer=optimizer)

        return optimizer, scheduler


    def _get_modules_for_eval(self, criterion = None):
        metric_conf = self.taskrc['metric']

        # select evaluator
        metrics = {}
        for name in metric_conf['selects']:
            metrics[name] = create_instance(metric_conf[name])
        loss_metric = LossScore(criterion) if criterion else None

        # show metrics
        if len(metrics) >= 0 or loss_metric:
            show("[Runner] Metrics:")
            for i, name in enumerate(metrics.keys(), start=1):
                show(f"\t{i}. {name}")
            if loss_metric:
                show(f"\t{len(metrics.keys())+1}. {metric_conf['loss_name']}")
        else:
            show("[Runner] No metrics.")

        return metrics, loss_metric

    @staticmethod
    def _get_dataset(datasets_conf:Dict, select:str):
        dataset = datasets_conf[select]
        return create_instance(dataset)

    @staticmethod
    def _get_dataloader(loader_conf:Dict, process_conf:Dict, select:str, dataset):
        conf = loader_conf[select]
        kw_instances = {}
        if collate_fn := conf['collate_fn']:
            kw_instances['collate_fn'] = Runner._get_process_fn(process_conf, collate_fn)
        return DataLoader(dataset, **kw_instances, **conf['kwargs'])

    @staticmethod
    def _get_process_fn(process_conf: Dict[str, Dict[str, Any]], select: str|None):
        if select:
            conf = process_conf[select]
            return partial(import_instance(conf), **conf['kwargs'])
        return None


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
        show(f'[Runner] Mod: TRAIN')

        # load models
        model = self._get_model()
        criterion = self._get_criterion()
        optimizer, scheduler = self._get_modules_for_train(model.parameters())
        metrics, loss_metric = self._get_modules_for_eval(criterion)

        # load model from checkpoint if have
        if self.args.ckpt:
            # first move model to device to avoid bug of 'device mismatch'
            model.to(self.args.device)
            model.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'model'))
            if self.args.resume:
                optimizer.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'optimizer'))
                scheduler.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'scheduler'))


        data_conf = self.taskrc['data']
        datasets_conf = data_conf['datasets']
        loader_conf = data_conf['dataloaders']
        process_conf = data_conf['process_fns']

        # check and show dataset
        train_select = data_conf['train_dataset']
        valid_selects = data_conf['valid_datasets']
        if len(valid_selects) != len(set(valid_selects)):
            raise ValueError('Duplicate loaders in taskrc["data"]["valid_datasets"]')
        show(f"[Runner] Train dataset: {train_select}.")
        show("[Runner] Valid dataset:")
        for i, select in enumerate(valid_selects, start=1):
            show(f"\t{i}. {select}")
        # create dataset
        train_dataset = self._get_dataset(datasets_conf, train_select)
        valid_datasets = [self._get_dataset(datasets_conf, select) for select in valid_selects]

        # show dataloader
        train_select = data_conf['train_dataloader']
        valid_select = data_conf['valid_dataloader']
        show(f"[Runner] Train dataloader: {train_select}.")
        show(f"[Runner] Valid dataloader: {valid_select}.")
        # create dataloader
        train_loader = self._get_dataloader(loader_conf, process_conf, train_select, train_dataset)
        valid_loaders = [self._get_dataloader(loader_conf, process_conf, valid_select, valid_dataset) for valid_dataset in valid_datasets]

        # batch preprocess function
        batch_preprocess_fn = None
        if data_conf['batch_preprocess']:
            batch_preprocess_fn = data_conf['batch_preprocess_fn']
            show(f"[Runner] Preprocess batch with {batch_preprocess_fn}.")
            batch_preprocess_fn = self._get_process_fn(process_conf, batch_preprocess_fn)

        # augment batch_fn
        augment_fn = None
        if data_conf['augment']:
            augment_fn = data_conf['augment_fn']
            show(f"[Runner] Augment data with {augment_fn}.")
            augment_fn = self._get_process_fn(process_conf, augment_fn)


        # create checkpoint manager if needed
        if not self.args.disable_save:
            ckpt_conf = self.taskrc['ckpt']
            if ckpt_conf['check_datasets'] is None:
                ckpt_conf['check_datasets'] = valid_selects
            ckpt_manager = CheckPointManager(self.args.ckpt_dir, **ckpt_conf)

        # init status from checkpoint or from scratch
        if self.args.resume: # resume from checkpoint
            meta_conf = CheckPointManager.load_config(self.args.ckpt, 'meta')
            start_step = meta_conf['step']
            best_step = meta_conf['best_step']
            best_results = meta_conf['best_results']
            run_id = meta_conf['run_id']
            if self.args.WandB:
                wandb_conf = self.taskrc['WandB']
                if run_id is None: # generate run_id if not have
                    run_id = wandb.util.generate_id()
                    show(f'[WandB] Cannot find run_id in checkpoint.')
                    show(f'[WandB] Generate new run_id: {run_id}.')
                show(f'[Runner] Initialize W&B to record training.')
                if self.args.resume: # resume from checkpoint
                    wandb.init(
                        project=wandb_conf['project'],
                        resume="allow",
                        id=run_id,
                        **wandb_conf['init_kwargs']
                    )
            show(f'[Runner] Resume training from step {start_step}.')
            show('-' * 50)
            show(f'[Runner] Last Best result:')
            self._show_valid_results(best_step, best_results)
        else: # start from scratch
            start_step = 0
            best_step = 0
            best_results = {}
            run_id = None
            if self.args.WandB:
                assert self.args.name is not None, 'Please specify the name of this run.'
                run_id = wandb.util.generate_id()
                show(f'[WandB] Generate new run_id: {run_id}.')
                wandb_conf = self.taskrc['WandB']
                wandb.init(
                    project=wandb_conf['project'],
                    name=self.args.name,
                    id=run_id,
                    reinit=True,
                    **wandb_conf['init_kwargs']
                )
            show(f'[Runner] Start training from scratch.')

        # watch model params in wandb if needed
        if self.args.WandB and self.taskrc['WandB']['track_params']:
            wandb.watch(models=model, criterion=criterion, **self.taskrc['WandB']['watch_kwargs'])

        # create trainer and valider
        runner_conf = self.taskrc['runner']
        metric_conf = self.taskrc['metric']
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.args.device,
            metrics=MetricCollection(metrics) if metric_conf['apply_on_train'] else None,
            batch_preprocess_fn=batch_preprocess_fn,
            augment_fn=augment_fn,
            silent=self.args.silent,
            **runner_conf['trainer_kwargs']
        )
        valider_metrics = metrics.copy()
        if metric_conf['use_loss']:
            assert metric_conf['loss_name'] not in metrics, f'Loss name {metric_conf["loss_name"]} is already used.'
            valider_metrics[metric_conf['loss_name']] = loss_metric
        valider = Valider(
            model=model,
            metrics=MetricCollection(valider_metrics),
            device=self.args.device,
            batch_preprocess_fn=batch_preprocess_fn,
            silent=self.args.silent,
            **runner_conf['valider_kwargs']
        )

        # start training
        show('-' * 50)
        pbar = tqdm(total=runner_conf['total_steps'], initial=start_step, desc='Overall', dynamic_ncols=True, disable=self.args.silent)
        for step in range(start_step+1, runner_conf['total_steps'] + 1):
            # train one gradient step or one epoch
            if runner_conf['step_mod'] == 'grad':
                trainer.one_step()
            elif runner_conf['step_mod'] == 'epoch':
                trainer.one_epoch()
            else:
                raise ValueError(f'Unknown step_mod: {runner_conf["step_mod"]}')

            # update one step
            if not self.args.silent:
                pbar.update()

            # should print slash in the end of this step
            print_slash = False

            # show result per log_freq steps
            if step % runner_conf['log_freq'] == 0 or step == runner_conf['total_steps']:
                train_result = trainer.pop_result()
                if not self.args.silent:
                    pbar.refresh() # refresh tqdm bar to show the latest result
                self._show_train_result(step, trainer.lr, train_result)
                print_slash = True
                if self.args.WandB: # log train result to wandb
                    log_result = {f'train-{k}':v for k,v in train_result.items()}
                    wandb.log(log_result, step=step, commit=False)

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
                    'meta': {
                        'step': step,
                        'best_step': best_step,
                        'best_results': best_results,
                        'run_id': run_id
                    }
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
                for name, valid_loader in zip(valid_selects, valid_loaders):
                    # free gpu cache
                    if 'cuda' in self.args.device:
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                    # validate a epoch
                    valider.one_epoch(valid_loader, name)
                    current_result = valider.pop_result()
                    current_results[name] = current_result
                    if self.args.WandB: # log valid result to wandb
                        log_result = {f'{name}-{k}':v for k,v in current_result.items()}
                        wandb.log(log_result, step=step, commit=False)
                # free gpu cache
                if 'cuda' in self.args.device:
                    with torch.cuda.device(self.args.device):
                        torch.cuda.empty_cache()

                if not self.args.silent:
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

        show(f'[Runner] Training finished.')
        show(f'[Runner] Best result:')
        self._show_valid_results(best_step, best_results)
        show('-' * 50)

        pbar.close()
        trainer.close()
        valider.close()


    def evaluate(self):
        show(f'[Runner] Mod: EVALUATE')

        # load models
        model = self._get_model()
        if self.taskrc['metric']['use_loss']:
            metrics, loss_metrics = self._get_modules_for_eval(self._get_criterion())
            metrics[self.taskrc['metric']['loss_name']] = loss_metrics
        else:
            metrics, _ = self._get_modules_for_eval()

        # load model from checkpoint if have
        model.load_state_dict(CheckPointManager.load_param(self.args.ckpt, 'model'))


        data_conf = self.taskrc['data']
        datasets_conf = data_conf['datasets']
        loader_conf = data_conf['dataloaders']
        process_conf = data_conf['process_fns']

        # check and show dataset
        valid_selects = self.args.valid_datasets
        if len(valid_selects) != len(set(valid_selects)):
            raise ValueError('Duplicate loaders in taskrc["data"]["valid_datasets"]')
        show("[Runner] Valid dataset:")
        for i, select in enumerate(valid_selects, start=1):
            show(f"\t{i}. {select}")
        # create dataset
        valid_datasets = [self._get_dataset(datasets_conf, select) for select in valid_selects]

        # show dataloader
        valid_select = self.args.valid_dataloader or data_conf['valid_dataloader']
        show(f"[Runner] Valid dataloader: {valid_select}.")
        # create dataloader
        valid_loaders = [self._get_dataloader(loader_conf, process_conf, valid_select, valid_dataset) for valid_dataset in valid_datasets]

        # batch preprocess function
        batch_preprocess_fn = None
        if data_conf['batch_preprocess']:
            batch_preprocess_fn = data_conf['batch_preprocess_fn']
            show(f"[Runner] Preprocess batch with {batch_preprocess_fn}.")
            batch_preprocess_fn = self._get_process_fn(process_conf, batch_preprocess_fn)


        # create valider
        runner_conf = self.taskrc['runner']
        valider = Valider(
            model=model,
            metrics=MetricCollection(metrics),
            device=self.args.device,
            batch_preprocess_fn=batch_preprocess_fn,
            silent=self.args.silent,
            **runner_conf['valider_kwargs']
        )

        # start validating
        show("[Runner] Start evaluating.")
        show('-' * 50)
        current_results = {}
        for name, valid_loader in zip(self.args.valid_datasets, valid_loaders):
            # validate a epoch
            valider.one_epoch(valid_loader, name)
            # show result
            current_results[name] = valider.pop_result()
        self._show_valid_results(0, current_results)

        valider.close()
