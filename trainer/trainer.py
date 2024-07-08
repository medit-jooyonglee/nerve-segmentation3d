import os
import glob
import random
import torch
import numpy as np
import torch.nn as nn
import time
import yaml
import mlflow
import ml_collections
from collections.abc import MutableMapping

from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataloder import get_train_loaders
from .losses import get_loss_criterion
from .metrics import get_evaluation_metric
from .modelloader import get_model
from .configloader import load_config
from .utils import get_logger
from .torch_utils import get_number_of_learnable_parameters
from . import utils
from . import torch_utils
from .torch_utils import create_lr_scheduler, create_optimizer
from .imagesaver import get_image_saver
from .configloader import find_config

logger = get_logger('Trainer')

"""
https://github.com/wolny/pytorch-3dunet
"""


def log_metrics(metrics, step, prefix=''):
    prefix_metrics = {prefix + k: v for k, v in metrics.items()}
    mlflow.log_metrics(prefix_metrics, step)


def load_detection_model(config, to_search=''):
    detection_config_file = config['model'].get('detection')
    # detection_model = None
    if detection_config_file:
        detection_config_file = find_config(detection_config_file)
        detection_config = load_config(detection_config_file)
        detection_model = get_model(detection_config['model'])
        detection_model.eval()

        detection_checkpoint_dir = detection_config['trainer'].get('checkpoint_dir')
        best_checkpoints = glob.glob(os.path.join(detection_checkpoint_dir, 'best*'))
        if len(best_checkpoints) == 0:
            detection_checkpoint_dir = os.path.join(to_search, detection_checkpoint_dir)
            best_checkpoints = glob.glob(os.path.join(detection_checkpoint_dir, 'best*'))

        if len(best_checkpoints) == 1:
            best_checkpoint_file = best_checkpoints[0]
            try:
                state = utils.load_checkpoint(best_checkpoint_file, detection_model)

                # state = utils.load_checkpoint(resume, self.model, self.optimizer)
                logger.info(
                    f"detection model best Checkpoint loaded from '{best_checkpoint_file}'. Epoch: {state['num_epochs']}.  "
                    f"Iteration: {state['num_iterations']}. "
                    f"Best val score: {state['best_eval_score']}."
                )
            except Exception as e:
                logger.error(e.args[0])
    else:
        detection_model = None

    return detection_model


def to_dict(obj):
    if isinstance(obj, ml_collections.ConfigDict):
        return obj.to_dict()
    elif isinstance(obj, (dict,)):
        return obj


def create_trainer(config, package=__file__):
    # Create the model
    model = get_model(config['model'])

    # detection_config_file = config['model'].get('detection')

    detection_model = load_detection_model(config)

    check_pointer_dir = config['trainer']['checkpoint_dir']
    check_pointer_dir = check_pointer_dir if os.path.isabs(check_pointer_dir) else \
        os.path.join(os.path.dirname(os.path.abspath(package)), check_pointer_dir)
    if not os.path.exists(check_pointer_dir):
        os.makedirs(check_pointer_dir)
    config_save_path = os.path.join(check_pointer_dir, 'config')
    if not os.path.exists(config_save_path):
        os.makedirs(config_save_path)
    # config['trainer']['checkpoint_dir'] = check_pointer_dir

    yaml.safe_dump(to_dict(config),
                   open(os.path.join(config_save_path, f'config_{time.strftime("%Y%m%d%H%M%S")}.yaml'), 'w'))

    trainer_config = to_dict(config['trainer'])
    # checkpoint 수정한 경로로 변경
    trainer_config['checkpoint_dir'] = check_pointer_dir

    device_str = config.get('device', 'cuda:0')
    device = torch.device(device_str)
    if device.index >= torch.cuda.device_count():
        logger.error(f'invalid devices. {device}')
        torch.cuda.set_device(torch.device('cuda:0'))
    else:
        torch.cuda.set_device(device)
    # if device_str.startswith('cuda'):
    # deivice = torch.device
    if torch.cuda.device_count() > 1 and not device_str == 'cpu' and config['model'].get('parallel'):
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
        model = model.cuda()

    if torch.cuda.is_available() and not device_str == 'cpu':
        model = model.cuda()
        if detection_model is not None:
            detection_model = detection_model.cuda()

    # if is
    # if device.startswith('cuda'):
    #     device =

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)
    for data_loader in loaders.values():
        if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'initialize_detection_model'):
            data_loader.dataset.initialize_detection_model(detection_model)

    # Create the optimizer
    optimizer = create_optimizer(trainer_config.pop('optimizer', {}), model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(trainer_config.pop('lr_scheduler', None), optimizer)

    # Create tensorboard formatter
    # trainer_config_dict = to_dict(trainer_config)
    resume = trainer_config.pop('resume', True)
    pre_trained = trainer_config.pop('pre_trained', None)
    #
    image_save_func = get_image_saver(config.get('saver', {}))
    image_save_dir = config.get('saver', {}).get('image_save_dir', 'd:/temp/unet3d_trainer')

    all_params = to_dict(config)  # dict(config)

    trainer_config = {
        **trainer_config,
        'config': all_params
    }

    return Trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                   eval_criterion=eval_criterion, loaders=loaders,
                   resume=resume, pre_trained=pre_trained,
                   eval_metrics=eval_criterion,
                   image_save_func=image_save_func,
                   image_save_dir=image_save_dir,
                   device=device,
                   **trainer_config)


def create_evaluator(config):
    trainer = create_trainer(config)
    # 로더만 변경한다.
    # all_data = trainer.k_folder.get_all_dataset()
    trainer.loaders['valid'].dataset.en_augmented = False
    # trainer.loaders['valid'].dataset._num_save_image = False
    trainer.loaders['valid'].dataset._num_reuse_image = 1
    # trainer.loaders['valid'].dataset.set_data_path_pair(trainer.k_folder.get_all_dataset())
    # trainer.loaders['valid'].dataset.eval()
    return trainer


class Trainer:
    """UNet trainer
    https://github.com/wolny/pytorch-3dunet
    .

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        loaders (dict): 'train' and 'valid' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, checkpoint_dir,
                 max_num_epochs=200, max_num_iterations=100000, validate_after_iters=200, log_after_iters=100,
                 validate_iters=None,
                 num_iterations=1, num_epoch=0, eval_score_higher_is_better=True, tensorboard_formatter=None,
                 skip_train_validation=True, resume=None, pre_trained=None, eval_metrics=[], multiple_evaluate=True,
                 number_of_inputs=1,
                 device: torch.device = None,
                 log_image_iter=500,
                 image_save_func=None,
                 image_save_dir='',
                 **kwargs):

        # from commons import get_runtime_logger
        mlflow_path = kwargs.get('mlflow_path', 'd:/mlflow/mlruns')
        if not 'experiment_name' in kwargs:
            logger.warning('emtpy experiment name for mlflow')
        experiment_name = kwargs.get('experiment_name', 'teethnet')

        uri = mlflow_path if mlflow_path.startswith('http') else f'file:///{mlflow_path}'
        # mlflow.set_tracking_uri(uri=uri)
        # mlflow.set
        logger.info(f'{mlflow_path=}')
        mlflow.set_tracking_uri(uri=uri)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = exp.experiment_id
        self.experiment_id = experiment_id # experiment_name + time.strftime('%Y%m%d%H%M%S')
        self.run_name = kwargs.get('run_name', time.strftime('%Y%m%d%H%M%S'))
        self.config = kwargs.get('config', {})

        self.device = device

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.log_valid_after_iters = 5
        self.log_image_iter = log_image_iter
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.number_of_inputs = number_of_inputs
        self.batch_on = kwargs.get('batch_on', False)

        self.image_save_dir = os.path.join(image_save_dir, time.strftime("%Y%m%d%H%M%S"))
        self.image_save_func = image_save_func

        def get_post_process(name):
            try:
                funcs = utils.get_function(name, [
                    'interfaces.pidnetmodel'
                ])
            except Exception as e:
                funcs = lambda *x: x

            return funcs
            # return

        self.train_post_process = get_post_process(kwargs.get('train_post_process', ''))
        self.valid_post_process = get_post_process(kwargs.get('valid_post_process', ''))
        # getattr(model, kwargs.get('train_post_process', ''), lambda *x: x)

        # self.train_post_process = getattr(model, kwargs.get('train_post_process', ''), lambda *x: x)
        # self.valid_post_process = getattr(model, kwargs.get('valid_post_process', ''), lambda *x: x)
        self.pred_post_process = self.train_post_process or self.valid_post_process
        # yolo 같이 loss 값을 계산할대 모델 속성이 필요할 대
        self.loss_model_dependency = kwargs.get('loss_model_dependency', False)

        self.evaluated_metrics = eval_metrics

        # self.k_folder: DatasetKfold = loaders.get('k_folder')
        # default값으로 false로 지정.
        self.k_fold_cross_validation = kwargs.get('k_fold_cross_validation', False)

        self.multiple_evaluate = multiple_evaluate
        # logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')
        logger.info(f'train{len(self.loaders["train"].dataset)} / valid {len(self.loaders["valid"].dataset)}')

        self._log_image_enable = True  # 3d model일 경우, image log disable
        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        # self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        # assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        # self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            if isinstance(resume, bool):
                # founded best weights
                best_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'best*'))
                if len(best_checkpoints) >= 1:
                    resume = best_checkpoints[0]
            if isinstance(resume, str) and os.path.exists(resume):
                logger.info(f"Loading checkpoint '{resume}'...")
                state = dict()
                try:
                    state = utils.load_checkpoint(resume, self.model, self.optimizer, strict=False)
                    # state = utils.load_checkpoint(resume, self.model)
                    logger.info(
                        f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                        f"Best val score: {state['best_eval_score']}."
                    )
                except Exception as e:
                    logger.error(e.args[0])
                self.best_eval_score = state.get('best_eval_score', 0)
                # self.best_eval_score = 0
                self.num_iterations = state.get('num_iterations', 0) + 1
                self.num_epochs = state.get('num_epochs', 0)
                self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            utils.load_checkpoint(pre_trained, self.model, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        def flatten(dictionary, parent_key='', separator='_'):
            items = []
            for key, value in dictionary.items():
                new_key = parent_key + separator + key if parent_key else key
                if isinstance(value, MutableMapping):
                    items.extend(flatten(value, new_key, separator=separator).items())
                else:
                    items.append((new_key, value))
            return dict(items)

        with mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id) as run, \
            torch.autograd.set_detect_anomaly(True):
            flatten_config = flatten(self.config)
            for key, val in flatten_config.items():
                mlflow.log_params({key: val})

            for _ in range(self.num_epochs, self.max_num_epochs):
                # train for one epoch
                should_terminate = self.train()

                if should_terminate:
                    logger.info('Stopping criterion is satisfied. Finishing training')
                    return

                self.num_epochs += 1
                logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def prediction_collate(self, dataset, pred):
        if hasattr(dataset, 'prediction_collate'):
            dataset.prediction_collate(torch_utils.to_numpy(pred))

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage('loss')
        train_eval_scores = utils.RunningAverage('eval_scores')

        # sets the model in training mode
        self.model.train()
        self.validate_after_iters = len(self.loaders['train'])
        # for t in self.loaders['train']:
        # for i in range(len(self.loaders['train'].dataset)):
        for i, t in enumerate(self.loaders['train']):
            # t0 = next(self.loaders['train'])
            t = torch_utils.data_convert(t, device=self.device, batch=self.batch_on)

            inputs, target, weight = self._split_training_batch(t, self.number_of_inputs)
            try:
                output, loss = self._forward_pass(inputs, target, weight=weight)
            except Exception as e:
                logger.error(e.args)
                continue
            eval_score = self.evaluate_multiple_metrics(output, target)

            self.prediction_collate(self.loaders['train'].dataset, output)

            train_losses.update(loss, self._batch_size(inputs))
            train_eval_scores.update(eval_score, self._batch_size(inputs))

            if self.num_iterations % self.log_after_iters == 0:
                logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                            f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}] '
                            f'Loss [{train_losses.mean_avg_str}]  '
                            f'Metric [ {train_eval_scores}]  ')

                log_metrics(train_eval_scores.avg_dict, self.num_iterations, 'train')
                log_metrics(train_losses.avg_dict, self.num_iterations, 'train_')

            self.optimizer.zero_grad()
            # backward
            if isinstance(loss, dict):
                loss_sum = sum(loss.values())
                loss_sum.backward()
            elif isinstance(loss, torch.Tensor):
                loss.backward()
            # loss.backward()
            self.optimizer.step()

            # approximate 1 epoch check
            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score = self.validate()
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)
                # if self.k_fold_cross_validation:
                #     self.k_folder.shift_folding()
                #     self.loaders['train'].dataset.set_data_path_pair(self.k_folder.get_dataset('train'))
                #     self.loaders['valid'].dataset.set_data_path_pair(self.k_folder.get_dataset('valid'))

                self.num_epochs += 1

            if self.num_iterations % self.log_image_iter == 0:
                # compute eval criterion
                # if not self.skip_train_validation:
                #     if self.multiple_evaluate:
                #         eval_score = self.evaluate_multiple_metrics(output, target)
                #         train_eval_scores.update(eval_score, self._batch_size(inputs))
                #     else:
                #         eval_score = self.eval_criterion(output, target)
                #         train_eval_scores.update(eval_score.item(), self._batch_size(inputs))

                # log stats, params and images
                # logger.info(
                #     f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                # self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                # self._log_params()
                self._log_images(inputs, target, output, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def evaluate_multiple_metrics(self, input, target):
        scores = dict()
        inputs = (input, target)
        input = self.pred_post_process(*inputs)[0]
        if self.multiple_evaluate:
            for metric in self.evaluated_metrics:
                name = metric.__class__.__name__
                val = metric(input, target)
                if isinstance(val, torch.Tensor):
                    scores[name] = val.item()
                elif isinstance(val, dict):
                    for key, val in val.items():
                        scores[name + '_' + key] = val
        else:
            raise NotImplementedError
        return scores

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage('loss')
        val_scores = utils.RunningAverage('eval_scores')
        # from tools import vtk_utils
        self.loaders['valid'].dataset.train(False)
        with torch.no_grad():
            # for i in range(len(self.loaders['valid'].dataset)):

            for i, t in enumerate(self.loaders['valid']):
                t = torch_utils.data_convert(t, device=self.device, batch=self.batch_on)
                inputs, target, weight = self._split_training_batch(t, self.number_of_inputs)
                try:
                    output, loss = self._forward_pass(inputs, target, weight=weight)
                except Exception as e:
                    logger.error(e.args)
                    continue
                # output, loss = self._forward_pass(inputs, target, weight)
                eval_score = self.evaluate_multiple_metrics(output, target)

                # ins, pred, tar = torch_utils.to_numpy((inputs, torch.argmax(output, dim=1), target))
                # ins = ins.clip(0, 1)
                # pred, tar = pred.astype(np.int32), tar.astype(np.int32)
                # vtk_utils.split_show([ins], [ins, pred], item3= [ins, tar])

                self.prediction_collate(self.loaders['valid'].dataset, output)

                val_losses.update(loss, self._batch_size(inputs))
                val_scores.update(eval_score, self._batch_size(inputs))

                if i % self.log_valid_after_iters == 0:
                    self._log_images(inputs, target, output, 'val_')
                    logger.info(f'validate : {val_scores}')

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            log_metrics(val_losses.avg_dict, self.num_epochs, 'valid_')
            log_metrics(val_scores.avg_dict, self.num_epochs, 'valid_')

            self._log_stats('valid', val_losses.avg, val_scores.avg)
            if self.multiple_evaluate:
                logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores}')
                return val_scores.mean_avg
            else:
                logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                return val_scores.avg

    def _split_training_batch(self, t, num_inputs=1):
        def _move_to_gpu(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_gpu(x) for x in input])
            else:
                if torch.cuda.is_available():
                    input = input.cuda(non_blocking=True)
                return input

        # inputs, target = t[:-num_target], t[-num_target]

        inputs, target = t
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)

        inputs, targets = t if len(t) == 2 else (t[:num_inputs], t[num_inputs:])
        weight = None

        return inputs, targets, weight

    def _forward_pass(self, inputs, target, weight=None):

        output = self.model(inputs)
        output = self.pred_post_process(*(output, target))[0]

        loss_inputs = (output, target)
        if weight is not None:
            loss_inputs += (weight,)
        if self.loss_model_dependency:
            loss_inputs += (self.model,)
        # compute the loss
        loss = self.loss_criterion(*loss_inputs)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        utils.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        # self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        pass
        # tag_value = {
        #     f'{phase}_loss_avg': loss_avg,
        #     f'{phase}_eval_score_avg': eval_score_avg
        # }
        #
        # for tag, value in tag_value.items():
        #     self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        # for name, value in self.model.named_parameters():
        #     self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
        #     self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, inputs, target, output, prefix=''):
        if self._log_image_enable and self.image_save_func:
            output = self.train_post_process(output)
            self.image_save_func(inputs, target, output, save_path=os.path.join(self.image_save_dir, prefix))
        return
        #
        # if isinstance(self.model, nn.DataParallel):
        #     net = self.model.module
        # else:
        #     net = self.model
        #
        # if net.final_activation is not None:
        #     prediction = net.final_activation(prediction)
        #
        # inputs_map = {
        #     # 'inputs': input,
        #     'targets': target,
        #     'predictions': prediction
        # }
        #
        # if isinstance(inputs, (tuple, list)):
        #     input = {'inputs_{}'.format(i):v for i,v in enumerate(inputs)}
        # else:
        #     input = {'inputs': inputs}
        # inputs_map.update(input)
        #
        # img_sources = {}
        # for name, batch in inputs_map.items():
        #     if isinstance(batch, list) or isinstance(batch, tuple):
        #         for i, b in enumerate(batch):
        #             img_sources[f'{name}{i}'] = b.data.cpu().numpy()
        #     else:
        #         img_sources[name] = batch.data.cpu().numpy()
        #
        # for name, batch in img_sources.items():
        #     for tag, image in self.tensorboard_formatter(name, batch):
        #         if image.ndim > 3:
        #             self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return Trainer._batch_size(input[0])
        else:
            return input.size(0)


def train(config_or_configfilename):
    config = find_config(config_or_configfilename)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        # torch.backends.cudnn.deterministic = True

    trainer = create_trainer(config)
    trainer.train()
