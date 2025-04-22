import os
import random
from copy import deepcopy
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.cuda.amp as amp
from model.loss import *

from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger, FeatureType,
)

from recbole.trainer import Trainer


class EarlyStopping_BasedonLoss:
    def __init__(self, patience=10, min_delta_pct=0.01):
        """
        Early stopping mechanism that stops training when the loss decrease
        falls below a specified percentage.
        :param patience: Number of epochs to wait before stopping if no significant improvement.
        :param min_delta_pct: Minimum percentage decrease in loss to be considered an improvement (e.g., 0.01 means 1%).
        """
        self.patience = patience
        self.min_delta_pct = min_delta_pct
        self.best_loss = np.inf
        self.counter = 0

    def should_stop(self, current_loss):
        """
        Determines whether training should stop based on loss improvement.
        :param current_loss: The current loss value.
        :return: True if training should stop, False otherwise.
        """
        if self.best_loss == np.inf:
            self.best_loss = current_loss
            return False

        relative_delta = (self.best_loss - current_loss) / self.best_loss

        if relative_delta > self.min_delta_pct:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class CustomTrainer(Trainer):
    def __init__(self, config, model):
        super(CustomTrainer, self).__init__(config, model)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.device = config["device"]

        saved_model_file = "{}-{}-{}.pth".format(self.config["model"], self.config["dataset"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)


    def fit(
            self,
            train_data,
            valid_data=None,
            verbose=True,
            saved=True,
            show_progress=False,
            callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.96)
        for epoch_idx in range(self.start_epoch, self.epochs):
            self.logger.info(f'Learning rate is set to {scheduler.get_last_lr()[0]}')
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )
            scheduler.step()

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                                             set_color("epoch %d evaluating", "green")
                                             + " ["
                                             + set_color("time", "blue")
                                             + ": %.2fs, "
                                             + set_color("valid_score", "blue")
                                             + ": %f]"
                                     ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                        set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                            epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            self.resume_checkpoint(checkpoint_file)

        self.model.eval()
        results = {}

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = tqdm(eval_data, total=len(eval_data), ncols=100,
                         desc=set_color(f"Evaluate   ", "pink")) if show_progress else eval_data

        num_sample = 0
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                for batch_idx, batched_data in enumerate(iter_data):
                    num_sample += len(batched_data)
                    interaction, scores, positive_u, positive_i = eval_func(batched_data)
                    self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
                self.eval_collector.model_collect(self.model)
                struct = self.eval_collector.get_data_struct()
                result = self.evaluator.evaluate(struct)
                self.logger.info(f"Overall results: {result}")  # 输出评估结果

        if 'ndcg@10' not in result:
            raise KeyError("Evaluation result does not contain 'ndcg@10'")

        results['overall'] = result

        # 按照user-seqlen分组评估
        item_lengths = eval_data.dataset.inter_feat['item_length'].cpu().numpy()
        groups = [(0, 5), (5, 10), (10, 15), (15, 20), (20, float('inf'))]
        # groups = [(0, 20), (20, 50),(50, float('inf'))]
        # groups = [(0, 5), (5, 20),(20, float('inf'))]
        for start, end in groups:
            group_mask = (item_lengths >= start) & (item_lengths < end)
            group_indices = np.where(group_mask)[0]
            if len(group_indices) == 0:
                group_result, group_sample_count = {"hit@10": 0, 'ndcg@10': 0, 'mrr@10': 0}, 0
            else:
                group_result, group_sample_count = self.evaluate_group(eval_data, group_indices)
            results[f'user_group_{start}_{end}'] = group_result
            self.logger.info(f"User Group Len:{start} - {end} | Samples: {group_sample_count}: {group_result}")

        # 按照item分组评估
        groups = [(0, 10), (10, 50), (50, 100), (100, 200), (200, float('inf'))]
        for start, end in groups:
            item_id_list = list(
                filter(lambda k: start <= eval_data.trainData_item_counter[k] < end, eval_data.trainData_item_counter))
            group_mask = torch.isin(eval_data.dataset.inter_feat.interaction[eval_data.dataset.iid_field],
                                    torch.tensor(item_id_list))
            group_indices = np.where(group_mask)[0]
            if len(group_indices) == 0:
                group_result, group_sample_count = {"hit@10": 0, 'ndcg@10': 0, 'mrr@10': 0}, 0
            else:
                group_result, group_sample_count = self.evaluate_group(eval_data, group_indices)
            results[f'item_group_{start}_{end}'] = group_result
            self.logger.info(f"Item Group Len:{start} - {end} | Samples: {group_sample_count}: {group_result}")

        return results

    def evaluate_group(self, eval_data, group_indices):
        # Create a subset of the dataset
        subset_inter_feat = eval_data.dataset.inter_feat[group_indices]

        temp_all_inter_feat = eval_data.dataset.inter_feat
        eval_data.dataset.inter_feat = subset_inter_feat
        subset_eval_data = FullSortEvalDataLoader(
            config=eval_data.config, dataset=eval_data.dataset, sampler=eval_data.sampler, shuffle=False
        )

        # Reinitialize the collector
        self.eval_collector = Collector(config=self.config)

        iter_data = tqdm(subset_eval_data, total=len(subset_eval_data), ncols=100,
                         desc=set_color(f"Evaluate Subset", "pink"))
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                for batch_idx, batched_data in enumerate(iter_data):
                    interaction, scores, positive_u, positive_i = self._full_sort_batch_eval(batched_data)
                    self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
                self.eval_collector.model_collect(self.model)
                struct = self.eval_collector.get_data_struct()
                result = self.evaluator.evaluate(struct)

            eval_data.dataset.inter_feat = temp_all_inter_feat  # restore full evaluate data

        return result, len(group_indices)

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress
        )
        valid_result = valid_result['overall']
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result