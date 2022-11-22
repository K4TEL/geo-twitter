import time
from datetime import datetime
from pathlib import Path

from itertools import product

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, get_linear_schedule_with_warmup

from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, StepLR, MultiStepLR, CyclicLR

import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from utils.twitter_dataset import *
from utils.result_manager import *
from utils.cosine_scheduler import *
from utils.regressor import *
from utils.benchmarks import *
from utils.result_visuals import *

# model training + evaluation + hp tuning


class ModelTrainer():
    def __init__(self, file_prefix, twitter_dataloader, epochs, batch, outcomes=1, covariance=None, weighted=False,
                 loss_dist=True, loss_mf="mean", loss_prob="pos", loss_total="sum",
                 learn_rate_max=4e-5, learn_rate_min=1e-8, original_model=None):
        self.data = twitter_dataloader
        self.features = self.data.features
        self.n_features = self.data.n_features
        self.key_feature = self.data.key_feature
        self.minor_features = self.data.minor_features

        self.scaled = self.data.scaled

        self.val_feature = self.data.val_feature

        self.original_model = original_model
        self.model_file = f"{file_prefix}.pth"

        self.epochs = epochs
        self.batch_size = batch
        self.lr_max = learn_rate_max
        self.lr_min = learn_rate_min

        self.prefix = file_prefix

        if torch.cuda.is_available():
            #print(f"DEVICE\tAvailable GPU has {torch.cuda.device_count()} devices, using {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
            self.cluster = True
        else:
            #print(f"DEVICE\tNo GPU available, using the CPU with {torch.get_num_threads()} threads instead.")
            self.device = torch.device("cpu")
            self.cluster = False

        self.outcomes = outcomes
        self.covariance = covariance
        self.weighted = weighted if self.outcomes != 1 else False
        self.prob = self.covariance is not None
        self.model_folder = "prob" if self.prob else "spat"

        bert_wrapper = BERTregModel(self.outcomes, self.covariance, self.weighted, self.features, original_model)
        self.model = bert_wrapper.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr_max, eps=1e-8)

        self.key_outputs = bert_wrapper.key_output
        self.minor_outputs = bert_wrapper.minor_output

        loss_total = "type" if not self.prob else loss_total
        self.benchmark = ModelBenchmark(bert_wrapper, loss_dist, loss_prob, loss_mf, loss_total)

        self.key_loss = self.benchmark.key_feature_loss
        self.r2_score = self.benchmark.r2
        self.result_metrics = self.benchmark.result_metrics
        self.minor_loss = None if self.n_features == 1 else self.benchmark.minor_feature_loss
        if self.prob:
            self.prob_models = self.benchmark.prob_models

    def load_local_model(self, local_model=None):
        if local_model is None:
            local_model = self.model_file
        local_model = f"models/full/{self.model_folder}/{local_model}"
        print(f"LOAD\tLoading model from {local_model}")

        if not Path(local_model).is_file():
            print(f"LOAD [ERROR] Unable to load local model: file {local_model} does not exist")
            return

        if self.cluster:
            state = torch.load(local_model)
        else:
            state = torch.load(local_model, map_location='cpu')

        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save_local_model(self, local_model=None):
        if local_model is None:
            local_model = self.model_file
        local_model = f"models/full/{self.model_folder}/{local_model}"
        print(f"SAVE\tSaving model to file {local_model}")

        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict()}

        torch.save(state, local_model)

    def load_checkpoint(self, ckp_file=None):
        if ckp_file is None:
            ckp_file = self.ckp_file
        ckp_file = f"models/ckp/full/{self.model_folder}/{ckp_file}"
        print(f"LOAD\tLoading model checkpoint from {ckp_file}")

        if not Path(ckp_file).is_file():
            print(f"LOAD [ERROR] Unable to load local model checkpoint: file {ckp_file} does not exist")
            return 0, None

        if self.cluster:
            checkpoint = torch.load(ckp_file)
        else:
            checkpoint = torch.load(ckp_file, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if "train_loss_mean" not in checkpoint:
            train_loss_mean = None
        else:
            train_loss_mean = checkpoint['train_loss_mean']

        if "start_epoch" not in checkpoint:
            starting_epoch = 0
        else:
            starting_epoch = checkpoint['start_epoch']

        return starting_epoch, train_loss_mean

    def save_checkpoint(self, current_epoch, current_loss, ckp_file=None):
        if ckp_file is None:
            ckp_file = self.ckp_file
        ckp_file = f"models/ckp/full/{self.model_folder}/{ckp_file}"
        print(f"SAVE\tSaving model checkpoint to file {ckp_file}")

        checkpoint = {'start_epoch': current_epoch + 1,
                      'train_loss_mean': current_loss,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}

        torch.save(checkpoint, ckp_file)

    # raw prediction results for spat models
    def predict(self, dataloader):
        self.model.eval()
        output = np.empty((0, self.key_outputs), float)
        for batch in dataloader:
            batch_inputs, batch_masks, _ = tuple(b.to(self.device) for b in batch)
            with torch.no_grad():
                if self.cluster:
                    output = np.append(output, self.model(batch_inputs, batch_masks, self.key_feature).cpu().numpy(), axis=0)
                else:
                    output = np.append(output, self.model(batch_inputs, batch_masks, self.key_feature).numpy(), axis=0)
        return output

    #  training per-epoch metrics
    def test_metrics(self, dataloader, model=None):
        model = self.model if model is None else model
        val_metric = np.zeros([len(dataloader), 2, 2], dtype=float)
        model.eval()
        for step, batch in enumerate(dataloader):
            batch_metric = np.zeros((2, 2), dtype=float)
            batch_inputs, batch_masks, batch_labels = tuple(b.to(self.device) for b in batch)

            batch_inputs = batch_inputs.to(torch.int64)
            batch_masks = batch_masks.to(torch.int64)

            if self.cluster:
                torch.cuda.empty_cache()
                batch_inputs, batch_masks, batch_labels = batch_inputs.cuda(), batch_masks.cuda(), batch_labels.cuda()

            with torch.no_grad():
                outputs = model(batch_inputs, batch_masks, self.key_feature)

            spat_loss, prob_loss = self.key_loss(outputs, batch_labels)

            batch_metric[0, 0] = spat_loss.item()
            batch_metric[0, 1] = self.r2_score(outputs, batch_labels).item()
            if self.prob:
                batch_metric[1, 0] = prob_loss.item()
                batch_metric[1, 1] = torch.exp(-prob_loss).item()

            val_metric[step, :] = batch_metric.reshape(1, 2, 2)

        return val_metric

    # result metric
    def results(self, dataloader, val_size, model=None):
        model = self.model if model is None else model
        model.eval()

        val_metric = np.zeros([val_size, 2, 2], dtype=float)
        pm = [] if self.prob else None

        for step, batch in enumerate(dataloader):
            current_batch_size = len(batch[0])
            batch_metric = np.zeros((current_batch_size, 2, 2), dtype=float)
            batch_inputs, batch_masks, batch_labels = tuple(b.to(self.device) for b in batch)
            if self.cluster:
                torch.cuda.empty_cache()
                batch_inputs, batch_masks, batch_labels = batch_inputs.cuda(), batch_masks.cuda(), batch_labels.cuda()
            with torch.no_grad():
                outputs = model(batch_inputs, batch_masks, self.key_feature)

            spat_loss, prob_loss = self.result_metrics(outputs, batch_labels)

            if self.cluster:
                batch_metric[:, 0, 0] = spat_loss.cpu().numpy().flatten()
                batch_metric[:, 0, 1] = self.r2_score(outputs, batch_labels).cpu().numpy().flatten()
                if self.prob:
                    batch_metric[:, 1, 0] = prob_loss.cpu().numpy().flatten()
                    batch_metric[:, 1, 1] = torch.exp(-prob_loss).cpu().numpy().flatten()
            else:
                batch_metric[:, 0, 0] = spat_loss.numpy().flatten()
                batch_metric[:, 0, 1] = self.r2_score(outputs, batch_labels).numpy().flatten()
                if self.prob:
                    batch_metric[:, 1, 0] = prob_loss.numpy().flatten()
                    batch_metric[:, 1, 1] = torch.exp(-prob_loss).numpy().flatten()

            current_slize = step*self.batch_size
            val_metric[current_slize:current_slize+current_batch_size, :] = batch_metric.reshape(current_batch_size, 2, 2)

            if self.prob:
                pm.append(self.prob_models(outputs))

        return val_metric, pm

    # evaluation entry point
    def eval(self, val_size, threshold=200, map_size=1000, by_user=False, skip_size=0):
        if map_size > val_size:
            map_size = val_size

        print(f"\nStarting evaluation for model {self.model_file}")
        self.load_local_model()

        self.data.form_validation(self.batch_size, val_size, by_user, skip_size)
        val_size = len(self.data.val_dataloader.dataset)

        print(f"\nCalculating result metrics for {val_size} samples")
        val_metric, prob_models = self.results(self.data.val_dataloader, val_size)
        print(f"LOG\tVal mean:\n\tGeospatial:\t{'D^2' if self.benchmark.dist else 'Coord'} Loss:\t{np.mean(val_metric[:, 0, 0], axis=0)}\tCoord R2:\t{np.mean(val_metric[:, 0, 1], axis=0)}")
        if self.prob:
            print(f"\tProbabilistic:\t-LLH Loss:\t{np.mean(val_metric[:, 1, 0], axis=0)}\tProbability:\t{np.mean(val_metric[:, 0, 1], axis=0)}")

        result = ResultManager(self.data.val_df,
                               None,
                               self.val_feature,
                               self.device,
                               self.benchmark,
                               self.scaled,
                               by_user,
                               self.prefix)
        if self.prob:
            result.soft_outputs(prob_models)
        else:
            result.coord_outputs(self.predict(self.data.val_dataloader))

        result.metrics(val_metric)
        result.save_df()

        result.performance()

        visual = ResultVisuals(result)
        visual.density()
        visual.cum_dist(False, threshold)

    # training flow
    def train(self, start_epoch, train_loss_saved, scheduler, writer, log_step, ckp=True, clip_value=2):
        current_tlm = train_loss_saved
        if current_tlm is not None:
            print(f"TRAIN\tMean loss of saved model:\t{current_tlm}")

        start_epoch = int(start_epoch)
        epoch_steps = len(self.data.train_dataloader)
        print(f"TRAIN\tProgress of each epoch with {epoch_steps} steps will be logged every {log_step}th batch iteration")

        for epoch in range(start_epoch, self.epochs+start_epoch):
            print(f"\nEPOCH\t{epoch+1}\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\tStarting training ...")

            #train_metric = torch.empty((0, self.n_features, 2, 2), device=batch_inputs.device)
            train_metric = np.zeros([epoch_steps, self.n_features, 2, 2], dtype=float)

            self.model.train()
            for step, batch in enumerate(self.data.train_dataloader):
                ts = time.time()
                batch_inputs, batch_masks, batch_labels = tuple(b.to(self.device) for b in batch)

                self.optimizer.zero_grad()
                if self.cluster:
                    torch.cuda.empty_cache()
                batch_loss = torch.zeros((self.n_features, 2), device=batch_inputs.device)
                batch_metric = np.zeros((self.n_features, 2, 2), dtype=float)
                #minor_loss = torch.zeros(len(self.minor_features), device=batch_inputs.device)
                for f in range(len(self.features)):
                    feature = self.features[f]
                    feature_inputs = batch_inputs[:, f, :].to(torch.int64)
                    feature_masks = batch_masks[:, f, :].to(torch.int64)
                    if self.cluster:
                        feature_inputs, feature_masks, batch_labels = feature_inputs.cuda(), feature_masks.cuda(), batch_labels.cuda()

                    outputs = self.model(feature_inputs, feature_masks, feature)
                    spat_loss, prob_loss = self.key_loss(outputs, batch_labels) if feature == self.key_feature else self.minor_loss(outputs, batch_labels)

                    batch_metric[f, 0, 0] = spat_loss.item()
                    if feature == self.key_feature:
                        batch_metric[f, 0, 1] = self.r2_score(outputs, batch_labels).item()
                    if self.prob:
                        batch_metric[f, 1, 0] = prob_loss.item()
                        if feature == self.key_feature:
                            batch_metric[f, 1, 1] = torch.exp(-prob_loss).item()

                    batch_loss[f] = torch.stack((spat_loss, prob_loss))

                train_metric[step, :] = batch_metric.reshape(self.n_features, 2, 2)
                # print(batch_loss)
                # print(torch.mean(batch_loss, dim=0))
                # print(torch.sum(torch.mean(batch_loss, dim=0), dim=0))
                train_loss = self.benchmark.total_batch_loss(batch_loss)
                # print(train_loss)
                train_loss.backward()

                if (step % log_step) == 0:
                    print(f"EPOCH\t{epoch+1}\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\tStep: {step+1}/{epoch_steps} took {round(time.time()-ts,2)}s")
                    self.benchmark.log(writer, step + epoch*epoch_steps + 1, scheduler.optimizer.param_groups[0]["lr"], train_metric, step+1)
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(np.mean(train_loss))

                clip_grad_norm_(self.model.parameters(), clip_value)
                self.optimizer.step()
                if not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()

            print(f"EPOCH\t{epoch+1}\tCalculating evaluation metrics")
            val_metric = self.test_metrics(self.data.test_dataloader)
            self.benchmark.log(writer, (epoch+1)*epoch_steps, scheduler.optimizer.param_groups[0]["lr"], train_metric, step+1, val_metric)

            current_tlm = self.benchmark.mean_epoch_train_loss

            if ckp:
                self.save_checkpoint(epoch, current_tlm)
                if train_loss_saved is None:
                    self.save_local_model()
                    train_loss_saved = current_tlm
                else:
                    if current_tlm <= train_loss_saved:
                        print(f"EPOCH\t{epoch+1}\tTraining loss decreased: {train_loss_saved} --> {current_tlm}.  Saving model ...")
                        self.save_local_model()
                        train_loss_saved = current_tlm
                    else:
                        print(f"EPOCH\t{epoch+1}\tTraining loss increased: {train_loss_saved} --> {current_tlm}")

            else:
                self.save_local_model()

        return current_tlm

    # training entry point
    def finetune(self, train_size, test_ratio, local_model=None, ckp=True, log_step=1000, scheduler_type="cosine", skip_size=0):
        self.data.form_training(self.batch_size, train_size, test_ratio, skip_size=skip_size)

        if local_model is not None:
            self.model_file = local_model

        self.load_local_model()
        start_epoch = 0
        train_loss_saved = None

        if ckp:
            self.ckp_file = f"ckp-{self.prefix}.pth"
            start_epoch, train_loss_saved = self.load_checkpoint()

        log_folder = f"./runs/full/{self.model_folder}/{self.prefix}-{datetime.today().strftime('%Y-%m-%d')}_logs"
        Path(log_folder).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_folder)

        print(f"\nStarting training for {self.epochs} epochs with {len(self.data.train_dataloader) * self.epochs} steps in total")

        self.train(start_epoch,
                   train_loss_saved,
                   self.get_scheduler(scheduler_type),
                   writer,
                   log_step,
                   ckp=ckp,
                   clip_value=2)

        print(f"\nTraining is finished:\nBest model is saved to {self.model_file}")
        if ckp is not None:
            print(f"Last model checkpoint is saved to {self.ckp_file}")

        writer.close()

    # hp tuning entry point
    def hp_tuning(self, train_size, test_ratio, param_values, log_step=1000):
        self.data.form_training(self.batch_size, train_size, test_ratio, True)

        epoch_steps = len(self.data.train_dataloader)
        total_steps = epoch_steps * self.epochs

        logs_folder = f"./runs/hptune_logs/full/{self.model_folder}/{datetime.today().strftime('%Y-%m-%d')}_schedulers"
        Path(logs_folder).mkdir(parents=True, exist_ok=True)

        print(f"\nHyper parameter tuning for {len(list(product(*param_values)))} runs without model saving and evaluation")

        for run_id, (max_lr, min_lr, scheduler) in enumerate(product(*param_values)):
            bert_wrapper = BERTregModel(self.outcomes, self.covariance, self.weighted)
            model = bert_wrapper.model.to(self.device)
            optimizer = AdamW(model.parameters(), lr=self.lr_max, eps=1e-8)

            run_scheduler = self.get_scheduler(scheduler, max_lr, min_lr, optimizer)

            print(f"\nStarting training for {self.epochs} epochs with {total_steps} steps in total")
            print(f"\tRun ID\t{run_id + 1}\n\tMAX lr\t{max_lr}\n\tMIN lr\t{min_lr}\n\tscheduler\t{scheduler}")

            run_folder = f"{logs_folder}/SCHEDULER:{scheduler};MAX_LR:{max_lr};MIN_LR:{min_lr}"
            Path(run_folder).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(run_folder)

            for epoch in range(self.epochs):
                print(f"\nEPOCH\t{epoch+1}\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\tStarting training ...")

                spatial_loss, spatial_r2 = [], []
                lh_loss, prob = None, None
                if self.prob:
                    lh_loss, prob = [], []

                model.train()
                for step, batch in enumerate(self.data.train_dataloader):
                    ts = time.time()
                    batch_inputs, batch_masks, batch_labels = tuple(b.to(self.device) for b in batch)
                    if self.cluster:
                        torch.cuda.empty_cache()
                        batch_inputs, batch_masks, batch_labels = batch_inputs.cuda(), batch_masks.cuda(), batch_labels.cuda()

                    optimizer.zero_grad()
                    minor_loss = torch.zeros(len(self.minor_features), device=batch_inputs.device)
                    for f in range(len(self.features)):
                        feature = self.features[f]
                        feature_inputs = batch_inputs[:, f, :]
                        feature_masks = batch_masks[:, f, :]
                        outputs = model(feature_inputs.to(torch.int64), feature_masks.to(torch.int64), feature)

                        if feature == self.key_feature:
                            key_loss = self.loss_function(outputs, batch_labels)
                            spatial_loss.append(self.benchmark.spatial_loss(outputs, batch_labels).item())
                            spatial_r2.append(self.benchmark.r2(outputs, batch_labels).item())

                            if self.prob:
                                lh = self.benchmark.log_likelihood_loss(outputs, batch_labels)
                                lh_loss.append(lh.item())
                                prob.append(torch.exp(-lh).item())
                        else:
                            minor_loss[f-1] = self.benchmark.minor_spatial_loss(outputs, batch_labels)

                    total_batch_loss = key_loss + torch.sum(minor_loss)
                    total_batch_loss.backward()

                    optimizer.step()

                    if (step % log_step) == 0:
                        print(f"EPOCH\t{epoch+1}\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\tStep: {step+1}/{epoch_steps} took {round(time.time()-ts,2)}s")
                        self.benchmark.log(writer, step + epoch*epoch_steps + 1, run_scheduler.optimizer.param_groups[0]["lr"],
                                           spatial_loss, spatial_r2, total_batch_loss, lh_loss, prob)
                        if isinstance(run_scheduler, ReduceLROnPlateau):
                            run_scheduler.step(np.mean(lh_loss if self.prob else spatial_loss))

                    clip_grad_norm_(model.parameters(), 2)
                    if not isinstance(run_scheduler, ReduceLROnPlateau):
                        run_scheduler.step()

                print(f"EPOCH\t{epoch+1}\tCalculating evaluation metrics")
                val_spatial_loss, val_spatial_r2, val_lh_loss, val_prob, = self.test_metrics(self.data.test_dataloader)
                self.benchmark.log(writer, (epoch+1)*epoch_steps, run_scheduler.optimizer.param_groups[0]["lr"],
                                   spatial_loss, spatial_r2, total_batch_loss, lh_loss, prob,
                                   val_spatial_loss, val_spatial_r2, val_lh_loss, val_prob)

            if self.prob:
                writer.add_hparams(
                    {"max_lr": max_lr, "min_lr": min_lr, "scheduler": scheduler},
                    {"train_spatial_loss": np.mean(spatial_loss), "val_spatial_loss": np.mean(val_spatial_loss),
                    "train_lh_loss": np.mean(lh_loss), "val_lh_loss": np.mean(val_lh_loss)}
                )
            else:
                writer.add_hparams(
                    {"max_lr": max_lr, "min_lr": min_lr, "scheduler": scheduler},
                    {"train_spatial_loss": np.mean(spatial_loss), "val_loss": np.mean(val_spatial_loss)}
                )
            writer.flush()

        writer.close()

    # scheduler hp tuning choice
    def get_scheduler(self, type, max_lr=None, min_lr=None, optimizer=None):
        if min_lr is None:
            min_lr = self.lr_min

        if max_lr is None:
            max_lr = self.lr_max

        if optimizer is None:
            optimizer = self.optimizer

        koef = 0.5

        epoch_steps = len(self.data.train_dataloader)
        total_steps = epoch_steps * self.epochs
        long_launch = 5

        print(f"TRAIN\tScheduler type:\t{type}\n\t\tMax LR:\t{max_lr}\n\t\tMin LR\t{min_lr}")

        if type == "cosine":
            scheduler = CyclicCosineDecayLR(optimizer,
                                            init_decay_epochs=total_steps,
                                            min_decay_lr=min_lr)
        if type == "cosine-long":
            scheduler = CyclicCosineDecayLR(optimizer,
                                            init_decay_epochs=total_steps*long_launch,
                                            min_decay_lr=min_lr)
        elif type == "cyclic":
            scheduler = CyclicLR(optimizer,
                                 cycle_momentum=False,
                                 base_lr=min_lr,
                                 max_lr=max_lr)
        elif type == "plateau":
            scheduler = ReduceLROnPlateau(optimizer,
                                          patience=2,
                                          mode="min",
                                          min_lr=min_lr)
        elif type == "step":
            scheduler = StepLR(optimizer,
                               step_size=epoch_steps//5,
                               gamma=koef)
        elif type == "multi-step":
            scheduler = MultiStepLR(optimizer,
                                    milestones=[epoch_steps//5, total_steps-epoch_steps//5],
                                    gamma=koef)
        elif type == "one-cycle":
            scheduler = OneCycleLR(optimizer,
                                   total_steps=total_steps,
                                   max_lr=max_lr)
        elif type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=total_steps)

        return scheduler

