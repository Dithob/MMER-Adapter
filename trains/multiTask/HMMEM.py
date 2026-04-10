import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain

logger = logging.getLogger('MSA')

class HMMEM():
    def __init__(self, args):

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):

        # ── Precision strategy: prefer bf16 on supported hardware ──
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        # bf16 doesn't need GradScaler; fp16 does
        scaler = None if use_bf16 else GradScaler()

        # ── Gradient Accumulation ──
        grad_accum_steps = getattr(self.args, 'gradient_accumulation_steps', 1)

        # Only optimize parameters that require gradients.
        # Without LoRA: Adapter + LSTM + Mixer + Fusion (~2M params)
        # With LoRA: above + LoRA injected params (~2M + ~3-7M)
        # This also avoids wasting optimizer state memory on frozen LLM params.
        trainable_params = [p for p in model.Model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=self.args.learning_rate, eps=1e-4)
        total_steps = len(dataloader['train'])*self.args.warm_up_epochs   #大致的一个训练step数
        # Adjust total_steps for gradient accumulation (optimizer steps, not forward steps)
        optimizer_steps = total_steps // grad_accum_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*optimizer_steps), num_training_steps=optimizer_steps)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")

        # initilize results
        logger.info("Start training...")
        logger.info(f"  Batch size = {self.args.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {grad_accum_steps}")
        logger.info(f"  Effective batch size = {self.args.batch_size * grad_accum_steps}")
        logger.info(f"  AMP dtype = {amp_dtype}")
        epochs, best_epoch = 0, 0
        losses = []

        CPC_Losses = []
        # valid_F1 = []
        lr = []
        min_or_max = 'min' if self.args.KeyEval in ['MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0     #评价阈值的初始化
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': []}
            y_true = {'M': []}
            model.train()
            train_loss = 0.0
            CPC_Loss_sum = 0.0
            ids = []
            optimizer.zero_grad(set_to_none=True)
            with tqdm(dataloader['train']) as td:
                for step, batch_data in enumerate(td):

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if self.args.train_mode == 'regression':
                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        prefix_label = batch_data['labels_prefix']
                        cur_id = batch_data['id']
                        ids.extend(cur_id)
                    else:
                        labels_m = batch_data['labels']['M']

                    indexes = batch_data['index'].view(-1)


                    if not self.args.need_data_aligned:
                        text_lengths = batch_data['text_lengths'].to(self.args.device)
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                    # forward
                    with autocast('cuda', dtype=amp_dtype):
                        output= model(labels_m, (text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))
                        loss = output['Loss']
                        # Add optional auxiliary losses
                        for aux_key in ['MoE_LB_Loss', 'DiffLoss', 'ExpertDiffLoss', 'NCELoss']:
                            if aux_key in output:
                                loss = loss + output[aux_key]
                        # Scale loss by gradient accumulation steps
                        loss = loss / grad_accum_steps

                    # backward
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    train_loss += loss.item() * grad_accum_steps  # unscale for logging
                    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])

                    # update parameters every grad_accum_steps
                    if (step + 1) % grad_accum_steps == 0:
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

            # Handle remaining steps that didn't complete a full accumulation cycle
            if (step + 1) % grad_accum_steps != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss = train_loss / len(dataloader['train'])

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f" % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            losses.append(train_loss)

            # validation

            if epochs >= 1:         #前3epochs不做eval
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]
                # save best model
                isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                if isBetter:
                    best_valid, best_epoch = cur_valid, epochs
                    # save model
                    self.save_model(model, epochs, self.args.model_save_path)
                    model.to(self.args.device)

                # early stop
                if epochs - best_epoch >= self.args.early_stop:     #如果比best_epoch再过了early_stop轮之后还没有出现新的best_epoch，就停止训练
                    if self.args.save_labels:
                        with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                            plk.dump(saved_labels, df, protocol=4)
                    return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}

        # Use same amp_dtype as training for consistency
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        if self.args.train_mode == 'regression':
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        with autocast('cuda', dtype=amp_dtype):
                            outputs = model.generate((text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))

                        predict_label = torch.Tensor(outputs).to(self.args.device)

                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        
                        y_pred['M'].append(predict_label.cpu())
                        y_true['M'].append(labels_m.cpu())
            pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
            logger.info(mode + "-(%s)" % self.args.modelName + " >>" )
            eval_results = self.metrics(pred, true)
            logger.info('M: >> ' + dict_to_str(eval_results))
        else:
            # train_mode == 'classification'
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        with autocast('cuda', dtype=amp_dtype):
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths),
                                                     (vision, vision_lengths))

                        predict_label = outputs
                        labels_m = batch_data['labels']['M']
                        
                        y_pred['M'].append(predict_label)
                        y_true['M'].append(labels_m)
            
            pred, true = list(chain(*y_pred['M'])), list(chain(*y_true['M']))
            
            eval_results = self.metrics(pred, true)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            logger.info('M: >> ' + dict_to_str(eval_results))

        return eval_results
    
    def l1_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            loss = torch.mean(torch.abs(y_pred - y_true))
        return loss


    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def save_model(self, model, epoch, save_path):
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.cpu().state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        logging.info("Saving checkpoint at epoch {} to {}.".format(epoch, save_path))
        torch.save(state_dict, save_path)
