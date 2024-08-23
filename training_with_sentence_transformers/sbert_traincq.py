from sentence_transformers import SentenceTransformer
import transformers
import torch 
import json
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
import math
import queue
import tempfile
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import import_from_string, batch_to_device, fullname, snapshot_download
import os
import csv
import numpy as np
import psutil
class SentenceTransformerCQ(SentenceTransformer):
        
    def fit(self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch = 1,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = transformers.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        accum_iter: int = 1,
        save_loss: bool = False
        ):
       
        info_loss_functions =  []
        dataloader, loss_model = train_objectives[0]
        info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss_model))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        #info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch,  "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        

        # Use smart batching
        dataloader.collate_fn = self.smart_batching_collate

        loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = len(dataloader)

        num_train_steps = int(steps_per_epoch * epochs)

        
        # Prepare optimizers
        
        param_optimizer = list(loss_model.named_parameters())

        for name, para in loss_model.named_parameters():
            if name.startswith("model.0."):
                para.requires_grad = False

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)
        
        global_step = 0
        data_iterator = iter(dataloader)
        skip_scheduler = False
        
        loss_accum = 0

        if save_loss:
            csv_path = os.path.join(checkpoint_path, "train_losses.csv")
            #output_file_exists = os.path.isfile(csv_path)
            #ft = open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8")
            ft = open(csv_path, newline='', mode='w', encoding="utf-8")
            writer = csv.writer(ft)

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            #print("ram", psutil.virtual_memory().percent)
            loss_model.zero_grad()
            loss_model.train()

            try:
                data = next(data_iterator)
            except StopIteration:
                print("reach dataloader end")
                data_iterator = iter(dataloader)
                data = next(data_iterator)

            features, labels = data
            if use_amp:
                with autocast():
                    loss_value = loss_model(features, labels)
                    
                scale_before_step = scaler.get_scale()
                scaler.scale(loss_value/accum_iter).backward()
                
                if (epoch + 1) % accum_iter == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    #print(scaler.get_scale(), scale_before_step)
                    skip_scheduler = scaler.get_scale() != scale_before_step
                    optimizer.zero_grad()

                    
            else:
                loss_value = loss_model(features, labels)/accum_iter
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                
                if (epoch + 1) % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            loss_accum += loss_value.detach()

            if (epoch + 1) % accum_iter == 0:    
                if not skip_scheduler:
                    scheduler.step()
                global_step += 1
                if save_loss:
                    
                    #print(f"global step {global_step}, epoch {epoch} and loss {loss_accum}.")
                    writer.writerow([global_step, epoch, loss_accum.tolist()])
                    ft.flush()
                    loss_accum = 0
                #print(f"update grad at epoch {epoch}, global epoch {global_step}...")
                
                if evaluation_steps > 0 and global_step % evaluation_steps == 0:
                    self._eval_during_training(evaluator, checkpoint_path, save_best_model, epoch, global_step, callback)

                
                loss_model.zero_grad()
                loss_model.train()

                #print("sbert, cls layer", epoch, loss_model.model[0].cls.module.decoder.state_dict()['weight'])
                #print("sbert, cls layer", epoch, loss_model.model[0].auto_model.module.cls.predictions.decoder.state_dict()['weight'])

            if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0 and (epoch + 1) % accum_iter == 0:
                print("save at global step ", global_step)
                print(loss_model.model[0].parameters)
                self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
                


            #self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
        
        if save_loss:
            ft.close()