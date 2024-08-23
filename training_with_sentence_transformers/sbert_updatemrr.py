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
from torch.cuda.amp import autocast
import psutil
class SentenceTransformerA(SentenceTransformer):
        
    def fit(self,
        train_objective: Tuple[DataLoader, nn.Module],
        scheduler,
        optimizer, 
        scaler,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch = 1,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        accum_iter: int = 1,
        save_loss: bool = False
        ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        : param 
        : Number of batches for gradient accumulation 
        """

       
        info_loss_functions =  []
        dataloader, loss_model  = train_objective
        info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss_model))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch,  "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)

        self.to(self._target_device)

        dataloader.collate_fn = self.smart_batching_collate
 
        loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = len(dataloader)

        num_train_steps = int(steps_per_epoch * epochs)
        
        global_step = 0
        skip_scheduler = False
        data_iterator = iter(dataloader)
        loss_accum = 0

        if save_loss:
            csv_path = os.path.join(checkpoint_path, "train_losses.csv")
            #output_file_exists = os.path.isfile(csv_path)
            #ft = open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8")
            ft = open(csv_path, newline='', mode='w', encoding="utf-8")
            writer = csv.writer(ft)
    
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            print("ram", psutil.virtual_memory().percent)
            loss_model.zero_grad()
            loss_model.train()
            
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                data = next(data_iterator)
                print("reach the end of dataloader...")

            features, labels = data
            # run forward pass with autocasting.
            with autocast():
                loss_value = loss_model(features, labels)
                
            scale_before_step = scaler.get_scale()
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss_value/accum_iter).backward()
            loss_accum += loss_value.detach()
            if (epoch + 1) % accum_iter == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                skip_scheduler = scaler.get_scale() != scale_before_step
                
                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
                global_step += 1
                if save_loss:
                    
                    print(f"global step {global_step}, epoch {epoch} and loss {loss_accum}.")
                    writer.writerow([global_step, epoch, loss_accum.tolist()])
                    ft.flush()
                    loss_accum = 0
                #print(f"update grad at epoch {epoch}, global epoch {global_step}...")

                if evaluation_steps > 0 and global_step % evaluation_steps == 0:
                    self._eval_during_training(evaluator, checkpoint_path, save_best_model, epoch, global_step, callback)

                
                loss_model.zero_grad()
                loss_model.train()
                
                #print("learning rate", scheduler.get_last_lr())


            if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0 and (epoch + 1) % accum_iter == 0:
                print("save at global step ", global_step)
                self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


            #self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        #if checkpoint_path is not None:
        #    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
        
        if save_loss:
            ft.close()