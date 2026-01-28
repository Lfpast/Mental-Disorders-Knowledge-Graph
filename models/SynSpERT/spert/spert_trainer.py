"""
Trainer implementation for SpERT models.

Provides `SpERTTrainer` that wraps data loading, training loops, evaluation
and model checkpointing for joint entity and relation extraction.

Inputs:
- CLI args (`argparse.Namespace`) and a `BertConfig` instance.

Outputs:
- Trained models saved to disk and evaluation logs written to the log path.
"""

import argparse
import math
import os

import torch
import numpy as np
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizer

from spert import models
from spert import sampling
from spert import util  ##DKS
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, Loss
from tqdm import tqdm
from spert.trainer import BaseTrainer
from transformers import BertConfig
import sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace, config: BertConfig):
        super().__init__(args, config)

        # byte-pair encoding
        #DKS: Commented for now
        
        # Robust tokenizer loading: try from_pretrained, then fallback to local vocab.txt
        try:
            self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                            do_lower_case=args.lowercase,
                                                            cache_dir=args.cache_path)
            print(f"Loaded tokenizer from {args.tokenizer_path} via from_pretrained()")
        except Exception as exc:
            print(f"Warning: failed to load tokenizer from '{args.tokenizer_path}' with from_pretrained(): {exc}")
            # try direct vocab file in the given path
            vocab_path = os.path.join(args.tokenizer_path, 'vocab.txt')
            if os.path.exists(vocab_path):
                print(f"Loading tokenizer from vocab file: {vocab_path}")
                self._tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=args.lowercase)
            else:
                # search first-level subdirectories for vocab.txt
                found = None
                if os.path.isdir(args.tokenizer_path):
                    for name in os.listdir(args.tokenizer_path):
                        candidate = os.path.join(args.tokenizer_path, name, 'vocab.txt')
                        if os.path.exists(candidate):
                            found = candidate
                            break
                if found:
                    print(f"Found vocab in subdir, loading tokenizer from: {found}")
                    self._tokenizer = BertTokenizer(vocab_file=found, do_lower_case=args.lowercase)
                else:
                    raise OSError(f"Can't load tokenizer for '{args.tokenizer_path}'. Ensure the path contains a tokenizer or a subdirectory with 'vocab.txt'. Original error: {exc}")
        
        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def _load_pretrained_model(self, input_reader: BaseInputReader):
        # create model
        model_class = models.get_model(self.args.model_type) 
        #(Above) self.args.model_type = "syn_spert", model_class = 'SpERT'

        # load model
        #config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        config = self.config   #DKS
        util.check_version(config, model_class, self.args.model_path)

        config.spert_version = model_class.VERSION
        print("**** Loading pretrained model: TYPE: ", self.args.model_type, "****")
        
        # First, instantiate the model with random weights
        model = model_class(config=config,
                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                            relation_types=input_reader.relation_type_count - 1,
                            entity_types=input_reader.entity_type_count,
                            max_pairs=self.args.max_pairs,
                            prop_drop=self.args.prop_drop,
                            size_embedding=self.args.size_embedding,
                            freeze_transformer=self.args.freeze_transformer,
                            use_pos=self.args.use_pos,
                            use_entity_clf=self.args.use_entity_clf)
        
        # Now load the pretrained weights manually with proper key prefix handling
        ckpt_path = None
        if os.path.isdir(self.args.model_path):
            # Try pytorch_model.bin first, then model.safetensors
            for fname in ['pytorch_model.bin', 'model.safetensors']:
                cand = os.path.join(self.args.model_path, fname)
                if os.path.exists(cand):
                    ckpt_path = cand
                    break
        else:
            ckpt_path = self.args.model_path

        if ckpt_path is None:
            print(f"WARNING: No checkpoint file found in {self.args.model_path}. Model will use random initialization!")
        else:
            print(f"Loading pretrained weights from: {ckpt_path}")
            
            # Load the state dict
            state_dict = None
            try:
                if ckpt_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as load_safetensors
                    state_dict = load_safetensors(ckpt_path, device='cpu')
                else:
                    try:
                        state_dict = torch.load(ckpt_path, map_location='cpu')
                    except Exception:
                        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except Exception as le:
                print(f"WARNING: Failed to load checkpoint: {le}. Model will use random initialization!")
                state_dict = None
            
            if state_dict is not None:
                # Check if keys need 'bert.' prefix added
                # Model expects keys like 'bert.embeddings.word_embeddings.weight'
                # But pretrained file may have keys like 'embeddings.word_embeddings.weight'
                model_keys = set(model.state_dict().keys())
                
                # Check multiple keys to determine if prefix is needed
                needs_bert_prefix = False
                checkpoint_keys = list(state_dict.keys())
                
                # Look for a definitive key to test (word_embeddings is always present)
                for test_key in checkpoint_keys:
                    if 'word_embeddings.weight' in test_key or 'encoder.layer' in test_key:
                        if not test_key.startswith('bert.'):
                            # Check if adding 'bert.' prefix would match model keys
                            prefixed_key = 'bert.' + test_key
                            if prefixed_key in model_keys:
                                needs_bert_prefix = True
                                print("Detected pretrained weights without 'bert.' prefix. Adding prefix...")
                        break
                
                if needs_bert_prefix:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = 'bert.' + k
                        new_state_dict[new_key] = v
                    state_dict = new_state_dict
                
                # Check vocab size compatibility
                emb_key = 'bert.embeddings.word_embeddings.weight'
                if emb_key in state_dict:
                    ck_vocab, emb_dim = state_dict[emb_key].shape
                    model_vocab = model.bert.embeddings.word_embeddings.weight.shape[0]
                    print(f"Checkpoint vocab: {ck_vocab}, Model vocab: {model_vocab}")
                    
                    if ck_vocab != model_vocab:
                        print(f"WARNING: Vocab size mismatch! Checkpoint has {ck_vocab} tokens, model expects {model_vocab}.")
                        if ck_vocab < model_vocab:
                            print(f"Copying first {ck_vocab} embeddings, remaining {model_vocab - ck_vocab} will be random.")
                            model.bert.embeddings.word_embeddings.weight.data[:ck_vocab, :] = state_dict[emb_key]
                        else:
                            print(f"Truncating checkpoint embeddings to model size {model_vocab}.")
                            model.bert.embeddings.word_embeddings.weight.data[:] = state_dict[emb_key][:model_vocab, :]
                        # Remove from state_dict to avoid overwriting
                        del state_dict[emb_key]
                
                # Load remaining weights with strict=False to allow partial loading
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                # Report loading results
                loaded_count = len(state_dict) - len(unexpected_keys)
                print(f"Successfully loaded {loaded_count} parameter tensors.")
                if missing_keys:
                    # Filter out expected missing keys (task-specific heads)
                    unexpected_missing = [k for k in missing_keys if not any(
                        x in k for x in ['entity_classifier', 'rel_classifier', 'size_embeddings', 'pos_embeddings']
                    )]
                    if unexpected_missing:
                        print(f"WARNING: {len(unexpected_missing)} expected keys not found in checkpoint:")
                        for k in unexpected_missing[:5]:
                            print(f"  - {k}")
                        if len(unexpected_missing) > 5:
                            print(f"  ... and {len(unexpected_missing) - 5} more")
                if unexpected_keys:
                    print(f"NOTE: {len(unexpected_keys)} keys in checkpoint not used (expected for BERT-only pretrained).")

        print("Model type = ", type(model))

        return model


        return model

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        #ipconfig = self.config  #DKS
        train_label, valid_label = 'train', 'valid'

        self._logger.info("-" * 40)
        self._logger.info(f"{'Configuration':<15}")
        self._logger.info("-" * 40)
        self._logger.info(f"{'Model Type':<15}: {args.model_type}")
        self._logger.info(f"{'Train Data':<15}: {os.path.basename(train_path)}")
        self._logger.info(f"{'Valid Data':<15}: {os.path.basename(valid_path)}")
        self._logger.info("-" * 40)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        
        train_dataset = input_reader.get_dataset(train_label)
        
        
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info(f"{'Updates/Epoch':<15}: {updates_epoch}")
        self._logger.info(f"{'Total Updates':<15}: {updates_total}")
        self._logger.info("-" * 40)

        # create model
        model = self._load_pretrained_model(input_reader)
        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        
        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        #best_model = None #DKS
        #best_rel_f1_micro=0
        #best_epoch=0
        #model_saved = False

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets [DKS]
            if not args.final_eval or (epoch == args.epochs - 1):
               self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
               #ner_f1_micro,rel_f1_micro = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
               
               #DKS: check if model is best
               #if rel_f1_micro > best_rel_f1_micro:
                   #best_rel_f1_micro=rel_f1_micro
                   #best_model=model
                   #best_epoch=epoch+1
                   #extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                   #self._save_model(self._save_path, best_model, self._tokenizer, 0,
                   #    optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                   #    include_iteration=False, name='best_model')
                   #model_saved=True
                   #break
               
            
        #sys.exit(0)
        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')
        
        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()
        

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        #ipconfig = self.config  #DKS

        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model = self._load_pretrained_model(input_reader)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info(f"\n[Epoch {epoch}/{self.args.epochs}] Starting training...")
 
        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()
        
        #print("*************** train_dataset = ", dataset)
        #sys.exit(-1)
        
        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)
            
            # forward step
            
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'],
                                              dephead= batch['dephead'], deplabel =batch['deplabel'], pos= batch['pos'] )

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        # Initialize tensors for active learning features
        n = 9  # Number of relation classes
        prediction_entropy_relation = np.ones((1,))
        prediction_entropy_entities = np.ones((1,))
        prediction_label_all = np.ones((1, n))
        pooler_output_all = torch.ones((1, 768)).to(device=self._device)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               dephead= batch['dephead'], deplabel =batch['deplabel'] , pos= batch['pos'] , evaluate=True)
                entity_clf, rel_clf, rels, h_pooler = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

                # ===== Collect features for Active Learning =====
                # Pooler output
                pooler_output_all = torch.cat([pooler_output_all, h_pooler], dim=0)
                
                # Entropy for relations
                if rel_clf.shape[1] != 1:
                    s = -1 * rel_clf * torch.log(rel_clf + 1e-12)
                    s = torch.sum(s, dim=2, keepdim=False)
                    s = s.sum(dim=1, keepdim=True).cpu().numpy()
                else:
                    s = np.zeros((rel_clf.shape[0], 1))
                prediction_entropy_relation = np.vstack((prediction_entropy_relation, s))
                
                # Entropy for entities
                if entity_clf.shape[1] != 1:
                    e = -1 * entity_clf * torch.log(entity_clf + 1e-12)
                    e = torch.sum(e, dim=2, keepdim=False)
                    e = e.sum(dim=1, keepdim=True).cpu().numpy()
                else:
                    e = np.zeros((entity_clf.shape[0], 1))
                prediction_entropy_entities = np.vstack((prediction_entropy_entities, e))
                
                # Label predictions (relation classification)
                one = torch.ones_like(rel_clf)
                zero = torch.zeros_like(rel_clf)
                prediction_label = torch.where(rel_clf >= 0.4, one, zero)
                pred_labels = torch.sum(prediction_label, dim=1, keepdim=False)[0, :].cpu().numpy()
                # Ensure we have n classes
                if len(pred_labels) < n:
                    pred_labels = np.pad(pred_labels, (0, n - len(pred_labels)), mode='constant')
                else:
                    pred_labels = pred_labels[:n]
                prediction_label_all = np.vstack((prediction_label_all, pred_labels.reshape(1, -1)))

        # Save features for Active Learning
        if self.args.save_features:
            Names = self.args.Names if hasattr(self.args, 'Names') else 'run_v1'
            save_dir = self._log_path
            
            torch.save(torch.from_numpy(prediction_entropy_relation[1:, :]), 
                       os.path.join(save_dir, f'entropy_relation_{Names}.pt'))
            torch.save(torch.from_numpy(prediction_entropy_entities[1:, :]), 
                       os.path.join(save_dir, f'entropy_entities_{Names}.pt'))
            torch.save(torch.from_numpy(prediction_label_all[1:, :]), 
                       os.path.join(save_dir, f'labelprediction_{Names}.pt'))
            torch.save(pooler_output_all[1:, :], 
                       os.path.join(save_dir, f'pooler_output_{Names}.pt'))
            
            self._logger.info(f"Active Learning features saved in: {save_dir}")

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()
        
        return ner_eval[2], rel_eval[2]

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
