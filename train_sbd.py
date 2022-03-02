# This is adapted from https://github.com/monologg/JointBERT

import os
from tqdm import tqdm, trange
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed, AutoTokenizer, AutoConfig
from loguru import logger

from evaluate import evaluate_sbd_bio
from sbd_loader import load_and_cache_examples
from bert_sbd import SbdBERT

import params


class Trainer(object):
    def __init__(self,
                 args,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config = AutoConfig.from_pretrained(args.model_name_or_path,
                                                 finetuning_task=args.task)
        self.model = SbdBERT.from_pretrained(args.model_name_or_path,
                                             config=self.config,
                                             args=args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available(
        ) and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=train_sampler,
                                      batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(
                train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(
                train_dataloader
            ) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            self.args.weight_decay
        }, {
            'params': [
                p for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total)

        # Train
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total train batch size = {self.args.train_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {t_total}")
        logger.info(f"  Logging steps = {self.args.logging_steps}")
        logger.info(f"  Save steps = {self.args.save_steps}")

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'slot_labels_ids': batch[3]
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset,
                                     sampler=eval_sampler,
                                     batch_size=self.args.eval_batch_size)

        # Eval
        logger.info(f"***** Running evaluation on {mode} dataset *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        slot_preds = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'slot_labels_ids': batch[3]
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, slot_logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu(
                ).numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(
                        slot_preds,
                        np.array(self.model.crf.decode(slot_logits)),
                        axis=0)
                else:
                    slot_preds = np.append(slot_preds,
                                           slot_logits.detach().cpu().numpy(),
                                           axis=0)

                out_slot_labels_ids = np.append(
                    out_slot_labels_ids,
                    inputs["slot_labels_ids"].detach().cpu().numpy(),
                    axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(out_slot_labels_ids[i][j])
                    slot_preds_list[i].append(slot_preds[i][j])

        total_result = evaluate_sbd_bio(slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info(f"  {key} = {results[key]:.2f}")

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args,
                   os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info(f"Saving model checkpoint to {self.args.model_dir}")

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = SbdBERT.from_pretrained(self.args.model_dir,
                                                 args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task",
                        default="MultiWOZ_2.1",
                        type=str,
                        help="The name of the task to train")
    parser.add_argument("--model_dir",
                        default="dialog-flow-extraction/out/todbert_sbd",
                        type=str,
                        help="Path to save, load model")
    parser.add_argument("--data_dir",
                        default="dialog-flow-extraction/data",
                        type=str,
                        help="The input data dir")
    parser.add_argument("--model_name_or_path",
                        default="TODBERT/TOD-BERT-JNT-V1",
                        type=str,
                        help="Model name or saved path")
    parser.add_argument('--seed',
                        type=int,
                        default=666,
                        help="random seed for initialization")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Batch size for evaluation.")
    # TODO:
    parser.add_argument(
        "--max_seq_len",
        default=50,
        type=int,
        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate",
                        default=0.1,
                        type=float,
                        help="Dropout for fully-connected layers")
    parser.add_argument('--logging_steps',
                        type=int,
                        default=200,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps',
                        type=int,
                        default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument('--test_domain',
                        type=str,
                        default='hotel',
                        help="MultiWOZ domain to test")
    # set to 0 if use_crf
    parser.add_argument(
        "--ignore_index",
        default=-100,
        type=int,
        help=
        'Specifies a target value that is ignored and does not contribute to the input gradient'
    )
    # CRF option
    parser.add_argument("--use_crf",
                        action="store_true",
                        help="Whether to use CRF")

    args = parser.parse_args()
    params.test_domain = args.test_domain

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")
