"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import argparse
import json
import random
import sys
from typing import Counter
from collections import Counter

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sacrebleu import corpus_bleu

from util import draw_split

logger.remove()
logger.add(sys.stderr, level="INFO")

g_step = 0

# enable to draw state overlap, log details and save the model
EXP_VERBOSE = True
if not EXP_VERBOSE:
    logger.remove()
    logger.add(sys.stderr, level="ERROR")


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def build_input(history, sys_utt, tokenizer, max_len=500):
    encoded_input = tokenizer(history + sys_utt,
                              padding="max_length",
                              truncation=True,
                              max_length=max_len)
    input_ids = encoded_input.input_ids
    attention_mask = encoded_input.attention_mask
    labels = encoded_input.input_ids.copy()

    history_ids = tokenizer(history, truncation=True, max_length=max_len)

    labels = [
        -100 if mask == 0 else label
        for mask, label in zip(attention_mask, labels)
    ]  # https://github.com/huggingface/transformers/issues/2630

    for i in range(len(history_ids.input_ids)):
        labels[i] = -100

    assert len(input_ids) == max_len
    assert len(attention_mask) == max_len
    assert len(labels) == max_len

    return input_ids, attention_mask, labels


def convert_dialog_to_features(dialogs, tokenizer):
    features = []
    ex_index = 0
    for dialog in tqdm(dialogs):
        history_list = []
        for i in range(len(dialog)):
            usr_utt = dialog[i].split(" | ")[0] + tokenizer.eos_token
            sys_utt = dialog[i].split(" | ")[1] + tokenizer.eos_token
            history_list.append(usr_utt)
            history = "".join(history_list)
            input_ids, input_mask, labels = build_input(
                history, sys_utt, tokenizer)
            features.append(InputFeatures(input_ids, input_mask, None, labels))
            history_list.append(sys_utt)

            if ex_index < 3:
                logger.info(f"*** Example: {ex_index}***")
                logger.info(f"input: {history + sys_utt}")
                logger.info(f"sys_utt: {sys_utt}")
                logger.info(
                    f"input_ids: {' '.join([str(x) for x in input_ids])}")
                logger.info(
                    f"attention_mask: {' '.join([str(x) for x in input_mask])}"
                )
                logger.info(f"labels: {' '.join([str(x) for x in labels])}")
            ex_index += 1
    logger.info('#features: {}', len(features))
    return features


def eval(model, eval_features, batch_size=16, device="cpu"):
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = {}", len(eval_features))
    logger.info("  Batch size = {}", batch_size)

    all_lm_input_ids = torch.tensor([f.input_ids for f in eval_features],
                                    dtype=torch.long)
    all_lm_masks = torch.tensor([f.input_mask for f in eval_features],
                                dtype=torch.long)
    all_lm_labels = torch.tensor([f.label_id for f in eval_features],
                                 dtype=torch.long)

    eval_data = TensorDataset(all_lm_input_ids, all_lm_masks, all_lm_labels)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data,
                                 sampler=eval_sampler,
                                 batch_size=batch_size,
                                 drop_last=False)
    model.eval()
    eval_loss = 0
    nb_ev_steps = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            lm_input, lm_mask, lm_label = batch
            outputs = model(lm_input, attention_mask=lm_mask, labels=lm_label)
            loss = outputs.loss
            loss = loss.mean()
            eval_loss += loss
            nb_ev_steps += 1

        loss = eval_loss / nb_ev_steps
        ppl = torch.exp(loss)
    return loss, ppl


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def eval_bleu(model, tokenizer, eval_dialogs, device):
    logger.info("***** Running evaluation (BLEU) *****")
    logger.info("  Num dialogs = {}", len(eval_dialogs))
    generated_response = []
    reference_response = []

    input_sents = []
    for dialog in eval_dialogs:
        history_list = []
        for i in range(len(dialog)):
            usr_utt = dialog[i].split(" | ")[0] + tokenizer.eos_token
            sys_utt = dialog[i].split(" | ")[1]
            history_list.append(usr_utt)
            history = "".join(history_list)
            input_sents.append(history)
            reference_response.append(sys_utt)
            history_list.append(sys_utt + tokenizer.eos_token)

    # batch generation
    # https://github.com/huggingface/transformers/pull/7552#issue-497255933
    tokenizer.padding_size = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = model.module if hasattr(model, 'module') else model

    # adjust batch size (8) for generation as needed
    for input_batch in chunks(input_sents, 8):
        inputs = tokenizer(input_batch, return_tensors="pt", padding=True)
        # generated a response while limiting the total chat history to 1000 tokens,
        outputs = model.generate(
            inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id)
        generated_res = tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True)
        generated_response.extend(generated_res)
        # TODO: remove this line after debugging
        # logger.error(f"generated response: {generated_res}")

    if EXP_VERBOSE:
        with open(
                "dialog-flow-extraction/out/generaion/generated_response.txt",
                "w") as f1:
            for i in generated_response:
                f1.write(i + "\n")
        with open(
                "dialog-flow-extraction/out/generaion/reference_response.txt",
                "w") as f2:
            for i in reference_response:
                f2.write(i + "\n")
    return corpus_bleu(generated_response, [reference_response]).score


def load_and_cache_features(args, dialogs, tokenizer, mode):
    if os.path.exists(
            os.path.join(args.data_dir,
                         f"cached_{args.test_domain}_{mode}_features.pkl")):
        logger.info(f"Cache exists for {mode} features, loading ...")
        features = torch.load(
            os.path.join(args.data_dir,
                         f"cached_{args.test_domain}_{mode}_features.pkl"))
    else:
        logger.info(f"Converting dialogs to features ...")
        features = convert_dialog_to_features(dialogs, tokenizer)
        torch.save(
            features,
            os.path.join(args.data_dir,
                         f"cached_{args.test_domain}_{mode}_features.pkl"))
    return features


def converse(args):
    try:
        model = load_model(args)
    except Exception:
        logger.error("Loading failed, train the model first!")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") +
                                              tokenizer.eos_token,
                                              return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                                  dim=-1) if step > 0 else new_user_input_ids

        logger.debug(tokenizer.decode(bot_input_ids[0]))

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids,
                                          max_length=1000,
                                          pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(
            tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                             skip_special_tokens=True)))


def save_model(args, model):
    # Save model checkpoint (Overwrite)
    args.model_dir = args.model_dir + f"_{args.test_domain}"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.model_dir)

    # Save training arguments together with the trained model
    torch.save(args, os.path.join(args.model_dir, 'training_args.bin'))
    logger.info(f"Saving model checkpoint to {args.model_dir}")


def load_model(args):
    # Check whether model exists
    if not os.path.exists(args.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir)
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def augment(data, domain, aug_ratio=1.0):
    # data is only the data for the domain
    with open(f"dialog-flow-extraction/out/{domain}_state2text_bert_sbd.json",
              "r") as f:
        state_dict = json.load(f)
    with open(f"dialog-flow-extraction/out/{domain}_states_pred_bert_sbd.txt",
              "r") as f:
        states_pred = f.readlines()
    augmented_data = []
    for i, dialog in enumerate(data):
        new_dialog = {}
        new_dialog["domain"] = dialog["domain"]
        new_dialog["num_label"] = dialog["num_label"]
        new_dialog["text"] = []
        new_dialog["label"] = []
        prev_state = -1
        for state in states_pred[i].split():
            state = int(state)
            if state != prev_state:
                text_list = random.choice(state_dict[str(state)]).split(
                    "###")  # every state has a sequence of texts joint by ###
                for text in text_list:
                    new_dialog["text"].append(text)
                    new_dialog["label"].append(state)
            prev_state = state
        if new_dialog["text"] != dialog[
                "text"] and new_dialog not in augmented_data:
            augmented_data.append(new_dialog)
    aug_size = int(len(data) * aug_ratio)
    # using .choices() instead of .sample() to ensure the ratio is correct
    augmented_data = random.choices(augmented_data, k=aug_size)
    if EXP_VERBOSE:
        with open(
                f"dialog-flow-extraction/data/MultiWOZ_2.1/data_{domain}_augmented.json",
                "w") as f:
            json.dump(augmented_data, f)

    return augmented_data


def most_frequent_sampling(data, domain, aug_ratio=1.0):
    # Conversation Graph: Data Augmentation, Training and Evaluation for Non-Deterministic Dialogue Management
    # https://arxiv.org/pdf/2010.15411.pdf

    # Get ground truth {state: [utterances]}
    text_list = []
    labels_true = []
    for dialog in data:
        dial_text = dialog["text"]
        for turn in dial_text:
            text_list.append(turn)
        labels_true.extend(dialog["label"])

    label_text_dict = {}
    prev_label = labels_true[0]
    act = []
    for text, label in zip(text_list, labels_true):
        if label == prev_label:
            act.append(text)
        else:
            if prev_label in label_text_dict:
                label_text_dict[prev_label].append("###".join(act))
            else:
                label_text_dict[prev_label] = ["###".join(act)]
            act = [text]
        prev_label = label
    # last label
    if label in label_text_dict:
        label_text_dict[label].append("###".join(act))
    else:
        label_text_dict[label] = ["###".join(act)]

    # Get most frequent utterances for each label
    label_text_dict_most_frequent = {
        label: Counter(utt_list).most_common()[0][0]
        for label, utt_list in label_text_dict.items()
    }

    # augment
    augmented_data = []
    for dialog in data:
        new_dialog = {}
        new_dialog["domain"] = dialog["domain"]
        new_dialog["num_label"] = dialog["num_label"]
        new_dialog["text"] = []
        new_dialog["label"] = []
        prev_state = -1
        for state in dialog["label"]:
            if state != prev_state:
                text_list = label_text_dict_most_frequent[state].split(
                    "###")  # every state has a sequence of texts joint by ###
                for text in text_list:
                    new_dialog["text"].append(text)
                    new_dialog["label"].append(state)
            prev_state = state
        if new_dialog["text"] != dialog[
                "text"] and new_dialog not in augmented_data:
            augmented_data.append(new_dialog)
    if EXP_VERBOSE:
        with open(
                f"dialog-flow-extraction/data/MultiWOZ_2.1/data_{domain}_mfs.json",
                "w") as f:
            json.dump(augmented_data, f)

    aug_size = int(len(data) * aug_ratio)
    # using .choices() instead of .sample() to ensure the ratio is correct
    augmented_data = random.choices(augmented_data, k=aug_size)

    return augmented_data


def main(args):
    device = args.device

    # Get data
    with open(os.path.join(args.data_dir, "data_single.json"), "r") as f:
        data = json.load(f)
        domain = args.test_domain
        logger.warning(f"Domain: {domain}")
        data = data[domain]

        # train:valid:test = 60:20:20
        train_idx = int(len(data) * 0.6)
        train_data = data[:int(train_idx * args.train_ratio)]
        val_data = data[train_idx:]
        val_data = val_data[int(len(val_data) * 0.5):]
        test_data = val_data[:int(len(val_data) * 0.5)]

        logger.warning(
            f"Train: {len(train_data)}, Valid: {len(val_data)}, Test: {len(test_data)}"
        )
        if args.augment:
            aug_data = augment(train_data, domain, aug_ratio=args.aug_ratio)
            train_data.extend(aug_data)
            logger.warning(
                f"Augmented Train: {len(train_data)}, Valid: {len(val_data)}, Test: {len(test_data)}"
            )
        if args.mfs:
            aug_data = most_frequent_sampling(train_data,
                                              domain,
                                              aug_ratio=args.aug_ratio)
            train_data.extend(aug_data)
            logger.warning(
                f"Augmented Train: {len(train_data)}, Valid: {len(val_data)}, Test: {len(test_data)}"
            )

    # Compute and visualize the state overlap
    if EXP_VERBOSE:
        num_label = int(data[0]["num_label"])
        trans_adj = np.zeros((num_label, num_label))
        edge_visited = {}  # {(i, j): [train, val, test]}
        for i in range(num_label):
            for j in range(num_label):
                edge_visited[(i, j)] = [False, False, False]
        for i, dialog in enumerate(train_data):
            for t in range(len(dialog["label"]) - 1):
                current_idx = dialog["label"][t]
                next_idx = dialog["label"][t + 1]
                trans_adj[current_idx, next_idx] += 1
                edge_visited[(current_idx, next_idx)][0] = True
        for i, dialog in enumerate(val_data):
            for t in range(len(dialog["label"]) - 1):
                current_idx = dialog["label"][t]
                next_idx = dialog["label"][t + 1]
                trans_adj[current_idx, next_idx] += 1
                edge_visited[(current_idx, next_idx)][1] = True
        for i, dialog in enumerate(test_data):
            for t in range(len(dialog["label"]) - 1):
                current_idx = dialog["label"][t]
                next_idx = dialog["label"][t + 1]
                trans_adj[current_idx, next_idx] += 1
                edge_visited[(current_idx, next_idx)][2] = True

        logger.debug(trans_adj)
        sum_rows = np.sum(trans_adj, axis=1)
        trans_freq = trans_adj / sum_rows[:, np.newaxis]
        trans_freq = np.nan_to_num(trans_freq)
        logger.info(f"State transition adjcency matrix: {trans_freq}")
        # logger.debug(np.sum(trans_freq, axis=1))
        draw_split(trans_freq,
                   edge_visited,
                   domain=domain,
                   save_path=f"image/state_split_{domain}.png")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium").to(device)
    model.config.pad_token_id = model.config.eos_token_id
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if EXP_VERBOSE:
        writer = SummaryWriter(log_dir="dialog-flow-extraction/out/generaion/")

    # Convert data to features, load cached if available
    train_dialogs = [d["text"] for d in train_data]
    val_dialogs = [d["text"] for d in val_data]
    test_dialogs = [d["text"] for d in test_data]

    if args.augment:
        mode = f"_augmented_{args.train_ratio}_{args.aug_ratio}"
    elif args.mfs:
        mode = f"_mfs_{args.train_ratio}_{args.aug_ratio}"
    else:
        mode = f"_{args.train_ratio}"

    if EXP_VERBOSE:
        train_features = load_and_cache_features(args, train_dialogs,
                                                 tokenizer, "train" + mode)
        val_features = load_and_cache_features(args, val_dialogs, tokenizer,
                                               "val" + mode)
        test_features = load_and_cache_features(args, test_dialogs, tokenizer,
                                                "test" + mode)
    else:
        train_features = convert_dialog_to_features(train_dialogs, tokenizer)
        val_features = convert_dialog_to_features(val_dialogs, tokenizer)
        test_features = convert_dialog_to_features(test_dialogs, tokenizer)

    logger.info("***** Running training *****")
    logger.info("  Num examples = {}", len(train_features))
    logger.info("  Batch size = {}", args.batch_size)

    global g_step

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    all_lm_input_ids = torch.tensor([f.input_ids for f in train_features],
                                    dtype=torch.long)
    all_lm_masks = torch.tensor([f.input_mask for f in train_features],
                                dtype=torch.long)
    all_lm_labels = torch.tensor([f.label_id for f in train_features],
                                 dtype=torch.long)

    train_data = TensorDataset(all_lm_input_ids, all_lm_masks, all_lm_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  drop_last=True)

    for e in range(args.num_epochs):
        logger.info(f"Epoch: {e}")
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for _, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            lm_input, lm_mask, lm_label = batch
            outputs = model(lm_input, attention_mask=lm_mask, labels=lm_label)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += lm_input.size(0)
            nb_tr_steps += 1

            optimizer.step()
            model.zero_grad()
            g_step += 1

            if EXP_VERBOSE:
                writer.add_scalar('Loss/train', tr_loss / nb_tr_steps, g_step)
                # eval
                if g_step % args.eval_steps == 0:
                    val_loss, val_ppl = eval(model,
                                             val_features,
                                             batch_size=args.batch_size,
                                             device=device)
                    writer.add_scalar('Loss/eval', val_loss, g_step)
                    writer.add_scalar('Perplexity/eval', val_ppl, g_step)

    test_loss, test_ppl = eval(model,
                               test_features,
                               batch_size=args.batch_size,
                               device=device)
    if EXP_VERBOSE:
        writer.add_scalar('Loss/test', test_loss)
        writer.add_scalar('Perplexity/test', test_ppl)
    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test ppl: {test_ppl}")

    test_bleu = eval_bleu(model, tokenizer, test_dialogs, device=device)
    if EXP_VERBOSE:
        writer.add_scalar("BLEU/test", test_bleu)
        save_model(args, model)
    logger.info(f"Test BLEU: {test_bleu}")

    return test_ppl, test_bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_dir',
                        default="dialog-flow-extraction/data/MultiWOZ_2.1",
                        required=False,
                        help="MultiWOZ dialog data directory path")
    parser.add_argument('--batch_size',
                        default=2,
                        type=int,
                        required=False,
                        help="Train/eval batch size")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu_idx",
                        default=None,
                        type=int,
                        required=False,
                        help="If specified, then use single GPU.")
    parser.add_argument('--test_domain',
                        type=str,
                        default='hotel',
                        help="MultiWOZ domain to test")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('-e',
                        '--num_epochs',
                        type=int,
                        default=5,
                        help="Number of training epochs")
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=10000,  # disable
        help="Eval every n steps")
    parser.add_argument('--model_dir',
                        default="dialog-flow-extraction/out/DialoGPT",
                        required=False,
                        help="Model saving path")
    parser.add_argument("--play",
                        default=False,
                        action='store_true',
                        help="Converse with DialoGPT")
    parser.add_argument("--augment",
                        default=False,
                        action='store_true',
                        help="Whether not to augment original data")

    parser.add_argument(
        "--mfs",
        default=False,
        action='store_true',
        help=
        "Whether not to augment original data with Most Frequent Sampling (MFS)"
    )
    parser.add_argument('--train_ratio',
                        type=float,
                        default=1.0,
                        help="Ratio of original train set to actually use")
    parser.add_argument('--aug_ratio',
                        type=float,
                        default=1.0,
                        help="Ratio of augmented data to the train data used")

    args = parser.parse_args()

    set_seed(666)

    # Set device
    if args.gpu_idx is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda:" + str(args.gpu_idx))
        n_gpu = 1
    logger.info("device {} n_gpu {} ", device, n_gpu)
    args.device = device

    if EXP_VERBOSE:
        if args.play:
            converse(args)
        else:
            main(args)
    else:
        aug_opt = ""
        if args.augment:
            aug_opt = "_aug"
        if args.mfs:
            aug_opt = "_mfs"
        os.makedirs("dialog-flow-extraction/out/generation/", exist_ok=True)
        with open(
                f"dialog-flow-extraction/out/generation/results_{args.test_domain}{aug_opt}.txt",
                "w") as f:
            f.write("train_ratio aug_ratio PPL BLEU\n")
            for i in np.arange(0.2, 1.1, 0.2):
                for j in np.arange(0.0, 1.1, 0.2):
                    args.train_ratio = i
                    args.aug_ratio = j
                    ppl, bleu = main(args)
                    f.write(f"{i:.1f} {j:.1f} {ppl:.2f} {bleu:.2f}\n")
                    f.flush()
