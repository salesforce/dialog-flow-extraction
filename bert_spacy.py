"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModel, set_seed
from loguru import logger
import spacy
from tqdm import tqdm
from sklearn.cluster import Birch, KMeans, AgglomerativeClustering

import params
from evaluate import clustering_report_gt, evaluate_sbd, average
from util import get_slots


class CustomTokenizer(object):
    """Tokenizer that returns detected noun spans.

    Args:
        object ([type]): [description]
    """
    def __init__(self, config='bert-base-uncased'):
        self.config = config
        # fast tokenizer is required to use word_to_tokens()
        self.tokenizer = AutoTokenizer.from_pretrained(config, use_fast=True)
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, text_list, noun_span_gt):
        if "TODBERT" in self.config:
            tod_text_list = []
            for turn in text_list:
                usr_text = turn.split(" | ")[0]
                sys_text = turn.split(" | ")[1]
                tod_text_list.append(f"[usr] {usr_text} [sys] {sys_text}")
            outputs = self.tokenizer(tod_text_list,
                                     padding=True,
                                     truncation=True,
                                     return_tensors="pt")
        else:
            outputs = self.tokenizer(text_list,
                                     padding=True,
                                     truncation=True,
                                     return_tensors="pt")
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]

        noun_spans = []
        for text in text_list:
            doc = self.nlp(text.split(" | ")
                           [0])  # encode the whole turn but only detect the nouns in user utt
            noun_span = []
            for np in doc.noun_chunks:  # use np instead of np.text
                noun_span.append((np.start, np.end))
            noun_spans.append(noun_span)

        noun_token_spans = []
        for i in range(len(noun_spans)):
            noun_token_spans.append([])
            for j in range(len(noun_spans[i])):
                left_token = outputs.word_to_tokens(i, noun_spans[i][j][0])
                right_token = outputs.word_to_tokens(i,
                                                     noun_spans[i][j][1] - 1)

                if left_token is not None and right_token is not None:
                    if "TODBERT" in self.config:
                        noun_token_spans[i].append(
                            (left_token.start + 1, right_token.end +
                             1))  # add the offset of [usr] token manually
                    else:
                        noun_token_spans[i].append(
                            (left_token.start, right_token.end))

        # mismatch may exist here, since the tokenizer used in MultiWOZ_2.1 "span_info" is unclear
        # slot span ground truth
        noun_token_spans_gt = []
        for i in range(len(noun_span_gt)):
            noun_token_spans_gt.append([])
            for j in range(len(noun_span_gt[i])):
                left_token = outputs.word_to_tokens(i, noun_span_gt[i][j][0])
                right_token = outputs.word_to_tokens(i,
                                                     noun_span_gt[i][j][1] - 1)
                if left_token is not None and right_token is not None:
                    if "TODBERT" in self.config:
                        noun_token_spans_gt[i].append(
                            (left_token.start + 1, right_token.end +
                             1))  # add the offset of [usr] token manually
                    else:
                        noun_token_spans_gt[i].append(
                            (left_token.start, right_token.end))

        return input_ids, attention_mask, noun_token_spans, noun_token_spans_gt


def main(args):
    # TODO: multiple GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(args.seed)

    tokenizer = CustomTokenizer(args.bert_config)
    model = AutoModel.from_pretrained(args.bert_config).to(device)
    model.eval()

    domain_slot_dict, _ = get_slots(
        "dialog-flow-extraction/data/MultiWOZ_2.1/slot_descriptions.json")

    with open("dialog-flow-extraction/data/MultiWOZ_2.1/data_single.json",
              "r") as f:
        data = json.load(f)
        for domain in params.domain:
            logger.warning(f"Domain: {domain}")
            noun_vecs = []
            num_nouns = []
            labels_true = []
            P_token, R_token, F_token = [], [], []
            P_slot, R_slot, F_slot = [], [], []
            num_slots = len(domain_slot_dict[domain])
            logger.warning(f"#slots: {num_slots}")

            for dialog in tqdm(data[domain]):
                labels_true.extend(dialog["label"])
                utt_list = dialog["text"]

                input_ids, attention_mask, noun_spans, noun_spans_gt = tokenizer(
                    utt_list, dialog["slot_span"])
                # per dialog
                pt, rt, ft, ps, rs, fs = evaluate_sbd(noun_spans,
                                                      noun_spans_gt,
                                                      input_ids.shape[1])
                P_token.append(pt)
                R_token.append(rt)
                F_token.append(ft)
                P_slot.append(ps)
                R_slot.append(rs)
                F_slot.append(fs)

                # record number of nouns detected per turn
                for noun_span in noun_spans:
                    num_nouns.append(len(noun_span))
                with torch.no_grad():
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)

                    last_hidden_state = outputs.last_hidden_state
                    for i, noun_span in enumerate(noun_spans):
                        for span in noun_span:
                            # average the hidden states of the detected noun tokens
                            # TODO: prevent the noun from being split
                            noun_state = torch.mean(
                                last_hidden_state[i, span[0]:span[1]], dim=0)
                            noun_vecs.append(noun_state)
            noun_vecs = torch.stack(noun_vecs)
            # logger.debug(noun_vecs.shape)
            logger.info(f"Total slot tokens detected: {noun_vecs.shape[0]}")
            logger.info(
                f"SBD (token) P: {average(P_token):.2f}, R: {average(R_token):.2f}, F1: {average(F_token):.2f}"
            )
            logger.info(
                f"SBD (slot) P: {average(P_slot):.2f}, R: {average(R_slot):.2f}, F1: {average(F_slot):.2f}"
            )

            if args.clustering == "birch":
                cluster_model = Birch(n_clusters=num_slots)
            elif args.clustering == "kmeans":
                cluster_model = KMeans(n_clusters=num_slots,
                                       random_state=args.seed)
            elif args.clustering == "agg":
                cluster_model = AgglomerativeClustering(n_clusters=num_slots)
            else:
                logger.error("Clustering model not supported!")

            noun_vecs = noun_vecs.cpu().detach().numpy()
            slots_pred = cluster_model.fit_predict(noun_vecs)
            # logger.debug(
            #     f"slots_pred ({len(slots_pred)}): {slots_pred}")
            # logger.debug(f"num_nouns ({sum(num_nouns)}): {num_nouns}")
            # logger.debug(f"num_utterance: {len(num_nouns)}")

            # determine info state from slot clustering results
            cnt_utt = 0
            cnt_slot = 0
            unique_states = []
            info_states = []
            for dialog in data[domain]:
                info_state = [0] * num_slots
                for _ in dialog["text"]:
                    num_noun = num_nouns[cnt_utt]
                    cnt_utt += 1
                    for _ in range(num_noun):
                        info_state[slots_pred[
                            cnt_slot]] += 1  # modification time of a single slot
                        cnt_slot += 1
                    info_states.append(info_state.copy())
                    if info_state not in unique_states:
                        unique_states.append(info_state.copy())

            labels_pred = [unique_states.index(state) for state in info_states]
            # logger.debug(len(labels_pred))
            # logger.debug(len(labels_true))
            assert len(labels_true) == len(labels_pred)
            clustering_report_gt(labels_true, labels_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=666,
                        help="random seed for initialization")
    parser.add_argument('-c',
                        '--clustering',
                        type=str,
                        default='kmeans',
                        help="clustering model")
    parser.add_argument('--bert-config',
                        type=str,
                        default='TODBERT/TOD-BERT-JNT-V1',
                        help="BERT model config name")

    args = parser.parse_args()
    print(args)
    main(args)