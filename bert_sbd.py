import os
import json
import argparse
from collections import Counter
from matplotlib.pyplot import text
import itertools
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertPreTrainedModel, BertModel, set_seed, AutoTokenizer
from torchcrf import CRF
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, KMeans, AgglomerativeClustering

from util import get_slots, visualize, detect_cycle, detect_community_and_draw
from evaluate import clustering_report_gt, clustering_report_no_gt

logger.remove()
logger.add(sys.stderr, level="INFO")


class SbdBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(SbdBERT, self).__init__(config)
        self.args = args
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear = nn.Linear(config.hidden_size, 3)
        self.activation = nn.Softmax(dim=2)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "TODBERT/TOD-BERT-JNT-V1")

        if args.use_crf:
            logger.warning("using crf")
            self.crf = CRF(num_tags=3, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids,
                slot_labels_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )  # last_hidden_state, pooled_output, (hidden_states), (attentions)
        last_hidden_state = outputs[0]
        slot_logits = self.linear(self.dropout(last_hidden_state))
        slot_prob = torch.sum(self.activation(slot_logits)[:, :, 1:],
                              dim=2)  # Add the probs of being B- and I-
        slot_prob = attention_mask * slot_prob
        utt_state = torch.mul(slot_prob.unsqueeze(-1), last_hidden_state)
        utt_state = torch.mean(utt_state, dim=1)
        # logger.debug(last_hidden_state[0])
        # logger.debug(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
        # logger.debug(slot_prob[0])
        # logger.debug(utt_state)

        total_loss = 0
        # Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits,
                                     slot_labels_ids,
                                     mask=attention_mask.byte(),
                                     reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(
                    ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, 3)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, 3),
                                              slot_labels_ids.view(-1))
            total_loss += slot_loss

        outputs = (total_loss, slot_logits, utt_state, last_hidden_state,
                   attention_mask) + outputs[
                       2:]  # add hidden states and attention if they are here

        return outputs  # loss, logits, utt_state, hidden_states, attentions


def main(args):
    # TODO: multiple GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_seed(args.seed)
    domain_slot_dict, _ = get_slots(
        "dialog-flow-extraction/data/MultiWOZ_2.1/slot_descriptions.json")

    tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
    # Check whether model exists
    if not os.path.exists(args.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = SbdBERT.from_pretrained(args.model_dir, args=args)
        model.to(device)
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    with open("dialog-flow-extraction/data/MultiWOZ_2.1/data_single.json",
              "r") as f:
        data = json.load(f)
        domain = args.test_domain
        logger.warning(f"Domain: {domain}")
        num_slots = len(domain_slot_dict[domain])
        logger.warning(f"#slots: {num_slots}")

        dialogs = data[domain]
        logger.warning(f"#dialogs: {len(dialogs)}")
        text_list = []
        labels_true = []
        # TODO: try adjusting this
        num_cluster = dialogs[0]["num_label"]
        logger.warning(f"#clusters: {num_cluster}")
        for dialog in dialogs:
            dial_text = dialog["text"]
            for turn in dial_text:
                usr_text = f"[usr] {turn.split(' | ')[0]}"
                sys_text = f"[sys] {turn.split(' | ')[1]}"
                text_list.append(usr_text + " " + sys_text)
            labels_true.extend(dialog["label"])
        # logger.debug(len(text_list))
        # logger.debug(len(labels_true))
        # logger.debug(text_list[0])

        # BERT_clustering
        inputs = tokenizer(
            text_list,
            padding='max_length',
            max_length=50,  # TODO
            truncation=True,
            return_tensors="pt")
        sys_token_id = tokenizer.convert_tokens_to_ids("[sys]")
        idx = torch.cumsum((inputs["input_ids"] == sys_token_id), 1)
        slot_logits_mask = (~idx.bool()).long()

        eval_data = TensorDataset(inputs["input_ids"],
                                  inputs["attention_mask"], slot_logits_mask)
        sampler = SequentialSampler(eval_data)
        dataloader = DataLoader(eval_data,
                                sampler=sampler,
                                batch_size=args.eval_batch_size,
                                drop_last=False)
        for i in range(3):
            logger.info("*** Example ***")
            logger.info(f"id: {i}")
            logger.info(f"tokens: {text_list[i]}")
            logger.info(f"input_ids: {inputs['input_ids'][i]}")
            logger.info(f"attention_mask: {inputs['attention_mask'][i]}")
            logger.info(f"slot_logits_mask: {slot_logits_mask[i]}")

        model.eval()
        noun_vecs = []
        num_nouns = []
        nouns = []
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                input_ids, attention_mask, slot_logits_mask = batch
                outputs = model(input_ids.to(device),
                                attention_mask=attention_mask.to(device),
                                token_type_ids=None,
                                slot_labels_ids=None)
                if args.cluster_utt:
                    utt_state = outputs[2]
                    if step == 0:
                        text_vecs = utt_state
                    else:
                        text_vecs = torch.cat([text_vecs, utt_state], dim=0)
                else:
                    slot_logits = outputs[1].detach().cpu().numpy()
                    slot_preds = np.argmax(slot_logits, axis=2)
                    slot_logits_mask = slot_logits_mask.numpy()
                    slot_preds = slot_preds * slot_logits_mask
                    last_hidden_state = outputs[3]
                    noun_vec = []
                    noun = []
                    for i in range(slot_preds.shape[0]):
                        num_noun_utt = 0
                        for j in range(slot_preds.shape[1]):
                            # TODO: separate consecute slots detected
                            if slot_preds[i, j] != 0:
                                if noun_vecs == []:
                                    noun_vec = [last_hidden_state[i, j]]
                                    noun = [
                                        tokenizer.convert_ids_to_tokens(
                                            input_ids[i, j].item())
                                    ]
                                else:
                                    noun_vec.append(last_hidden_state[i, j])
                                    noun.append(
                                        tokenizer.convert_ids_to_tokens(
                                            input_ids[i, j].item()))
                            else:
                                if noun_vec != []:
                                    noun_state = torch.mean(torch.stack(
                                        noun_vec, dim=0),
                                                            dim=0)
                                    noun_vecs.append(noun_state)
                                    nouns.append(
                                        tokenizer.convert_tokens_to_string(
                                            noun))
                                    num_noun_utt += 1
                                    noun_vec = []
                                    noun = []
                        # logger.debug(
                        #     f"utt: {tokenizer.convert_ids_to_tokens(input_ids[i])}"
                        # )
                        # logger.debug(f"slot_preds: {slot_preds[i]}")
                        # logger.debug(f"num_noun_utt: {num_noun_utt}")
                        # logger.debug(f"noun_vecs: {len(noun_vecs)}")
                        # logger.debug(f"nouns: {nouns}")
                        num_nouns.append(num_noun_utt)
        # logger.debug(text_vecs.shape)
        if args.cluster_utt:
            if args.clustering == "birch":
                cluster_model = Birch(n_clusters=num_cluster)
            elif args.clustering == "kmeans":
                cluster_model = KMeans(n_clusters=num_cluster,
                                       random_state=args.seed)
            elif args.clustering == "agg":
                cluster_model = AgglomerativeClustering(n_clusters=num_cluster)
            else:
                logger.error("Clustering model not supported!")

            text_vecs = text_vecs.cpu().detach().numpy()
            labels_pred = cluster_model.fit_predict(text_vecs)
            # logger.debug(len(labels_pred))
            clustering_report_gt(labels_true, labels_pred)
            clustering_report_no_gt(text_vecs, labels_pred)
        else:
            noun_vecs = torch.stack(noun_vecs)
            logger.info(f"Total slot tokens detected: {noun_vecs.shape[0]}")
            # logger.debug(f"num_nouns ({sum(num_nouns)}): {num_nouns}")
            logger.warning(f"num_utterance: {len(num_nouns)}")

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
            #     f"predict_slots ({len(predict_slots)}): {predict_slots}")
            slot_cluster_dict = {}
            for i in range(num_slots):
                slot_cluster_dict[i] = []
            for i in range(len(nouns)):
                slot_cluster_dict[slots_pred[i]].append(nouns[i])
            # logger.debug(slot_cluster_dict)
            for i in range(num_slots):
                logger.info(
                    f"slot [{i}]: {Counter(slot_cluster_dict[i]).most_common(10)}"
                )

            # determine info state from slot clustering results
            cnt_utt = 0
            cnt_slot = 0
            unique_states = []
            info_states = []
            info_state_values = []

            for dialog in data[domain]:
                info_state_value = [""] * num_slots
                info_state = [0] * num_slots
                # logger.debug("===================")
                for turn in dialog["text"]:
                    num_noun = num_nouns[cnt_utt]
                    cnt_utt += 1
                    for _ in range(num_noun):
                        slot = slots_pred[cnt_slot]
                        if nouns[cnt_slot] != info_state_value[slot]:
                            info_state_value[slot] = nouns[cnt_slot]
                            info_state[
                                slot] += 1  # modification time of a single slot
                        cnt_slot += 1
                    # logger.debug(f"text: {turn}")
                    # logger.debug(f"info_state: {info_state}")
                    # logger.debug(f"info_state_value: {info_state_value}")
                    info_states.append(info_state.copy())
                    info_state_values.append(info_state_value.copy())
                    if info_state not in unique_states:
                        unique_states.append(info_state.copy())

            labels_pred = [unique_states.index(state) for state in info_states]
            # logger.debug(len(labels_pred))
            # logger.debug(len(labels_true))
            assert len(labels_true) == len(labels_pred)
            clustering_report_gt(labels_true, labels_pred)
            # for i in range(10):
            #     logger.debug(text_list[i])
            #     logger.debug(f"predict state: {unique_states[labels_pred[i]]}")
            #     logger.debug(f"#slots detected: {num_nouns[i]}")
            #     logger.debug(f"predict label: {labels_pred[i]}")
            #     logger.debug(f"true label: {labels_true[i]}")

            # visualize the state graph for the test domain
            trans_adj = np.zeros(
                (len(unique_states) + 1, len(unique_states) + 1))
            cnt = 0
            for dialog in data[domain]:
                for i in range(len(dialog["state"])):
                    if i != len(dialog["state"]) - 1:
                        trans_adj[labels_pred[cnt], labels_pred[cnt + 1]] += 1
                    else:
                        trans_adj[labels_pred[cnt],
                                  -1] += 1  # manually add the end state
                    cnt += 1
            assert cnt == len(labels_pred)
            # logger.debug(trans_adj)
            sum_rows = np.sum(trans_adj, axis=1)
            trans_freq = trans_adj / sum_rows[:, np.newaxis]
            trans_freq = np.nan_to_num(trans_freq)

            # logger.info(f"State transition adjcency matrix: {trans_freq}")
            logger.info(
                f"Any cycle in the graph: {detect_cycle(trans_adj, self_loop=False)}"
            )
            # logger.debug(np.sum(trans_freq, axis=1))
            visualize(
                trans_freq,
                unique_states,
                threshold=0.0,
                save_path=
                f"dialog-flow-extraction/image/{domain}_{len(unique_states)}_bert_sbd.png"
            )
            logger.info(
                f"writing image to dialog-flow-extraction/image/{domain}_{len(unique_states)}_bert_sbd.png"
            )

            # # community detection
            # community, color = detect_community_and_draw(
            #     trans_freq,
            #     k=10,
            #     domain=args.test_domain,
            #     save_path=
            #     f"dialog-flow-extraction/image/{domain}_{len(unique_states)}_bert_sbd_10.png"
            # )
            # logger.info(
            #     f"writing image to dialog-flow-extraction/image/{domain}_{len(unique_states)}_bert_sbd_10.png"
            # )
            # logger.info(f"community: {community}")
            # logger.info(f"color: {color}")

            # save {state: text}
            label_text_dict = {}
            cnt = 0
            for dialog in dialogs:
                dial_text = dialog["text"]
                prev_label = labels_pred[cnt]
                act = []
                for turn in dial_text:
                    label = labels_pred[cnt]
                    if label == prev_label:
                        act.append(turn)
                    else:
                        if prev_label in label_text_dict:
                            label_text_dict[prev_label].append("###".join(act))
                        else:
                            label_text_dict[prev_label] = ["###".join(act)]
                        act = [turn]
                    prev_label = label
                    cnt += 1
                # last label in a dialog session
                if label in label_text_dict:
                    label_text_dict[label].append("###".join(act))
                else:
                    label_text_dict[label] = ["###".join(act)]
            # TODO: check if label_text_dict is correct
            with open(
                    f"dialog-flow-extraction/out/{domain}_state2text_bert_sbd.json",
                    "w") as f:
                json.dump(label_text_dict, f)

            # # for analysis
            # for i in label_text_dict.keys():
            #     label_text_dict[i] = label_text_dict[
            #         i][:5]  # write the first 5 to csv
            # df = pd.DataFrame.from_dict(label_text_dict, orient='index')
            # df.transpose()
            # df.to_csv(
            #     f"dialog-flow-extraction/out/{domain}_{len(unique_states)}_bert_sbd.csv"
            # )

            # print a few dialog example and prediction, and save the prediction results
            cnt = 0
            with open(
                    f"dialog-flow-extraction/out/{domain}_states_pred_bert_sbd.txt",
                    "w") as f:
                for i, dialog in enumerate(data[domain]):
                    states = []
                    for _ in range(len(dialog["state"])):
                        # if i < 3:
                        #     logger.warning("============")
                        #     logger.warning(text_list[cnt])
                        #     logger.warning(
                        #         f"predicted label: {labels_pred[cnt]} {info_states[cnt]} {info_state_values[cnt]}"
                        #     )
                        #     logger.warning(
                        #         f"predicted community: {int(color[labels_pred[cnt]])}"
                        #     )
                        states.append(str(labels_pred[cnt]))
                        cnt += 1
                    # dialog_text = "###".join(dialog["text"])
                    # f.write(" ".join(states) + "@" + dialog_text)
                    f.write(" ".join(states))
                    f.write("\n")

            # # Causal finding based on heuristics
            # cnt = 0
            # all_labels = []
            # for dialog in data[domain]:
            #     labels = [-1]
            #     for _ in range(len(dialog["state"])):
            #         if labels_pred[cnt] != labels[-1]:
            #             labels.append(labels_pred[cnt])
            #         cnt += 1
            #     labels.append(-2)
            #     all_labels.append(labels)
            #     logger.warning(labels)

            # causal_dict, result_causal_dict = {}, {}
            # max_gram = 0
            # for label in all_labels:
            #     if len(label) - 2 > max_gram:
            #         max_gram = len(label) - 2
            # for gram in range(1, max_gram + 1):
            #     causal_dict[gram] = {}
            #     result_causal_dict[gram] = {}
            # logger.debug(causal_dict)
            # for label in all_labels:
            #     logger.debug(label)
            #     for gram in range(1, len(label) - 1):
            #         for i in range(1, len(label) - gram):
            #             key = tuple(label[i:i + gram])
            #             head = label[i - 1]
            #             tail = label[i + gram]
            #             # logger.debug(f"key: {key}, head: {head}, tail: {tail}")
            #             if key in causal_dict[gram]:
            #                 if (head, tail) not in causal_dict[gram][key] and (
            #                         head, tail) != (-1, -2):
            #                     causal_dict[gram][key].append((head, tail))
            #             elif (head, tail) != (-1, -2):
            #                 causal_dict[gram][key] = [(head, tail)]
            # logger.debug(causal_dict)

            # for gram in range(1, max_gram):
            #     for pair in itertools.combinations(causal_dict[gram].keys(),
            #                                        2):
            #         common_head_tail = list(
            #             set(causal_dict[gram][pair[0]])
            #             & set(causal_dict[gram][pair[1]]))
            #         if common_head_tail != [(-1, -2)
            #                                 ] and common_head_tail != []:
            #             result_causal_dict[gram][tuple(
            #                 pair)] = common_head_tail
            #             if gram > 1:
            #                 logger.error(pair)
            #                 logger.error(common_head_tail)
            # logger.debug(result_causal_dict)
            # for gram in range(1, max_gram):
            #     result_causal_dict[gram] = list(
            #         result_causal_dict[gram].keys())
            #     logger.warning(f"{gram}-gram: {result_causal_dict[gram]}")
            #     for pair in result_causal_dict[gram]:
            #         logger.warning(f"==================")
            #         for state in pair[0]:
            #             logger.warning(random.choice(label_text_dict[state]))
            #             logger.warning(f"state: {unique_states[state]}")
            #         logger.warning(f"-------------")
            #         for state in pair[1]:
            #             logger.warning(random.choice(label_text_dict[state]))
            #             logger.warning(f"state: {unique_states[state]}")


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
    parser.add_argument("--model_dir",
                        default="dialog-flow-extraction/out/todbert_sbd",
                        type=str,
                        help="Path to save, load model")
    parser.add_argument("--dropout_rate",
                        default=0.1,
                        type=float,
                        help="Dropout for fully-connected layers")
    parser.add_argument("--use_crf",
                        action="store_true",
                        help="Whether to use CRF")
    parser.add_argument("--cluster_utt",
                        action="store_true",
                        help="Whether to cluter utt or slot")
    parser.add_argument('--test_domain',
                        type=str,
                        default='hotel',
                        help="MultiWOZ domain to test")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Batch size for evaluation.")

    args = parser.parse_args()
    print(args)
    main(args)
