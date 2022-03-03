"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModel, set_seed
import numpy as np
from loguru import logger
from sklearn.cluster import Birch, KMeans, AgglomerativeClustering

import params
from util import visualize, get_slots, detect_cycle
from evaluate import clustering_report_gt, clustering_report_no_gt


def BERT_clustering(text_list,
                    tokenizer,
                    model,
                    num_cluster,
                    eval_batch_size=128,
                    clustering="kmeans",
                    device="cpu",
                    random_seed=666):
    inputs = tokenizer(text_list,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")

    eval_data = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    sampler = SequentialSampler(eval_data)
    dataloader = DataLoader(eval_data,
                            sampler=sampler,
                            batch_size=eval_batch_size,
                            drop_last=False)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            pooler_output = outputs.pooler_output  # [CLS] token
            if step == 0:
                text_vecs = pooler_output
            else:
                text_vecs = torch.cat([text_vecs, pooler_output], dim=0)
    # logger.info(text_vecs.shape)

    if clustering == "birch":
        cluster_model = Birch(n_clusters=num_cluster)
    elif clustering == "kmeans":
        cluster_model = KMeans(n_clusters=num_cluster,
                               random_state=random_seed)
    elif clustering == "agg":
        cluster_model = AgglomerativeClustering(n_clusters=num_cluster)
    else:
        logger.error("Clustering model not supported!")

    text_vecs = text_vecs.cpu().detach().numpy()
    labels_pred = cluster_model.fit_predict(text_vecs)

    return labels_pred, text_vecs


def main(args):
    # TODO: multi GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training with {device}")

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_config)
    model = AutoModel.from_pretrained(args.bert_config).to(device)

    domain_slot_dict, all_slots = get_slots(
        "dialog-flow-extraction/data/MultiWOZ_2.1/slot_descriptions.json")

    with open("dialog-flow-extraction/data/MultiWOZ_2.1/data_single.json",
              "r") as f:
        data = json.load(f)
        for domain in params.domain:
            logger.warning(f"Domain: {domain}")
            dialogs = data[domain]
            logger.warning(f"#dialogs: {len(dialogs)}")
            text_list = []
            labels_true = []
            # TODO: try adjusting this
            num_cluster = dialogs[0]["num_label"]
            logger.warning(f"#clusters: {num_cluster}")
            for dialog in dialogs:
                dial_text = dialog["text"]
                if "TODBERT" in args.bert_config:
                    for turn in dial_text:
                        usr_text = turn.split(" | ")[0]
                        sys_text = turn.split(" | ")[1]
                        text_list.append(f"[usr] {usr_text} [sys] {sys_text}")
                else:
                    text_list.extend(dialog["text"])
                labels_true.extend(dialog["label"])
            # logger.debug(len(text_list))
            # logger.debug(len(labels_true))
            assert len(text_list) == len(labels_true)

            # BERT_clustering with [CLS]
            labels_pred, text_vecs = BERT_clustering(
                text_list,
                tokenizer,
                model,
                num_cluster,
                eval_batch_size=args.eval_batch_size,
                clustering=args.clustering,
                device=device,
                random_seed=args.seed)
            # logger.debug(len(labels_pred))
            assert len(labels_true) == len(labels_pred)
            clustering_report_gt(labels_true, labels_pred)
            clustering_report_no_gt(text_vecs, labels_pred)

            # visualize the state graph for the test domain
            states = []
            for dialog in data[domain]:
                for turn in dialog["state"]:
                    state = [turn[s][1] for s in domain_slot_dict[domain]]
                    if state not in states:
                        states.append(state)
            logger.info(f"Domain: {domain}, Number of states: {len(states)}")
            assert num_cluster == len(states)
            trans_adj = np.zeros((num_cluster, num_cluster))
            cnt = 0
            for dialog in data[domain]:
                for _ in range(len(dialog["state"]) - 1):
                    trans_adj[labels_pred[cnt], labels_pred[cnt + 1]] += 1
                    cnt += 1
                cnt += 1  # skip the end of a dialog
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
                states,  #[str(i) for i in list(range(num_cluster))],
                threshold=0.0,
                save_path=
                f"dialog-flow-extraction/image/{domain}_{num_cluster}_bert.png"
            )


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
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--bert-config',
                        type=str,
                        default='TODBERT/TOD-BERT-JNT-V1',
                        help="BERT model config name")

    args = parser.parse_args()
    print(args)
    main(args)