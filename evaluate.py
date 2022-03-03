"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from loguru import logger


def average(v_list):
    return sum(v_list) / len(v_list)


def evaluate_sbd(span_true, span_pred, length):
    """Evaluate slot boundary detection

    Args:
        span_true ([type]): [description]
        span_pred ([type]): [description]
        length ([type]): [description]
    """
    P_token, R_token, F_token = [], [], []
    P_slot, R_slot, F_slot = [], [], []
    for i in range(len(span_true)):
        mask_true, mask_pred = [0] * length, [0] * length
        # token level
        for span in span_true[i]:
            for j in range(span[0], span[1]):
                mask_true[j] = 1
        for span in span_pred[i]:
            for j in range(span[0], span[1]):
                mask_pred[j] = 1

        pt = precision_score(mask_true, mask_pred, zero_division=1)
        P_token.append(pt)
        rt = recall_score(mask_true, mask_pred, zero_division=1)
        R_token.append(rt)
        ft = f1_score(mask_true, mask_pred, zero_division=1)
        F_token.append(ft)

        # slot level
        tp = 0
        for sp in span_pred[i]:
            for st in span_true[i]:
                if sp[0] == st[0] and sp[1] == st[1]:
                    tp += 1
        ps = tp / len(span_pred[i]) if len(
            span_pred[i]) != 0 else 1  # zero_division=1
        rs = tp / len(span_true[i]) if len(
            span_true[i]) != 0 else 1  # zero_division=1
        fs = 2 * ps * rs / (ps + rs) if (ps + rs) != 0 else 1
        P_slot.append(ps)
        R_slot.append(rs)
        F_slot.append(fs)
    return average(P_token), average(R_token), average(F_token), average(
        P_slot), average(R_slot), average(F_slot)


# evaluate results in BIO format (w/o slot name)
def evaluate_sbd_bio(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(slot_preds, slot_labels)

    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_span_from_mask(mask):
    prev = 0
    span = []
    for i in range(len(mask)):
        if mask[i] != 0 and prev == 0:
            start = i
            if i == (len(mask) - 1):  # last token
                span.append((start, start + 1))
        elif mask[i] == 0 and prev != 0:
            end = i
            span.append((start, end))
        prev = mask[i]
    return span


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    P_token, R_token, F_token = [], [], []
    P_slot, R_slot, F_slot = [], [], []

    for i in range(len(preds)):
        # token level
        length = len(preds[i])
        mask_true, mask_pred = [0] * length, [0] * length
        for j in range(length):
            if labels[i][j] != 0:
                mask_true[j] = 1
            if preds[i][j] != 0:
                mask_pred[j] = 1
        pt = precision_score(mask_true, mask_pred, zero_division=1)
        rt = recall_score(mask_true, mask_pred, zero_division=1)
        ft = f1_score(mask_true, mask_pred, zero_division=1)
        P_token.append(pt)
        R_token.append(rt)
        F_token.append(ft)

        # slot level
        span_pred = get_span_from_mask(mask_pred)
        span_true = get_span_from_mask(mask_true)
        tp = 0
        for sp in span_pred:
            for st in span_true:
                if sp[0] == st[0] and sp[1] == st[1]:
                    tp += 1
        ps = tp / len(span_pred) if len(
            span_pred) != 0 else 1  # zero_division=1
        rs = tp / len(span_true) if len(
            span_true) != 0 else 1  # zero_division=1
        fs = 2 * ps * rs / (ps + rs) if (ps + rs) != 0 else 1
        P_slot.append(ps)
        R_slot.append(rs)
        F_slot.append(fs)
    return {
        "token P": sum(P_token) / len(P_token),
        "token R": sum(R_token) / len(R_token),
        "token F1": sum(F_token) / len(F_token),
        "slot P": sum(P_slot) / len(P_slot),
        "slot R": sum(R_slot) / len(R_slot),
        "slot F1": sum(F_slot) / len(F_slot)
    }


def get_sentence_frame_acc(slot_preds, slot_labels):
    """For the cases that all the slots are correct (in one sentence)"""

    # Get the slot comparision result
    slot_result = []
    # i = 0
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
        # if not one_sent_result:
        #     logger.debug(preds)
        #     logger.debug(labels)
        #     logger.debug(i)
        # i += 1
    slot_result = np.array(slot_result)

    sementic_acc = slot_result.mean()
    return {"sementic_frame_acc": sementic_acc}


def clustering_report_gt(labels_true, labels_pred):
    rand_score = metrics.rand_score(labels_true, labels_pred)
    logger.info(f"RI: {rand_score:.3f}")

    adjust_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    logger.info(f"ARI: {adjust_rand_score:.3f}")

    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(
        labels_true, labels_pred)
    logger.info(f"AMI: {adjusted_mutual_info_score:.3f}")


def clustering_report_no_gt(X, labels_true):
    silhouette_score = metrics.silhouette_score(X,
                                                labels_true,
                                                metric='euclidean')
    logger.info(f"Silhouette Coefficient: {silhouette_score:.3f}")
