"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import copy

import numpy as np
import pandas as pd
from loguru import logger

from util import visualize, get_slots
import params


def main():
    domain_slot_dict, all_slots = get_slots(
        "dialog-flow-extraction/data/MultiWOZ_2.1/slot_descriptions.json")

    data_single = {}
    data_mul = []
    # read dialog data
    with open("dialog-flow-extraction/data/MultiWOZ_2.1/data.json", "r") as f:
        data = json.load(f)
        for _, dial in data.items():
            # data to write in the new format
            domain = []
            text = []
            state_list = []
            slot_span = []
            # get the domain (single/multi) of the session
            for d in params.domain:
                if dial['goal'][d] != {}:
                    domain.append(d)

            # intialize info_state
            info_state = {}
            if len(domain) == 1:
                for slot in domain_slot_dict[domain[0]]:
                    info_state[slot] = ["", 0]  # count the modification
            else:
                for slot in all_slots:
                    info_state[slot] = ["", 0]  # count the modification

            # read metadata into info_state
            slot_span = []
            for t, turn in enumerate(dial["log"]):
                if t % 2 == 1:  # system turn
                    text.append(dial["log"][t - 1]["text"] + " | " +
                                turn["text"])
                else:  # user turn
                    s_span = []
                    if "span_info" in turn:
                        for sp in turn["span_info"]:
                            if sp[0].split("-")[0].lower() in domain_slot_dict:
                                slot_cands = domain_slot_dict[sp[0].split("-")
                                                              [0].lower()]
                                for cand in slot_cands:
                                    if sp[1].lower() in cand:
                                        s_span.append(
                                            (sp[-2], sp[-1] + 1, cand))
                    slot_span.append(s_span)

                if turn["metadata"] != {}:
                    for d in domain:
                        for slot_type, slots in turn["metadata"][d].items():
                            for slot, value in slots.items():
                                slot_name = "-".join([d, slot_type,
                                                      slot]).lower()
                                # logger.debug(slot_name)
                                if slot_name in info_state and value != info_state[
                                        slot_name][
                                            0] and value != "not mentioned":
                                    info_state[slot_name][0] = value
                                    info_state[slot_name][1] += 1
                    state_list.append(copy.deepcopy(info_state))
                    # logger.debug(state_list)

            new_dial = {
                "domain": domain,
                "text": text,
                "state": state_list,
                "slot_span": slot_span
            }
            if len(text) != len(state_list) or len(text) != len(slot_span):
                logger.warning("Skip bad log")
            else:
                if len(domain) == 1:
                    if domain[0] not in data_single:
                        data_single[domain[0]] = [new_dial]
                    else:
                        data_single[domain[0]].append(new_dial)
                else:
                    data_mul.append(new_dial)
    # logger.debug(data_single["taxi"])

    # visualize the state graph for each single domain
    for d in params.domain:
        states = []
        for dialog in data_single[d]:
            for turn in dialog["state"]:
                state = [turn[s][1] for s in domain_slot_dict[d]]
                if state not in states:
                    states.append(state)
        logger.info(f"Domain: {d}, Number of states: {len(states)}")
        # logger.debug(states)

        trans_adj = np.zeros((len(states)+1, len(states)+1))
        for dialog in data_single[d]:
            label = []
            for t in range(len(dialog["state"]) - 1):
                current_state = [
                    dialog["state"][t][s][1] for s in domain_slot_dict[d]
                ]
                current_idx = states.index(current_state)
                next_state = [
                    dialog["state"][t + 1][s][1] for s in domain_slot_dict[d]
                ]
                next_idx = states.index(next_state)
                trans_adj[current_idx, next_idx] += 1
                label.append(current_idx)
            trans_adj[next_idx, -1] += 1 # manually add end state
            label.append(next_idx)
            dialog["label"] = label
            dialog["num_label"] = len(states)
            assert len(dialog["text"]) == len(dialog["label"])
        # logger.debug(trans_adj)
        sum_rows = np.sum(trans_adj, axis=1)
        trans_freq = trans_adj / sum_rows[:, np.newaxis]
        trans_freq = np.nan_to_num(trans_freq)

        logger.info(f"State transition adjcency matrix: {trans_freq}")
        # logger.debug(np.sum(trans_freq, axis=1))

        visualize(
            trans_freq,
            states,
            threshold=0.0,
            save_path=f"dialog-flow-extraction/image/{d}_{len(states)}.png")

    # visualize the state graph for multi domain
    states = []
    for dialog in data_mul:
        for turn in dialog["state"]:
            state = [turn[s][1] for s in all_slots]
            if state not in states:
                states.append(state)
    logger.info(f"Multi domain together, Number of states: {len(states)}")
    # logger.debug(states)

    trans_adj = np.zeros((len(states), len(states)))
    for dialog in data_mul:
        label = []
        for t in range(len(dialog["state"]) - 1):
            current_state = [dialog["state"][t][s][1] for s in all_slots]
            current_idx = states.index(current_state)
            next_state = [dialog["state"][t + 1][s][1] for s in all_slots]
            next_idx = states.index(next_state)
            trans_adj[current_idx, next_idx] += 1
            label.append(current_idx)
        label.append(next_idx)
        dialog["label"] = label
        dialog["num_label"] = len(states)
    # logger.debug(trans_adj)
    sum_rows = np.sum(trans_adj, axis=1)
    trans_freq = trans_adj / sum_rows[:, np.newaxis]
    trans_freq = np.nan_to_num(trans_freq)

    logger.info(f"State transition adjcency matrix: {trans_freq}")
    # logger.debug(np.sum(trans_freq, axis=1))

    # Uncomment to visualize the graph
    # visualize(trans_freq, states, threshold=0.0)

    with open('dialog-flow-extraction/data/MultiWOZ_2.1/data_single.json',
              'w') as f:
        json.dump(data_single, f)
    with open('dialog-flow-extraction/data/MultiWOZ_2.1/data_mul.json',
              'w') as f:
        json.dump(data_mul, f)

    # Process data for MLM pretraining
    with open("dialog-flow-extraction/data/MultiWOZ_2.1/data_single.json",
              "r") as f:
        data = json.load(f)
        for d in params.domain:
            all_data = []
            for dial in data[d]:
                utt_list = dial["text"]
                dial_mlm = ""
                for turn in utt_list:
                    text = turn.split(" | ")
                    dial_mlm += f"[USR] {text[0]} [SYS] {text[1]}"
                    # logger.debug(dial_mlm)
                all_data.append(dial_mlm)
            df = pd.DataFrame({"text": all_data})
            df.to_csv(f"dialog-flow-extraction/data/MultiWOZ_2.1/{d}.csv")


if __name__ == "__main__":
    main()