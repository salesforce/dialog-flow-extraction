"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import random
import json
import argparse

from loguru import logger

import params
from evaluate import clustering_report_gt


def main(args):
    random.seed(args.seed)

    with open("dialog-flow-extraction/data/MultiWOZ_2.1/data_single.json",
              "r") as f:
        data = json.load(f)
        for domain in params.domain:
            logger.warning(f"Domain: {domain}")
            dialogs = data[domain]
            logger.warning(f"#dialogs: {len(dialogs)}")
            labels_true = []
            num_cluster = dialogs[0]["num_label"]
            logger.warning(f"#clusters: {num_cluster}")
            for dialog in dialogs:
                labels_true.extend(dialog["label"])
            # logger.debug(len(labels_true))

            # Random baseline
            labels_pred = random.choices(range(0, num_cluster),
                                         k=len(labels_true))
            clustering_report_gt(labels_true, labels_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=666,
                        help="random seed for initialization")

    args = parser.parse_args()
    print(args)
    main(args)