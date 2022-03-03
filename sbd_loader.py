"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import copy
import json

import spacy
import torch
from torch.utils.data import TensorDataset
from loguru import logger

import params


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        slot_labels: (Optional) list. The slot labels of the example.
    """
    def __init__(self, guid, words, slot_labels=None):
        self.guid = guid
        self.words = words
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids,
                 slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    """Processor for the SbdBERT data set """
    def __init__(self, args):
        self.args = args
        if args.task == "MultiWOZ_2.1":
            self.data_file = 'data_single.json'
        elif args.task == "ATIS" or args.task == 'SNIPS':
            self.data_file = 'seq.in'
            self.label_file = 'seq.out'

        self.tokenizer = spacy.load("en_core_web_sm").tokenizer

    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        return data

    def _tokenize(self, text):
        doc = self.tokenizer(text)
        words = [t.text for t in doc]
        return words

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        i = 0

        if set_type == "train":
            for domain in params.train_domains:
                for _, dialog in enumerate(data[domain]):
                    for turn in range(len(dialog["text"])):
                        guid = "%s-%s" % (set_type, i)
                        i += 1
                        # 1. input_text
                        usr_text = ["[usr]"] + self._tokenize(
                            dialog["text"][turn].split(" | ")[0])
                        sys_text = ["[sys]"] + self._tokenize(
                            dialog["text"][turn].split(" | ")[1])
                        # 2. slot
                        slot_labels = [0] * len(usr_text)  # O:0
                        for slot in dialog["slot_span"][turn]:
                            cnt = 0
                            for t in range(slot[0], slot[1]):  # B:1, I:2
                                slot_labels[
                                    t +
                                    1] = 1 if cnt == 0 else 2  # manually add the offset of "[usr]" token
                                cnt += 1
                        words = usr_text + sys_text
                        # sys utt is not classified but attented as context
                        slot_labels += [-100] * len(
                            sys_text)  # TODO: add to args
                        assert len(words) == len(slot_labels)
                        examples.append(
                            InputExample(guid=guid,
                                         words=words,
                                         slot_labels=slot_labels))
        else:
            for _, dialog in enumerate(data[params.test_domain]):
                for turn in range(len(dialog["text"])):
                    guid = "%s-%s" % (set_type, i)
                    i += 1
                    # 1. input_text
                    usr_text = ["[usr]"] + self._tokenize(
                        dialog["text"][turn].split(" | ")[0])
                    sys_text = ["[sys]"] + self._tokenize(
                        dialog["text"][turn].split(" | ")[1])

                    # 2. slot
                    slot_labels = [0] * len(usr_text)
                    for slot in dialog["slot_span"][turn]:
                        cnt = 0
                        for t in range(slot[0], slot[1]):
                            slot_labels[
                                t +
                                1] = 1 if cnt == 0 else 2  # manually add the offset of "[usr]" token
                            cnt += 1
                    words = usr_text + sys_text
                    # sys utt is not classified but attented as context
                    slot_labels += [-100] * len(sys_text)  # TODO: add to args

                    assert len(words) == len(slot_labels)
                    examples.append(
                        InputExample(guid=guid,
                                     words=words,
                                     slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task)
        logger.info(f"LOOKING AT {data_path}")
        if self.args.task == "MultiWOZ_2.1":
            return self._create_examples(data=self._read_file(
                os.path.join(data_path, self.data_file)),
                                         set_type=mode)
        elif self.args.task == "ATIS" or self.args.task == 'SNIPS':
            if mode == "train" or mode == "dev":
                with open(os.path.join(data_path, mode, self.data_file),
                          "r") as f:
                    inputs = f.read().splitlines()
                with open(os.path.join(data_path, mode, self.label_file),
                          "r") as f:
                    labels = f.read().splitlines()
                examples = []
                i = 0
                for turn in range(len(inputs)):
                    guid = "%s-%s" % (mode, i)
                    i += 1
                    # 1. input_text
                    words = inputs[turn].split()
                    # 2. slot
                    slot_labels = labels[turn].split()
                    slot_dict = {"O": 0, "B": 1, "I": 2}
                    slot_labels = [
                        slot_dict[s.split("-")[0]] for s in slot_labels
                    ]
                    assert len(words) == len(slot_labels)
                    examples.append(
                        InputExample(guid=guid,
                                     words=words,
                                     slot_labels=slot_labels))
            else:
                return self._create_examples(data=self._read_file(
                    os.path.join(self.args.data_dir, "MultiWOZ_2.1",
                                 'data_single.json')),
                                             set_type=mode)

            return examples


def convert_examples_to_features(examples,
                                 max_seq_len,
                                 tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")
        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            slot_labels_ids.extend([int(slot_label)] * len(word_tokens))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len -
                                                special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] *
                                           padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] *
                                             padding_length)

        assert len(input_ids
                   ) == max_seq_len, "Error with input length {} vs {}".format(
                       len(input_ids), max_seq_len)
        assert len(
            attention_mask
        ) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(
            token_type_ids
        ) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len)
        assert len(
            slot_labels_ids
        ) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info(f"guid: {example.guid}")
            logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.info(
                f"attention_mask: {' '.join([str(x) for x in attention_mask])}"
            )
            logger.info(
                f"token_type_ids: {' '.join([str(x) for x in token_type_ids])}"
            )
            logger.info(
                f"slot_labels: {' '.join([str(x) for x in slot_labels_ids])}")

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          slot_labels_ids=slot_labels_ids))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = Processor(args)

    if args.task == "MultiWOZ_2.1":
        # MultiWOZ uses the other domains for training
        cached_features_file = os.path.join(
            args.data_dir, args.task, 'cached_{}_{}_{}'.format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                params.test_domain))
    else:
        if mode == "train" or mode == "dev":
            cached_features_file = os.path.join(
                args.data_dir, args.task, 'cached_{}_{}'.format(
                    mode,
                    list(filter(None,
                                args.model_name_or_path.split("/"))).pop()))
        else:
            # Test ATIS and SNIPS on MultiWOZ
            cached_features_file = os.path.join(
                args.data_dir, "MultiWOZ_2.1", 'cached_{}_{}_{}'.format(
                    mode,
                    list(filter(None,
                                args.model_name_or_path.split("/"))).pop(),
                    params.test_domain))

    if os.path.exists(cached_features_file):
        # if False:
        logger.info(
            f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples,
            args.max_seq_len,
            tokenizer,
            pad_token_label_id=pad_token_label_id)
        logger.info(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                      dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                      dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features],
                                       dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_slot_labels_ids)
    return dataset
