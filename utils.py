from config import opt
from decimal import Decimal, localcontext, ROUND_HALF_UP
import numpy as np


def printlog(text):
    with open('{}.txt'.format(opt.log_filename), 'a') as f:
        f.write(text)
        f.write('\n')
    print(text)


def simple_accuracy(preds, labels):
    return (np.array(preds == labels)).mean()


def compute_metrics(preds, labels):
    return {"acc": simple_accuracy(preds, labels)}


def Round_up(x):
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        x = Decimal(x).to_integral_value()
    return int(x)


class ConvertText2Token():
    def __init__(self, seq_length, tokenizer):
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def bert_preprocess(self, text):
        tokens_a = self.tokenizer.tokenize(text)

        if len(tokens_a) > self.seq_length - 2:
            tokens_a = tokens_a[0:(self.seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)


        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(input_type_ids) == self.seq_length

        return (input_ids, input_mask, input_type_ids)

