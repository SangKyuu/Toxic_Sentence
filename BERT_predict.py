from __future__ import absolute_import, division, print_function

import torch
from torchtext.data import TabularDataset
import torchtext.data as data
from torch.utils.data import RandomSampler, DataLoader
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam

from dataloader import Toxicdataset
from config import opt
import utils

import os, sys
from tqdm import trange, tqdm
import numpy as np
import csv

if __name__ == '__main__':

    # device check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not device.type == 'cpu':
        torch.cuda.set_device(opt.gpu_id)
    print('runing device : {}'.format(device))


    # word embedding layer
    TEXT = data.Field(sequential=False,
                      use_vocab=True,
                      lower=True,
                      batch_first=True,
                      fix_length=20)

    INDEX = data.Field(sequential=False,
                       use_vocab=False,
                       preprocessing=lambda x: int(x),
                       batch_first=True)


    test_data = TabularDataset(path='./test.csv', format='csv', fields=[('index', INDEX), ('comment_text',TEXT)],skip_header=True)

    data_len = len(test_data)
    test_txt = []

    for i, data_t in enumerate(test_data):
        test_txt.append(data_t.comment_text)

    tokenizer = BertTokenizer.from_pretrained(opt.bert_model)
    ConvertText2Token = utils.ConvertText2Token(seq_length=opt.seq_len, tokenizer=tokenizer)

    test_dataset = Toxicdataset(
            textlist=test_txt,
            tokenizer=ConvertText2Token
        )

    dl_test = DataLoader(
        test_dataset,
        batch_size=opt.test_batch_size,
        num_workers=opt.workers,
        drop_last=False,
        pin_memory=True
    )

    num_labels = 2  #class number

    if opt.mode == 'test':

        use_gpu = torch.cuda.is_available()

        model = torch.load(opt.save_path+'14001_acc0.920407')

        model.to(device)

        global_step = 0
        nb_test_steps = 0

        utils.printlog("***** Running training *****")
        utils.printlog("  Num examples = {:d}".format(len(test_txt)))
        utils.printlog("  Batch size = {:d}".format(opt.test_batch_size))

        predictions = []
        model.eval()

        nb_test_examples, nb_test_steps = 0, 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dl_test, position=1, desc="Train Iteration")):
                batch = tuple(t.to(device) for t in batch)
                tokens_tensor, segments_tensors = batch
                logits = model(tokens_tensor, segments_tensors, labels=None)

                ss = torch.nn.Softmax(1)
                out = ss(logits)
                out = out[:,1]
                predictions.extend(out)

        with open('./submission.csv', 'w') as f:
            submission_csv = csv.writer(f)
            for i,pred in enumerate(predictions):
                submission_csv.writerow([i+7000000,pred.item()])


