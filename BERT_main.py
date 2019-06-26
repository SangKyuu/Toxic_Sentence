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

    LABEL = data.Field(sequential=False,
                       use_vocab=False,
                       preprocessing=lambda x: utils.Round_up(x),
                       batch_first=True)

    train_data = TabularDataset(path='./train_leftover.csv', format='csv', fields=[('index', INDEX), ('target', LABEL),('comment_text',TEXT)],skip_header=True)
    val_data = TabularDataset(path='./validation.csv', format='csv', fields=[('index', INDEX), ('target', LABEL),('comment_text',TEXT)],skip_header=True)

    data_len = len(train_data)+len(val_data)
    tr_txt = []
    tr_label = []
    val_txt = []
    val_label = []

    for i, data_t in enumerate(train_data):
        tr_txt.append(data_t.comment_text)
        tr_label.append(data_t.target)

    for j, data_v in enumerate(val_data):
        val_txt.append(data_v.comment_text)
        val_label.append(data_v.target)


    utils.printlog("cheak split: tr data {}\tval data {}\t total : {}\t".format(len(tr_label), len(val_label), data_len))

    tokenizer = BertTokenizer.from_pretrained(opt.bert_model)
    ConvertText2Token = utils.ConvertText2Token(seq_length=opt.seq_len, tokenizer=tokenizer)

    train_dataset = Toxicdataset(
            textlist=tr_txt,
            labellist=tr_label,
            tokenizer=ConvertText2Token
        )

    dl_tr = DataLoader(
        train_dataset,
        batch_size=opt.train_batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=opt.workers,
        drop_last=True,
        pin_memory=True
    )

    val_dataset = Toxicdataset(
        textlist=val_txt,
        labellist=val_label,
        tokenizer=ConvertText2Token
    )

    dl_val = DataLoader(
        val_dataset,
        batch_size=opt.val_batch_size,
        sampler=RandomSampler(val_dataset),
        num_workers=opt.workers,
        drop_last=True,
        pin_memory=True
    )

    num_labels = 2  #class number

    if opt.mode == 'train':

        use_gpu = torch.cuda.is_available()

        cache_dir = opt.cache_dir if opt.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(opt.local_rank))
        model = BertForSequenceClassification.from_pretrained(opt.bert_model,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels)
        model.to(device)

        #model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_train_optimization_steps = int(
            len(tr_label) / opt.train_batch_size / opt.gradient_accumulation_steps) * opt.num_train_epochs

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=opt.learning_rate,
                             warmup=opt.warmup_proportion,
                             t_total=num_train_optimization_steps)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        best_acc = 0.0

        utils.printlog("***** Running training *****")
        utils.printlog("  Num examples = {:d}".format(len(tr_label)))
        utils.printlog("  Batch size = {:d}".format(opt.train_batch_size))
        utils.printlog("  Num steps = {:d}".format(num_train_optimization_steps))

        model.train()

        # E = trange(int(opt.num_train_epochs), position=0, desc="Epoch")
        # t_B = tqdm_notebook(dl_tr, position=1, desc="Train Iteration")
        # v_B = tqdm_notebook(dl_val, position=2, desc="Validation")
        for _ in trange(int(opt.num_train_epochs), position=0, desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(dl_tr, position=1, desc="Train Iteration")):
                batch = tuple(t.to(device) for t in batch)
                tokens_tensor, segments_tensors, label = batch
                logits = model(tokens_tensor, segments_tensors, labels=None)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label.view(-1))
                loss.backward()

                tr_loss += loss.item()

                nb_tr_examples += tokens_tensor.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if step % 500 == 0:
                    with torch.no_grad():

                        eval_loss = 0
                        nb_eval_steps = 0
                        preds = []
                        val_labels = []

                        for val_step, val_batch in enumerate(tqdm(dl_val, position=3, desc="Validation")):
                            val_batch = tuple(t.to(device) for t in val_batch)
                            v_tokens_tensor, v_segments_tensors, v_label = val_batch
                            val_labels.extend(v_label)
                            logits = model(tokens_tensor, segments_tensors, labels=None)
                            loss_fct = CrossEntropyLoss()
                            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), v_label.view(-1))
                            eval_loss += tmp_eval_loss.mean().item()
                            nb_eval_steps += 1
                            if len(preds) == 0:
                                preds.append(logits.detach().cpu().numpy())
                            else:
                                preds[0] = np.append(
                                    preds[0], logits.detach().cpu().numpy(), axis=0)

                        eval_loss = eval_loss / nb_eval_steps
                        preds = preds[0]
                        preds = np.argmax(preds, axis=1)
                        val_labels = np.array(val_labels)
                        result = utils.compute_metrics(preds, val_labels)
                        loss = tr_loss / global_step if opt.mode == 'train' else None

                        result['eval_loss'] = eval_loss
                        result['global_step'] = global_step
                        result['loss'] = loss
                        if result['acc'] > best_acc:
                            utils.printlog("----BEST ACC FOUND------")
                            utils.printlog("BEST_ACC_FOUND_AT STEP{:d} ={:s}".format(global_step, str(result['acc'])))
                            best_acc = result['acc']
                            torch.save(model, opt.save_path+'{:d}_acc{:f}'.format(global_step,best_acc))

                        utils.printlog("***** Eval results *****")
                        for key in sorted(result.keys()):
                            utils.printlog("  {:s} = {:s}".format(key, str(result[key])))

        utils.printlog("----GLOBAL BEST ACC = {:d}------".format(best_acc))










