# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    mode = 'test'
    gpu_id = 0
    log_filename = './val_80_60_BERT'
    bert_model = 'bert-base-uncased'
    cache_dir = ''
    save_path = './savemodel/best_acc_vanila_bert_40_80_'

    #training var
    seq_len = 40
    learning_rate = 5e-5
    warmup_proportion = 0.1
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 32
    gradient_accumulation_steps = 1
    num_train_epochs = 3
    workers = 2
    local_rank =-1

opt = DefaultConfig()
