import csv

import torch



class Toxicdataset(torch.utils.data.Dataset):
    def __init__(self, textlist, tokenizer, labellist=None):
        self.textlist = textlist
        self.labellist = labellist
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.textlist)

    def __getitem__(self, index):
        indexed_tokens, segments_ids, _ = self.tokenizer.bert_preprocess(self.textlist[index])
        if self.labellist:
            y = self.labellist[index]
            tokens_tensor = torch.tensor(indexed_tokens)
            segments_tensors = torch.tensor(segments_ids)
            return (tokens_tensor, segments_tensors, y)
        else:
            tokens_tensor = torch.tensor(indexed_tokens)
            segments_tensors = torch.tensor(segments_ids)
            return (tokens_tensor, segments_tensors)

