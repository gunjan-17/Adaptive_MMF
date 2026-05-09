#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained("ai4bharat/indic-bert", dtype="auto", use_safetensors=True)

    def forward(self, txt, mask, segment=None):
        outputs = self.bert(
            input_ids=txt,
            attention_mask=mask
        )
        out = outputs.last_hidden_state[:, 0, :]
        return out


class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)
        init.xavier_uniform_(self.clf.weight)

    def forward(self, txt, mask, segment=None):
        x = self.enc(txt, mask, segment)
        feat_var = torch.var(x, dim=1, keepdim=True)
        feat_mean = torch.mean(torch.abs(x), dim=1, keepdim=True)
        
        quality = torch.sigmoid(feat_mean / (feat_var + 1e-6))

        x_enhanced = x * quality

        logits = self.clf(x_enhanced)

        pre_fusion_uncertainty = 1.0 - quality.squeeze(1) 

        return logits
