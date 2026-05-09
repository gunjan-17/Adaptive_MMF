#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.utils.utils import truncate_seq_pair, numpy_seed

import random

class JsonlDataset(Dataset):
    """Dataset for multimodal JSONL files supporting text, image and label data."""
    
    def __init__(self, data_path, tokenizer, transforms, mode, vocab, args):
        # Load JSONL data
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.n_classes = len(args.labels)
        self.mode = mode

        # Randomly drop images during training based on drop rate
        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        # Adjust max sequence length for multimodal models
        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        """Returns number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Retrieves a single sample and processes its components."""
        
        # Process text for VSNLI task (paired sentences)
        if self.args.task == "vsnli":
            sent1 = self.data[index]["sentence1"]
            sent2 = self.data[index]["sentence2"]

            encoding = self.tokenizer(
                sent1,
                sent2,
                add_special_tokens=True,
                max_length=self.args.max_seq_len,
                truncation=True,
            )

            sentence = torch.LongTensor(encoding["input_ids"])
            segment = torch.zeros(len(sentence))
        # Process text for other tasks
        else:
            text = self.data[index]["text"]

            # ===== Noise Injection  =====
            if self.args.noise_level > 0.0 and self.mode == "test":
                if np.random.choice([0, 1], p=[0.5, 0.5]):
                    wordlist = text.split(' ')
                    for i in range(len(wordlist)):
                        replace_p = 0.1 * self.args.noise_level
                        if np.random.choice([0, 1], p=[1-replace_p, replace_p]):
                            wordlist[i] = '_'
                    text = ' '.join(wordlist)

            # ===== TOKENIZATION  =====
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.args.max_seq_len,
                truncation=True,
            )

            sentence = torch.LongTensor(encoding["input_ids"])
            segment = torch.zeros(len(sentence))  


        # Handle multi-label vs single-label tasks
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[[self.args.labels.index(tgt) for tgt in self.data[index]["label"]]] = 1
        else:
            label = torch.LongTensor([self.args.labels.index(self.data[index]["label"])])

        # Process image for multimodal models
        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt", "latefusion", "tmc"]:
            if self.data[index]["img"]:
                image = Image.open(
                    os.path.join(self.data_dir, self.data[index]["img"])
                ).convert("RGB")
            else:
                # Create blank image if missing
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        # Special processing for MMBT model
        if self.args.model == "mmbt":
            segment = segment[1:]   # Remove first SEP used for images
            sentence = sentence[1:]  # Remove corresponding token
            segment += 1            # Shift segment IDs (image=0, text=1)

        return sentence, segment, image, label, torch.LongTensor([index])

class AddGaussianNoise(object):
    """Add Gaussian noise to PIL images.
    
    Parameters:
        mean: Mean of Gaussian distribution
        variance: Variance of Gaussian distribution
        amplitude: Scaling factor for noise values
    """
    
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        """Apply noise transformation to image."""
        img = np.array(img)
        h, w, c = img.shape
        np.random.seed(0)
        # Generate noise array matching image dimensions
        N = self.amplitude * np.random.normal(
            loc=self.mean, 
            scale=self.variance, 
            size=(h, w, 1)
        )
        # Duplicate noise for all channels
        N = np.repeat(N, c, axis=2)
        img = N + img
        # Clip values to valid pixel range
        img[img > 255] = 255
        return Image.fromarray(img.astype('uint8')).convert('RGB')

class AddSaltPepperNoise(object):
    """Add salt-and-pepper noise to PIL images.
    
    Parameters:
        density: Probability of noise occurrence
        p: Execution probability for each transformation call
    """
    
    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        """Apply salt-and-pepper noise with execution probability."""
        # Apply noise based on probability
        if random.uniform(0, 1) < self.p:  
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd  # Portion of pixels to leave unchanged
            
            # Create noise mask (0=pepper, 1=salt, 2=no change)
            mask = np.random.choice(
                (0, 1, 2), 
                size=(h, w, 1), 
                p=[Nd/2.0, Nd/2.0, Sd]
            )
            # Duplicate mask for all color channels
            mask = np.repeat(mask, c, axis=2)
            
            # Apply salt (255) and pepper (0) noise
            img[mask == 0] = 0    # Pepper
            img[mask == 1] = 255  # Salt
            
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        
        return img
