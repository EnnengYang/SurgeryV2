import os
from itertools import repeat

import torch
import math
import random
import numpy as np
import time
import sys
import tqdm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Callable, Optional, Sequence, Tuple

from heads import get_classification_head as get_finetuned_classification_head
from merging_cofficient import get_merging_cofficients

# Utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names
def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def l1losses(predicts, labels):
    loss_func = torch.nn.L1Loss()
    losses = 0.
    for y1, y2 in zip(predicts, labels):
        losses += loss_func(y1, y2)
    return losses

def l2losses(predicts, labels):
    loss_func = torch.nn.MSELoss()
    losses = 0.
    for y1, y2 in zip(predicts, labels):
        losses += loss_func(y1, y2)
    return losses

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

class ModelWrapper_Finetuned(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper_Finetuned, self).__init__()
        self.model = model

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, img):
        '''
         # def forward(self, images):
         #     features = self.model(images)
         #     return features

        Implement the ViT's forward function
        ref: https://github.com/lucidrains/vit-pytorch
        '''
        # print('forward imp')

        def _expand_token(token, batch_size: int):
            return token.view(1, 1, -1).expand(batch_size, -1, -1)

        def _global_pool(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            pooled, tokens = x[:, 0], x[:, 1:]
            return pooled, tokens

        x = self.model.model.visual.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.model.model.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.model.visual.positional_embedding.to(x.dtype)

        x = self.model.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.model.model.visual.transformer(x)

        blocks_cls = []
        layer_i = 0
        for r in self.model.model.visual.transformer.resblocks:
            x = r(x, attn_mask=None)
            x = x.permute(1, 0, 2)  #  LND-> NLD
            blocks_cls.append(x.mean(1).detach()) # No gradients needed!
            x = x.permute(1, 0, 2)  # NLD -> LND
            layer_i += 1

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.model.visual.ln_post(x)
        pooled, tokens = _global_pool(x)

        if self.model.model.visual.proj is not None:
            pooled = pooled @ self.model.model.visual.proj

        blocks_cls.append(pooled.detach())
        return pooled, blocks_cls

class ModelWrapper_Surgery_V2(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper_Surgery_V2, self).__init__()
        self.model = model

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, img, down_projs=None, up_projs=None, non_linear_func=torch.nn.ReLU()):
        '''
         # def forward(self, images):
         #     features = self.model(images)
         #     return features

        Implement the ViT's forward function
        ref: https://github.com/lucidrains/vit-pytorch
        '''
        # print('forward imp')

        def _expand_token(token, batch_size: int):
            return token.view(1, 1, -1).expand(batch_size, -1, -1)

        def _global_pool(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            pooled, tokens = x[:, 0], x[:, 1:]
            return pooled, tokens

        x = self.model.model.visual.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.model.model.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.model.visual.positional_embedding.to(x.dtype)

        x = self.model.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.model.model.visual.transformer(x)

        blocks_cls = []
        layer_i = 0
        for r in self.model.model.visual.transformer.resblocks:
            x = r(x, attn_mask=None)

            # deep representation surgery
            x = x.permute(1, 0, 2)  #  LND-> NLD
            feature_sub = down_projs[layer_i](x)
            feature_sub = non_linear_func(feature_sub)
            feature_sub = up_projs[layer_i](feature_sub)
            x = x - feature_sub
            blocks_cls.append(x.mean(1))
            x = x.permute(1, 0, 2)  # NLD -> LND

            layer_i += 1

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.model.visual.ln_post(x)
        pooled, tokens = _global_pool(x)

        if self.model.model.visual.proj is not None:
            pooled = pooled @ self.model.model.visual.proj

        # final representation surgery
        feature_sub = down_projs[-1](pooled)
        feature_sub = non_linear_func(feature_sub)
        feature_sub = up_projs[-1](feature_sub)
        pooled = pooled - feature_sub

        blocks_cls.append(pooled)
        return pooled, blocks_cls

class AlphaWrapper_Surgery_V2(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super(AlphaWrapper_Surgery_V2, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args

        # Surgery Config
        self.block_num = 12
        self.rank = args.rank
        self.hidden_feature_dim = 768
        self.final_feature_dim = 768 if args.model_name == 'ViT-L-14' else 512 # ViT-B/32 | ViT-B/16 = 512  ;  ViT-L-14 = 768

        ralpha = get_merging_cofficients(args.method_name, args.model_name, len(self.exam_datasets))
        self.alpha = torch.Tensor(ralpha)

        self.non_linear_func = torch.nn.ReLU()

        for dataset_name in exam_datasets:
            for layer_i in range(self.block_num):
                print('rank: ' + str(self.rank))
                # deep representation mapping
                down_proj = torch.nn.Linear(self.hidden_feature_dim, self.rank, bias=False)
                up_proj = torch.nn.Linear(self.rank, self.hidden_feature_dim, bias=False)
                torch.nn.init.kaiming_uniform_(down_proj.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(up_proj.weight)
                self.add_module('feature_mapping_down_proj_{}_{}'.format(dataset_name, layer_i), down_proj.to(args.device))
                self.add_module('feature_mapping_up_proj_{}_{}'.format(dataset_name, layer_i), up_proj.to(args.device))

            # final representation mapping
            down_proj = torch.nn.Linear(self.final_feature_dim, self.rank, bias=False)
            up_proj = torch.nn.Linear(self.rank, self.final_feature_dim, bias=False)
            torch.nn.init.kaiming_uniform_(down_proj.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(up_proj.weight)
            self.add_module('feature_mapping_down_proj_{}_{}'.format(dataset_name, 'final'), down_proj.to(args.device))
            self.add_module('feature_mapping_up_proj_{}_{}'.format(dataset_name, 'final'), up_proj.to(args.device))

            # classifier
            classification_head = get_finetuned_classification_head(args, dataset_name)
            classification_head.weight.requires_grad_(False)
            classification_head.bias.requires_grad_(False)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))

    def freeze_head(self, exam_datasets):
        for dataset_name in exam_datasets:
            layer_name = 'classifier_{}'.format(dataset_name)
            classification_head = getattr(self, layer_name)
            classification_head.weight.requires_grad_(False)
            classification_head.bias.requires_grad_(False)

    def collect_trainable_params(self):
        trainable_params = []

        # surgery parameter
        for dataset_name in self.exam_datasets:
            for layer_i in range(self.block_num):
                down_proj = getattr(self, 'feature_mapping_down_proj_{}_{}'.format(dataset_name, layer_i))
                up_proj = getattr(self, 'feature_mapping_up_proj_{}_{}'.format(dataset_name, layer_i))
                trainable_params.append(down_proj.weight)
                trainable_params.append(up_proj.weight)

            down_proj = getattr(self, 'feature_mapping_down_proj_{}_{}'.format(dataset_name, 'final'))
            up_proj = getattr(self, 'feature_mapping_up_proj_{}_{}'.format(dataset_name, 'final'))
            trainable_params.append(down_proj.weight)
            trainable_params.append(up_proj.weight)
        return trainable_params

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_feature_mapping(self, dataset_name):
        down_projs, up_projs = [], []
        for layer_i in range(self.block_num):
            down_proj = getattr(self, 'feature_mapping_down_proj_{}_{}'.format(dataset_name, layer_i))
            up_proj = getattr(self, 'feature_mapping_up_proj_{}_{}'.format(dataset_name, layer_i))
            down_projs.append(down_proj)
            up_projs.append(up_proj)
        down_proj = getattr(self, 'feature_mapping_down_proj_{}_{}'.format(dataset_name, 'final'))
        up_proj = getattr(self, 'feature_mapping_up_proj_{}_{}'.format(dataset_name, 'final'))
        down_projs.append(down_proj)
        up_projs.append(up_proj)
        return down_projs, up_projs

    def get_image_encoder(self):
        if self.alpha.size()[0] == 1:# task-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else: # layer-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        # raw feature
        if self.alpha.size()[0] == 1: # task-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else: # layer-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)

        down_projs, up_projs = self.get_feature_mapping(dataset_name)
        final_feature, hidden_features = self.model(inp, down_projs, up_projs)

        # classifier
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(final_feature)

        return out, hidden_features

class AlphaWrapper_Surgery(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super(AlphaWrapper_Surgery, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args

        ralpha = get_merging_cofficients(args.method_name, args.model_name, len(self.exam_datasets))
        self.alpha = torch.Tensor(ralpha)

        self.non_linear_func = torch.nn.ReLU()

        for dataset_name in exam_datasets:
            # mapping
            print('rank:' + str(args.rank))
            down_proj = torch.nn.Linear(512, args.rank, bias=False)
            up_proj = torch.nn.Linear(args.rank, 512, bias=False)
            torch.nn.init.kaiming_uniform_(down_proj.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(up_proj.weight)
            self.add_module('feature_mapping_to_head_down_proj_{}'.format(dataset_name), down_proj.to(args.device))
            self.add_module('feature_mapping_to_head_up_proj_{}'.format(dataset_name), up_proj.to(args.device))

            # classifier
            classification_head = get_finetuned_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))

    def collect_trainable_params(self):
        trainable_params = []

        # surgery parameter
        for dataset_name in self.exam_datasets:
            down_proj = getattr(self, 'feature_mapping_to_head_down_proj_{}'.format(dataset_name))
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
            trainable_params.append(down_proj.weight)
            trainable_params.append(up_proj.weight)
        return trainable_params

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_feature_mapping_to_head(self, dataset_name):
        down_proj = getattr(self, 'feature_mapping_to_head_down_proj_{}'.format(dataset_name))
        up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
        return down_proj, up_proj

    def get_image_encoder(self):
        if self.alpha.size()[0] == 1:# task-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else: # layer-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        # raw feature
        if self.alpha.size()[0] == 1: # task-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else: # layer-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)
        feature0 = feature

        # feature bias
        down_proj = getattr(self, 'feature_mapping_to_head_down_proj_{}'.format(dataset_name))
        up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
        feature_sub = down_proj(feature)
        feature_sub = self.non_linear_func(feature_sub)
        feature_sub = up_proj(feature_sub)

        # surgery feature
        feature = feature0 - feature_sub

        # classifier
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)

        return out, feature, feature0, feature_sub

