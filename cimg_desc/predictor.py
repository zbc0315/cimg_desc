#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 13:43
# @Author  : zhangbc0315@outlook.com
# @File    : predictor.py
# @Software: PyCharm
import importlib_resources as resources
import torch

from cimg_desc.handler import Handler
from cimg_desc.model import Net


class Predictor:

    def __init__(self, device):
        self.model = Net()
        self.handler = Handler()
        self.model.eval()
        self.model.load_state_dict(torch.load(resources.files('cimg_desc').joinpath('retro_rcts_199.pt'),
                                              map_location=torch.device(device))['model'])

    def predict(self, smiles: str):
        data = self.handler.initialization(smiles)
        res, _ = self.model(data)
        return res[0]

    def predict_batch(self, smiles_list: [str]):
        data = self.handler.initialization_batch(smiles_list)
        res, _ = self.model(data)
        return res


if __name__ == "__main__":
    print(Predictor('cpu').predict(smiles="c1ccccc1"))
