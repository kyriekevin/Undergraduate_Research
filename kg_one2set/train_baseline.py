# -*- coding: utf-8 -*-
# @Author: zyz
# @Time: 2021/8/12 9:15
# @File: train_baseline.py
# @Software: PyCharm
import argparse
import logging
import os
import time

import torch
from torch.optim import Adam

import config
import train_ml_baseline
from pykp.model_baseline import Seq2SeqModel, Model
from utils.data_loader import load_data_and_vocab
from utils.functions import common_process_opt
from utils.functions import time_since


def process_opt(opt):
    opt = common_process_opt(opt)

    if opt.set_loss and (not opt.fix_kp_num_len or not opt.one2many):
        raise ValueError("Set fix_kp_num_len and one2many when using set loss!")

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, 'initial.config'), 'rb')
        )
    else:
        torch.save(opt, open(os.path.join(opt.model_path, 'initial.config'), 'wb'))

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    return opt


def init_optimizer(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                     lr=opt.learning_rate, betas=(0.9, 0.998), eps=1e-09)
    return optimizer


def init_bertmodel_optimizer(model):
    unfreeze_layers = ['linear']
    # parameters = []
    # lr, classifier_lr, multiplier = 2e-5, 1e-4, 0.95
    # for layer in range(12, -1, -1):
    #     layer_params = {
    #         'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
    #         'lr': lr
    #     }
    #     parameters.append(layer_params)
    #     lr *= multiplier
    # classifier_params = {
    #     'params': [p for n,p in model.named_parameters() if 'layer_norm' in n or 'linear' in n
    #                or 'pooling' in n],
    #     'lr': classifier_lr
    # }
    # parameters.append(classifier_params)

    for name, param in model.named_parameters():
        param.requires_grad = False
        for layer in unfreeze_layers:
            if layer in name:
                param.requires_grad = True
                break

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    return optimizer


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    model = Seq2SeqModel(opt)
    model1 = Model(opt)

    logging.info(model)
    logging.info(model1)

    total_params = sum([param.nelement() for param in model.parameters() and model1.parameters()])
    logging.info('model parameters: %d, %.2fM' % (total_params, total_params * 4 / 1024 / 1024))

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        model.load_state_dict(torch.load(opt.train_from))
    return model.to(opt.device), model1.to(opt.device)


def main(opt):
    start_time = time.time()
    train_data_loader, valid_data_loader, vocab = load_data_and_vocab(opt, load_train=True)
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)

    start_time = time.time()

    model, model1 = init_model(opt)
    optimizer = init_optimizer(model, opt)
    optimizer1 = init_bertmodel_optimizer(model1)

    train_ml_baseline.train_model(model, model1, optimizer, optimizer1, train_data_loader, valid_data_loader, opt)
    training_time = time_since(start_time)
    logging.info('Time for training: %.1f' % training_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
