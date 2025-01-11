import logging
from copy import deepcopy

import os
import math
import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from model import *
import torch.nn as nn
import torch.optim as optim
import shutil
import tensorboardX
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


"""
Three sections: presentation, question, answers
Considering the consistency of all benchmarks, we use the sentence transformer as a substitute for Bi-lstm+attention

Todo List:
-[ ] check whether the dataset has been generated, if not, generate it.
"""


class ContrastiveMultiMLpDataset(Dataset):
    def __init__(self, paths, label_name='firm_std_10_post'):
        super(ContrastiveMultiMLpDataset, self).__init__()
        self.label_name = label_name
        data = self.loading_train_dataset(paths, label_name)
        self.input_pre = data['pre']
        self.input_q = data['q']
        self.input_a = data['a']
        self.input_pre_num = data['pre_num']
        self.input_q_num = data['q_num']
        self.input_a_num = data['a_num']
        self.label = np.log(data['label'])

    '''
    We simplify some details here so that the readers can easily understand what we are doing and 
    apply to their own specific tasks.
    
    In concrete, 
        we drop the collect_fn function, which we design for masking different length's data
    '''
    @staticmethod
    def loading_train_dataset(paths: list, label_name: str) -> dict:
        X_pre = []
        X_q = []
        X_a = []

        X_pre_num = []
        X_q_num = []
        X_a_num = []

        Y = []
        for path in paths:
            with open(path, 'rb') as fIn:
                stored_datas = pickle.load(fIn)
                for stored_data in tqdm(stored_datas):  # call
                    if label_name == 'firm_std_3_post':
                        Y.append(stored_data['label']['firm_std_3_post'])
                    elif label_name == 'firm_std_7_post':
                        Y.append(stored_data['label']['firm_std_7_post'])
                    elif label_name == 'firm_std_10_post':
                        Y.append(stored_data['label']['firm_std_10_post'])
                    elif label_name == 'firm_std_15_post':
                        Y.append(stored_data['label']['firm_std_15_post'])
                    elif label_name == 'firm_std_20_post':
                        Y.append(stored_data['label']['firm_std_20_post'])
                    elif label_name == 'firm_std_60_post':
                        Y.append(stored_data['label']['firm_std_60_post'])
                    else:
                        continue

                    X_pre.append(stored_data['pre_reps'])
                    X_q.append(stored_data['q_reps'])
                    X_a.append(stored_data['a_reps'])
                    X_pre_num.append(stored_data['pre_num'])
                    X_q_num.append(stored_data['a_num'])
                    X_a_num.append(stored_data['q_num'])

        return {
            'pre': X_pre,  # [M, N^p, d]
            'q': X_q,  # [M, N^{qa}, d]
            'a': X_a,  # [M, N^{qa}, d]
            'pre_num': X_pre_num,
            'q_num': X_q_num,
            'a_num': X_a_num,
            'label': Y  # [M]
        }

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {
            'pre': self.input_pre[index],
            'q': self.input_q[index],
            'a': self.input_a[index],
            'pre_num': self.input_pre_num[index],
            'q_num': self.input_q_num[index],
            'a_num': self.input_a_num[index],
            'label': self.label[index]
        }


class ModelWrappedWithMSELoss(nn.Module):
    def __init__(self,
                 trans_temperature,
                 conv_temperature,
                 qa_temperature,
                 lr_risk_pro,
                 lr_trans_pro,
                 lr_conv_pro,
                 lr_qa_pro,
                 device
                 ):
        super(ModelWrappedWithMSELoss, self).__init__()
        self.model = None
        self.criterion_risk = torch.nn.MSELoss(reduction='none')
        self.criterion_trans = InfoNCE(temperature=trans_temperature)
        self.criterion_conv = InfoNCE(temperature=conv_temperature)
        self.criterion_qa = InfoNCE(temperature=qa_temperature)
        self.lr_risk_pro = lr_risk_pro
        self.lr_trans_pro = lr_trans_pro
        self.lr_conv_pro = lr_conv_pro
        self.lr_qa_pro = lr_qa_pro

        self.device = device

    def init_model(self, args):
        self.model = ContrastiveMultiMlp(**vars(args)).to(self.device)

    def forward(self, inputs, target):
        output, a_vec, q_vec, mean_a_vec, mean_q_vec, mean_qa_vec, tilde_pre_vec = self.model(*inputs)
        target = target.view(-1).to(torch.float32)
        output = output.view(target.size(0), -1).to(torch.float32)
        if output.size(1) == 1:
            output = output.view(target.size(0))

        risk_loss = self.criterion_risk(output, target)
        trans_loss = self.criterion_trans(mean_qa_vec, tilde_pre_vec)
        conv_loss = self.criterion_conv(mean_a_vec, mean_q_vec)
        qa_loss = self.criterion_qa(a_vec, q_vec)

        backward_loss = risk_loss * self.lr_risk_pro + \
                        trans_loss * self.lr_trans_pro + \
                        conv_loss * self.lr_conv_pro + \
                        qa_loss * self.lr_qa_pro
        select_loss = torch.mean(risk_loss).view(1, 1)  # to select the best model

        return backward_loss, select_loss


class ProfetContrastiveTrainer(object):
    def __init__(self,
                 args,
                 config,
                 grad_clip=None,
                 patience_epochs=10
                 ):
        logging.info(f"initialize {self.__class__.__name__}")
        self.args = deepcopy(args)
        self.config = deepcopy(config)
        self.dic = torch.randn(10, 384).cuda()
        torch.set_num_threads(3)

        # tensorboard
        tb_path = os.path.join(self.args.result, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        os.makedirs(tb_path)
        self.tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # model
        self.model_with_loss = ModelWrappedWithMSELoss(trans_temperature=self.config.contrastive.trans_temperature,
                                                       conv_temperature=self.config.contrastive.conv_temperature,
                                                       qa_temperature=self.config.contrastive.qa_temperature,
                                                       lr_risk_pro=self.config.contrastive.lr_risk_pro,
                                                       lr_trans_pro=self.config.contrastive.lr_trans_pro,
                                                       lr_conv_pro=self.config.contrastive.lr_conv_pro,
                                                       lr_qa_pro=self.config.contrastive.lr_qa_pro,
                                                       device=self.config.device,
                                                       )
        self.model_with_loss.init_model(self.config.model)
        self.model = self.model_with_loss.model

        logging.info(self.model)

        num_parameters = sum([l.nelement() for l in self.model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

        # optim
        optim_params = vars(self.config.optim)
        if optim_params['optimizer'] == 'Adagrad':
            del optim_params['optimizer']
            optimizer = optim.Adagrad(self.model.parameters(), **optim_params)
        else:
            raise AssertionError('According to the original design, you should use "Adagrad" as the optimizer')
        self.optimizer = optimizer

    @staticmethod
    def __preprocess_data():
        """
        We omit this part, researches can change this part to their special domains
        :return:
        """
        pass

    def train(self):
        train_step = 0
        test_step = 0
        best_score = float('inf')

        # load data
        train_dataset = ContrastiveMultiMLpDataset(
            ['data_2015.pkl',
             'data_2016.pkl'],
            label_name=self.config.data.label)
        eval_dataset = ContrastiveMultiMLpDataset(
            ['data_2017.pkl'],
            label_name=self.config.data.label)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.config.train.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=0,
                                      drop_last=True)

        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=self.config.train.batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers=0,
                                     drop_last=True)

        for epoch in range(self.config.train.n_epochs):
            logging.info(f"Epoch {epoch}")
            '''train'''
            self.model.train()
            total_train_loss = []
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                train_step += 1

                self.optimizer.zero_grad()

                model_input = [data['pre'].to(self.config.device),
                               data['pre_num'],
                               data['q'].to(self.config.device),
                               data['q_num'],
                               data['a'].to(self.config.device),
                               data['a_num']
                               ]
                label = data['label'].to(self.config.device)

                backward_loss, train_select_loss = self.model_with_loss(model_input, label)
                backward_loss.backward()
                self.optimizer.step()
                total_train_loss.append(backward_loss.item())

                self.tb_logger.add_scalar('train_select_loss', train_select_loss, global_step=train_step)

            '''eval'''
            self.model.eval()
            with torch.no_grad():
                total_eval_loss = []
                cur_eval_score = []
                for i, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                    test_step += 1

                    model_input = [data['pre'].to(self.config.device),
                                   data['pre_num'],
                                   data['q'].to(self.config.device),
                                   data['q_num'],
                                   data['a'].to(self.config.device),
                                   data['a_num']
                                   ]
                    label = data['label'].to(self.config.device)

                    backward_loss, eval_select_loss = self.model_with_loss(model_input, label)
                    total_eval_loss.append(eval_select_loss.item())
                    cur_eval_score.append(eval_select_loss.item())
                    self.tb_logger.add_scalar('eval_select_loss', eval_select_loss, global_step=test_step)

            cur_score = np.mean(cur_eval_score)

            if cur_score < best_score:
                logging.info(f"best score is: {best_score}, current score is: {cur_score}, save best_checkpoint.pth")
                best_score = cur_score
                states = [
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                ]
                torch.save(states, os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))

        # save checkpoint
        states = [
            self.model.state_dict(),
            self.optimizer.state_dict(),
        ]
        torch.save(states, os.path.join(self.args.checkpoint, 'checkpoint.pth'))

    def test(self, load_pre_train=True):
        if load_pre_train:
            # load pretrained_model
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))
            self.model.load_state_dict(pretrained_data[0])

        test_dataset = ContrastiveMultiMLpDataset(
            ['data_2018.pkl'],
            label_name=self.config.data.label)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.config.train.batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers=0,
                                     drop_last=True)

        logging.info(f"Testing. Total {test_dataset.__len__()} data.")
        self.model.eval()
        output_list = []
        label_list = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

                model_input = [data['pre'].to(self.config.device),
                               data['pre_num'],
                               data['q'].to(self.config.device),
                               data['q_num'],
                               data['a'].to(self.config.device),
                               data['a_num']
                               ]
                label = data['label'].to(self.config.device)
                output, a_vec, q_vec, mean_a_vec, mean_q_vec, mean_qa_vec, tilde_pre_vec = self.model(*model_input)
                for out in output.cpu().numpy():
                    output_list.append(out)
                for label_in in label.cpu().numpy():
                    label_list.append(label_in)

        mse = mean_squared_error(output.cpu().numpy(), label.cpu().numpy())
        mae = mean_absolute_error(output.cpu().numpy(), label.cpu().numpy())
        tau, _ = stats.kendalltau(label_list, output_list)
        rou, _ = stats.spearmanr(label_list, output_list)

        logging.info(f'label is {self.config.data.label}, '
                     f'mse is {mse}, '
                     f'mae is {mae}, '
                     f'rou is {rou}, '
                     f'tau is {tau}')
