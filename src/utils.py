from abc import ABC, abstractmethod
# from custom_abcmeta import ABCMeta, abstract_attribute
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
from pickle import dump
import os
from collections import defaultdict
from _plot_cfg import *
logger = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    """Abstract base class for optimizer implementation."""
    def __init__(self, *args, **kwargs):
        self.global_weights = None

    @abstractmethod
    def aggregate(self, *args, **kwargs) -> None:
        """Abstract method to conduct optimization."""

    def fair_loss(self, *args, **kwargs) -> torch.Tensor:
        """Abstract method to calculate loss."""
        return torch.tensor(0)


class Role(ABC):
    """Abstract base class for role implementation."""

    ###########################################################################
    # The following functions need to be implemented the grandchild class.
    ###########################################################################
    @abstractmethod
    def __init__(self):
        self.device = None
        self.fair = None
        self.model = None

    @abstractmethod
    def train(self) -> None:
        """Abstract method to train a model."""

    @abstractmethod
    def evaluate(self) -> None:
        """Abstract method to evaluate a model."""

    def _cal_bias_hard(self, dataloader):
        res = torch.zeros(4)
        for data, target, group in dataloader:
            data, target, group = data.to(self.device), target.to(self.device), group.to(self.device)
            output = torch.exp(self.model(data))
            res += torch.Tensor(self._cal_bias_batch_hard(output, target, group))

        return res / len(dataloader.dataset)

    def _cal_bias_batch_hard(self, output, target, group):
        if self.fair == 'SP':
            a = torch.sum((output[:, 1] > 0.5) * (group == 0))
            c = torch.sum((output[:, 1] > 0.5) * (group == 1))
            b = torch.sum(group == 0)
            d = torch.sum(group == 1)
        elif self.fair == 'EOP':
            a = torch.sum((output[:, 1] > 0.5) * ((group == 0) & (target == 1)))
            c = torch.sum((output[:, 1] > 0.5) * ((group == 1) & (target == 1)))
            b = torch.sum((group == 0) & (target == 1))
            d = torch.sum((group == 1) & (target == 1))
        elif self.fair == 'CAL':
            a = torch.sum((output[:, 1] > 0.5) * ((group == 0) & (target == 1)))
            c = torch.sum((output[:, 1] > 0.5) * ((group == 1) & (target == 1)))
            b = torch.sum((output[:, 1] > 0.5) * (group == 0))
            d = torch.sum((output[:, 1] > 0.5) * (group == 1))
        else:
            raise ValueError('Fairness type not supported.')

        return a, b, c, d

    def _cal_bias_batch(self, output, target, group):
        if self.fair == 'SP':
            a = torch.sum(output[:, 1] * (group == 0))
            c = torch.sum(output[:, 1] * (group == 1))
            b = torch.sum(group == 0)
            d = torch.sum(group == 1)
        elif self.fair == 'EOP':
            a = torch.sum(output[:, 1] * ((group == 0) & (target == 1)))
            c = torch.sum(output[:, 1] * ((group == 1) & (target == 1)))
            b = torch.sum((group == 0) & (target == 1))
            d = torch.sum((group == 1) & (target == 1))
        elif self.fair == 'CAL':
            a = torch.sum(output[:, 1] * ((group == 0) & (target == 1)))
            c = torch.sum(output[:, 1] * ((group == 1) & (target == 1)))
            b = torch.sum(output[:, 1] * (group == 0))
            d = torch.sum(output[:, 1] * (group == 1))
        else:
            raise ValueError('Fairness type not supported.')

        return a, b, c, d

    def _cal_bias(self, dataloader):
        res = torch.zeros(4)
        for data, target, group in dataloader:
            data, target, group = data.to(self.device), target.to(self.device), group.to(self.device)
            output = torch.exp(self.model(data))
            res += torch.Tensor(self._cal_bias_batch(output, target, group))

        return res / len(dataloader.dataset)


class Server(Role):
    def __init__(self, model, device='cpu', test_loader=None, fair=None, algorithm=None) -> None:
        """Initialize a class instance."""
        self.test_loader = test_loader
        self.device = device
        self.model = model.to(self.device)
        self.metrics = defaultdict(list)
        self.bias = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'val': 0., 'sign': 0}
        self.num_sample = 0
        self.fair = fair
        self.agg_weights = None
        self.algorithm = algorithm

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target, group in self.test_loader:
                data, target, group = data.to(self.device), target.to(self.device), group.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target,
                    reduction='sum').item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        total = len(self.test_loader.dataset)
        test_loss /= total
        test_accuracy = correct / total

        # update metrics after each evaluation so that the metrics can be
        # logged in a model registry.
        self.update_metrics({
            'test-loss': test_loss,
            'test-accuracy': test_accuracy
        })

    def aggregate_weights(self, clients) -> None:
        global_weights = self.algorithm.aggregate(clients, self)
        # update model with global weights
        self.model.load_state_dict(global_weights)

        # update sign
        # self.update_bias(clients)

        self.evaluate()

    def update_bias(self, clients):
        if self.fair == 'SP' or self.fair == 'EOP':
            self.bias['val'] = sum([client.bias['val'] * client.dataset_size / self.num_sample for client in clients])
        # elif self.fair == 'CAL':
        #     for letter in 'abcd':
        #         self.bias[letter] = sum([client.bias[letter] * client.dataset_size / self.num_sample for client in clients])
        #     self.bias['val'] = self.bias['a']/self.bias['b'] - self.bias['c']/self.bias['d']
        # else:
        #     raise ValueError('Fairness type not supported.')
        # with torch.no_grad():
        #     a, b, c, d = self._cal_bias(self.test_loader)
        #     self.bias['a'] = a.item()
        #     self.bias['b'] = b.item()
        #     self.bias['c'] = c.item()
        #     self.bias['d'] = d.item()
        #     self.bias['val'] = (a/b-c/d).item()
        # if (np.sign(self.bias['val']) != self.bias['sign']) & (self.bias['sign'] != 0) & (self.algorithm.gamma != 'auto'):
        #     self.algorithm.gamma *= 0.95
        #     logger.debug(f'Gamma decrease to {self.algorithm.gamma}.')

        self.bias['sign'] = np.sign(self.bias['val'])
        self.update_metrics({'bias': self.bias['val']})

    def update_metrics(self, metrics):
        """Update metrics."""
        # self.metrics = self.metrics | metrics
        for k, v in metrics.items():
            self.metrics[k].append(v)

    def get_weights(self):
        return self.model.state_dict()

    def save_model(self, path='../saved_models/'):
        """Save model in a model registry."""
        if not os.path.exists(path):
            os.makedirs(path)
        if self.model:
            # model_name = str(datetime.now()).split('.')[0]
            model_name = f'{self.fair}_{self.algorithm.__class__.__name__}'
            with open(path+model_name+'.pkl', 'wb') as outp:
                dump(self.model, outp)
            with open(path+model_name+'_metric.pkl', 'wb') as outp:
                dump(self.metrics, outp)

    def plot_metrics(self, path='../plots/'):
        """Plot metrics."""
        if not os.path.exists(path):
            os.makedirs(path)
        for i, (k, v) in enumerate(self.metrics.items()):
            plt.plot(v, label=k, marker=markers[i], ls=line_types[i])
        plt.xlabel('Iteration')
        # plt.title(f'{self.fair}, {self.algorithm.__class__.__name__}')
        plt.legend(self.metrics.keys())
        # plt.savefig(path+str(datetime.now()).split('.')[0]+'.png')
        plt.tight_layout()
        plt.savefig(path+f'{self.fair}_{self.algorithm.__class__.__name__}'+'.png')
        plt.close()

    def train(self) -> None:
        pass


class Trainer(Role):
    def __init__(self, model, dataset, fair=None, idx=None, split_ratio=1, algorithm=None, lr=0.001, reweigh=False,
                 device='cpu', epochs=1, batch_size=32) -> None:
        """Initialize a class instance."""
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = batch_size
        idx = np.array(range(len(dataset))) if idx is None else idx
        np.random.shuffle(idx)
        idx_train = idx[:int(split_ratio * len(idx))]
        idx_test = idx[int(split_ratio * len(idx)):]
        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, idx_train),
                                                        batch_size=self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, idx_test),
                                                       batch_size=self.batch_size) if split_ratio < 1 else None
        self.dataset_size = len(self.train_loader.dataset)
        self.epochs = epochs
        # self.optimizer = optim.Adadelta(self.model.parameters())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1, 5], gamma=0.1)
        self.scheduler = None
        self.metrics = defaultdict(list)
        self.algorithm = algorithm
        self.reweigh = reweigh
        self.weights = cal_weights([self.train_loader], device=self.device)
        if fair:
            self.fair = fair
            self.bias = {'a': 0., 'b': 0., 'c': 0., 'd': 0.,
                         'local_val': 0.,
                         'val': 0., 'sign': 0.,
                         'global_a': 1, 'global_c': 1, 'global_b': 1, 'global_d': 1}
            self.update_bias()

    def update_bias(self):
        self.model.train(False)
        with torch.no_grad():
            a, b, c, d = self._cal_bias(self.train_loader)
            self.bias['a'] = a.item()
            self.bias['b'] = max(b.item(), 1e-8)
            self.bias['c'] = c.item()
            self.bias['d'] = max(d.item(), 1e-8)
            self.bias['local_val'] = self.bias['a'] / self.bias['b'] - self.bias['c'] / self.bias['d']
            self.bias['val'] = self.bias['a']/self.bias['global_b']-self.bias['c']/self.bias['global_d']
        # logger.debug(f'Fair: {self.bias}')
        self.update_metrics({'local_bias': self.bias['local_val']})

    def evaluate(self) -> None:
        pass

    def update_metrics(self, metrics):
        """Update metrics."""
        # self.metrics = self.metrics | metrics
        for k, v in metrics.items():
            self.metrics[k].append(v)

    def train(self) -> None:
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            if self.scheduler:
                self.scheduler.step()
        # if self.fair:
        #     self.update_bias()

    def _train_epoch(self, epoch):
        for batch_idx, (data, target, group) in enumerate(self.train_loader):
            data, target, group = data.to(self.device), target.to(self.device), group.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            weight = self.weights.to(self.device)[target, group] if self.reweigh else None
            acc_loss = weighted_nll_loss(output, target, weight) if weight is not None else F.nll_loss(output, target)
            f_loss = self.algorithm.fair_loss(self, output, target, group)
            loss = acc_loss + f_loss
            loss.backward()
            self.optimizer.step()
        # logger.debug(f"epoch: {epoch} acc loss: {acc_loss.item():.6f}")

    def get_weights(self):
        return self.model.state_dict()

    def update_model(self, server: Server):
        self.model.load_state_dict(server.get_weights())

    def sync(self, server):
        for letter in 'abcd':
            self.bias['global_'+letter] = server.bias[letter]
        self.bias['sign'] = server.bias['sign']
        self.bias['val'] = server.bias['val']


class SimData(torch.utils.data.Dataset):
    """Torch dataset for data loader."""
    def __init__(self, X, y, group=None):
        self.dataset = X.to_numpy() if type(X) == pd.DataFrame else X
        self.targets = y
        self.group = torch.zeros(len(y)) if group is None else group

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, :], self.targets[idx], self.group[idx]


def weighted_nll_loss(output, target, weight):
    """Weighted negative log likelihood loss."""
    return (-weight@output.gather(1, target.reshape(-1, 1)))[0]/len(weight)


def cal_weights(dataloaders, device='cpu'):
    """Calculate weights for each class."""
    weights = torch.zeros((2, 2), requires_grad=False)
    for dataloader in dataloaders:
        for data, target, group in dataloader:
            target, group = target.to(device), group.to(device)
            for i in range(len(target)):
                weights[target[i], group[i]] += 1
        y = weights.sum(dim=1)
        a = weights.sum(dim=0)
    n = y.sum()
    weights = torch.outer(y, a) / weights / n

    return weights
