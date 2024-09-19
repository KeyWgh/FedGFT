from utils import AbstractOptimizer
import logging
import torch
logger = logging.getLogger(__name__)


class FedAvg(AbstractOptimizer):
    def aggregate(self, clients, server) -> None:
        # reset global weights before aggregation
        self.global_weights = None

        # receive local model parameters from trainers
        n = server.num_sample
        for client in clients:
            if self.global_weights is None:
                self.global_weights = {k: v * client.dataset_size / n for k, v in client.get_weights().items()}
            else:
                for k, v in client.get_weights().items():
                    self.global_weights[k] += v * client.dataset_size / n

        return self.global_weights


class FairFed(AbstractOptimizer):
    def __init__(self, *args, beta=1, **kwargs):
        self.beta = beta
        super().__init__(*args, **kwargs)

    def aggregate(self, clients, server) -> None:
        # reset global weights before aggregation
        self.global_weights = None
        weight = torch.ones(len(clients)) if server.agg_weights is None else server.agg_weights

        # receive local model parameters from trainers
        n = server.num_sample
        # weight = []
        # for client in clients:
        #     weight.append(client.dataset_size / n *
        #                   torch.exp(-self.beta * torch.abs(torch.tensor(
        #                       server.bias['val'] - client.bias['local_val']))))
        #
        # weight = torch.tensor(weight)

        delta = torch.abs(torch.Tensor([server.bias['val'] - client.bias['local_val'] for client in clients])).squeeze()
        delta = delta - delta.mean()
        weight = torch.maximum(weight - self.beta * delta, torch.zeros_like(weight))
        weight /= weight.sum()
        server.agg_weights = weight
        for i, client in enumerate(clients):
            if self.global_weights is None:
                self.global_weights = {k: v * weight[i] for k, v in client.get_weights().items()}
            else:
                for k, v in client.get_weights().items():
                    self.global_weights[k] += v * weight[i]

        return self.global_weights


class FedGFT(FedAvg):
    def __init__(self, *args, gamma=1, **kwargs):
        self.gamma = gamma
        self.reg = kwargs.get('reg', 'l2')
        super().__init__(*args, **kwargs)

    def fair_loss(self, client, output, target, group):
        gamma = abs(client.bias['val']*2) if self.gamma == 'auto' else self.gamma
        if client.fair:
            a, b, c, d = client._cal_bias_batch(torch.exp(output), target, group)
            if client.fair == 'SP' or client.fair == 'EOP':
                res = (a / client.bias['global_b'] - c / client.bias['global_d']) / len(target)
            elif client.fair == 'CAL':
                # w1 = client.bias['global_a'] / client.bias['global_b'] ** 2
                # w2 = client.bias['global_c'] / client.bias['global_d'] ** 2
                # res = (- b / w1 + d / w2) / len(target)
                # res = (a/b - c/d)
                # res = (a / client.bias['global_b'] - c / client.bias['global_d'] - b / w1 + d / w2) / len(target)
                res = -(a / client.bias['global_b'] - c / client.bias['global_d']) / len(target)
            else:
                raise ValueError('Fairness type not supported.')
        else:
            res = 0

        # return torch.abs(res-client.bias['local_val']+client.bias['val'])*gamma
        # logger.debug(f'Fair loss: {res:.2f}; sign: {client.bias["sign"]}')
        coef = client.bias['sign'] if self.reg == 'id' else client.bias['val']
        return coef * gamma * res
        # return gamma * torch.abs(res)


class FedFairLocal(FedGFT):
    def fair_loss(self, client, output, target, group):
        gamma = abs(client.bias['val']*5) if self.gamma == 'auto' else self.gamma
        if client.fair:
            a, b, c, d = client._cal_bias_batch(torch.exp(output), target, group)
            if client.fair == 'SP' or client.fair == 'EOP':
                res = (a / client.bias['b'] - c / client.bias['d']) / len(target)
            elif client.fair == 'CAL':
                res = (a/b - c/d)
            else:
                raise ValueError('Fairness type not supported.')
        else:
            res = 0

        # return torch.abs(res-client.bias['local_val']+client.bias['val'])*gamma
        # logger.debug(f'Fair loss: {res:.2f}; sign: {client.bias["sign"]}')
        coef = torch.abs(res) if self.reg == 'id' else res**2/2
        return gamma * coef


def get_optimizer(optimizer_name, *args, **kwargs):
    optimizer_name = 'FedAvg' if optimizer_name in ['LRW', 'RW'] else optimizer_name
    if optimizer_name == 'FedAvg':
        optimizer = FedAvg(*args, **kwargs)
    elif optimizer_name == 'FairFed':
        optimizer = FairFed(*args, **kwargs)
    elif optimizer_name == 'FedGFT':
        optimizer = FedGFT(*args, **kwargs)
    elif optimizer_name == 'FedFairLocal':
        optimizer = FedFairLocal(*args, **kwargs)
    else:
        raise ValueError("Invalid optimizer.")
    return optimizer


if __name__ == '__main__':
    pass
