import torch.utils.data

from utils import Server, Trainer, SimData
from optimizer import get_optimizer
from nn import *
import numpy as np
import torchvision.datasets as datasets
import logging
import sys, os
from datetime import datetime
from mpi4py import MPI
from pickle import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
rng = np.random.RandomState(0)


def my_excepthook(excType, excValue, traceback):
    logger.error("Logging an uncaught exception",
                 exc_info=(excType, excValue, traceback))


def adult_split(num_clients, group, alpha=5.):
    path = '~/projects/Datasets/adult/'
    # col_names = pd.read_table(path + 'colname.txt', header=None, sep='\n')[0].tolist()
    adult_dataset = pd.read_table(path + 'adult.csv', sep=',', header=0)
    adult_dataset = adult_dataset.drop(['fnlwgt', 'education'], axis=1)
    adult_dataset['race'] = 1 * (adult_dataset['race'] == 'White')
    # adult_dataset['sex'] = 1 * (adult_dataset['sex'] == 'Male')
    adult_dataset['income'] = 1 * (adult_dataset['income'] == '>50K')
    df = pd.get_dummies(adult_dataset, drop_first=True)
    dataset = SimData(df.drop('income', axis=1).astype('float32'), df['income'], df[group])
    test_size = int(len(adult_dataset) * .3)
    _, adult_testset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

    logger.info("Splitting Adult dataset.")
    logger.info('Heterogeneity parameter alpha: {}'.format(alpha))
    adult_data = [np.where((dataset.group == i) & (dataset.targets == j))[0] for i in range(2) for j in range(2)]
    dict_users = {i: np.empty(0).astype(int) for i in range(num_clients)}
    idx = np.zeros(4, dtype=np.int64)

    props = np.random.dirichlet([alpha] * num_clients, size=(4))
    props = np.array([[len(v)] for v in adult_data]) * props
    for user in range(num_clients):
        for j in range(4):
            num_samples = int(props[j, user])
            dict_users[user] = np.concatenate((dict_users[user], adult_data[j][idx[j]:idx[j] + num_samples]), axis=0)
            idx[j] += num_samples

    logger.debug("Clients sample size: {}".format([len(v) for v in dict_users.values()]))
    return dataset, adult_testset, group, dict_users


def adult(num_clients=5, num_iter=10, gamma=0.2, group='race', fair='SP', optimizer='FedFair', pre=True, alpha=5.,
          save_model=False, device='cpu', lr=2e-3, epochs=1):
    logger.info("Running Adult example.")
    adult_dataset, adult_testset, group, client_groups = adult_split(num_clients, group, alpha=alpha)
    num_features = adult_dataset.dataset.shape[1]
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Initializing server and {num_clients} clients.")
    logger.info(f"Penalty parameter gamma: {gamma}")
    logger.info(f"Fairness metric: {fair}")
    logger.info(f"Optimizer: {optimizer}")
    optimizer_args = {'gamma': gamma, 'beta': 1, 'reg': 'l2'}
    reweigh = True if optimizer in ['FairFed', 'LRW', 'RW'] else False
    algorithm = get_optimizer(optimizer, **optimizer_args)

    testloader = torch.utils.data.DataLoader(adult_testset, batch_size=batch_size, shuffle=True)
    if pre:
        with open(f'../saved_models/adult/{fair}_FedAvg_{alpha}.pkl', "rb") as input_file:
            logger.info("Loading pre-trained model.")
            pre_model = load(input_file)
    else:
        pre_model = LogisticRegression(in_features=num_features)
    server = Server(pre_model,
                    device=device, test_loader=testloader, fair=fair, algorithm=algorithm)
    clients = []
    for i in range(num_clients):
        client = Trainer(LogisticRegression(in_features=num_features), fair=fair, algorithm=algorithm, reweigh=reweigh,
                         lr=lr, epochs=epochs,
                         dataset=adult_dataset, idx=client_groups[i], device=device, batch_size=batch_size)
        clients.append(client)

    # w_list = [sum(adult_dataset.targets[c] == 1)/len(c) for c in client_groups.values()]
    # logger.debug(f'b_list: {b_list}')
    # logger.debug(f'w_list: {w_list}')

    server.num_sample = sum([client.dataset_size for client in clients])
    for letter in 'abcd':
        server.bias[letter] = sum([client.bias[letter] * client.dataset_size / server.num_sample for client in clients])

    b_list = np.array([client.bias['b'] / client.bias['d'] for client in clients])
    logger.info(f'ratio: {np.max(b_list) - np.min(b_list)}')
    logger.info(f'DH: {np.max(np.abs(b_list * (server.bias["d"] / server.bias["b"]) - 1))}')

    logger.info(f"Starting FL training for {num_iter} iterations.")
    for t in range(num_iter):
        # Update clients
        for client in clients:
            client.sync(server)
            client.train()  # local update weights

        # Update server
        server.aggregate_weights(clients)
        for client in clients:
            client.update_model(server)
            client.update_bias()
        server.update_bias(clients)

        logger.debug(f"Iteration {t + 1}/{num_iter}, server loss: {[(k, v[-1]) for k, v in server.metrics.items()]}")
        if pre and (np.abs(server.metrics['bias'][-1]) < 0.01):
            server.metrics['bias'].extend([server.metrics['bias'][-1]] * (num_iter - t - 1))
            server.metrics['test-accuracy'].extend([server.metrics['test-accuracy'][-1]] * (num_iter - t - 1))
            break

    if save_model:
        with open(f'../saved_models/adult/{fair}_{optimizer}_{alpha}.pkl', 'wb') as outp:
            dump(server.model, outp)
    logger.info(f"FL training finished.")
    return server


def compare():
    nrep = 10 // size
    methods = ['FedAvg', 'FairFed', 'LRW', 'FedFairLocal', 'FedGFT']
    fair_list = ['SP', 'EOP'] # 'SP', 
    # g_list = [1, 1]
    # fair_list = [, 'EOP', 'CAL']
    alpha_list = [0.5, 5., 100.]
    # alpha_list = [100., ]
    # g_list = [5, 5]
    # gamma = {k: v for k, v in zip(fair_list, g_list)}
    v = 20
    epochs = 1
    lr = 2e-3
    gamma = {'FedAvg': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'LRW': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'RW': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'FairFed': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'FedGFT': {'SP': dict(zip(alpha_list, [v, v, v])),
                         'EOP': dict(zip(alpha_list, [v, v, v]))},
             'FedFairLocal': {'SP': dict(zip(alpha_list, [10, 10, 10])),
                              'EOP': dict(zip(alpha_list, [10, 10, 10]))}}
    num_iter = 50
    for alpha in alpha_list:
        for method in methods:
            for fair in fair_list:
                if rank == 0:
                    logger.warning('Alpha: {}, Method: {}, Fairness: {}'.format(alpha, method, fair))
                acc = np.zeros((nrep, num_iter))
                bias = np.zeros((nrep, num_iter))
                for i in range(nrep):
                    server = adult(num_clients=10, num_iter=num_iter, gamma=gamma[method][fair][alpha],
                                   fair=fair, optimizer=method, pre=False, alpha=alpha, lr=lr, epochs=epochs)
                    acc[i, :] = server.metrics['test-accuracy']
                    bias[i, :] = server.metrics['bias']

                acc = comm.gather(acc, root=0)
                bias = comm.gather(bias, root=0)
                if rank == 0:
                    res = np.concatenate(acc, axis=0)
                    ans_err = np.concatenate(bias, axis=0)
                    with open(
                            f'../saved_models/adult/{fair}_{method}_{alpha}_{gamma[method][fair][alpha]}_client_{10}_epoch_{epochs}_lr_{lr}.pkl',
                            'wb') as output:
                        dump({'test-accuracy': res, 'bias': ans_err}, output)


if __name__ == '__main__':
    logPath = '../logs/'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    sys.excepthook = my_excepthook
    fileName = str(datetime.now()).split('.')[0]
    logging.basicConfig(
        # level=logging.DEBUG,
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("{0}{1}.log".format(logPath, fileName)),
            # logging.StreamHandler()
        ]
    )
    logger.warning('Adult dataset')
    # mnist(num_clients=10, num_iter=2)
    # adult(num_clients=10, num_iter=40, fair='EOP', gamma=20, optimizer='FedGFT', pre=False, alpha=5., lr=2e-3, epochs=3)
    compare()
    # pre_train()
    # pass

