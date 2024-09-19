import torch.utils.data

from utils import Server, Trainer, SimData, cal_weights
from optimizer import get_optimizer
from nn import *
import numpy as np
# import torchvision.datasets as datasets
import logging
import sys, os
from datetime import datetime
from mpi4py import MPI
from pickle import dump, load
import pandas as pd


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
rng = np.random.RandomState(0)


def my_excepthook(excType, excValue, traceback):
    logger.error("Logging an uncaught exception",
                 exc_info=(excType, excValue, traceback))


def compas_split(num_clients, group, alpha=5.):
    path = '~/projects/Datasets/compas/'
    # col_names = pd.read_table(path + 'colname.txt', header=None, sep='\n')[0].tolist()
    compas_dataset = pd.read_table(path + 'propublica_data_for_fairml.csv', sep=',', header=0)
    # drop_list = ['Person_ID', 'AssessmentID', 'Case_ID', 'LastName', 'ScaleSet_ID', 'FirstName', 'MiddleName',
    #              'AssessmentReason', 'Screening_Date', 'RawScore', 'DecileScore', 'IsCompleted', 'IsDeleted']
    compas_dataset = compas_dataset.drop(['score_factor'], axis=1)
    # compas_dataset['DateOfBirth'] = compas_dataset['DateOfBirth'].apply(lambda x: int(x[-2:]))
    # compas_dataset['Ethnic_Code_Text'] = 1 * (compas_dataset['Ethnic_Code_Text'] == 'Caucasian')
    # compas_dataset['Two_yr_Recidivism'] = 1 * (compas_dataset['Two_yr_Recidivism'] == 'Low')
    # df = pd.get_dummies(compas_dataset, drop_first=True)
    df = compas_dataset
    dataset = SimData(df.drop(['Two_yr_Recidivism'], axis=1).astype('float32'), df['Two_yr_Recidivism'], df[group])
    test_size = int(len(compas_dataset) * .3)
    _, compas_testset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

    logger.info("Splitting Compas dataset.")
    logger.info('Heterogeneity parameter alpha: {}'.format(alpha))
    compas_data = [np.where((dataset.group == i) & (dataset.targets == j))[0] for i in range(2) for j in range(2)]
    dict_users = {i: np.empty(0).astype(int) for i in range(num_clients)}
    idx = np.zeros(4, dtype=np.int64)

    # props = np.random.dirichlet([alpha] * num_clients, size=(4))
    # for user in range(num_clients):
    #     if user % 2 == 0:
    #         props[2, user] += props[0, user]
    #         props[3, user] += props[1, user]
    #         props[0, user] = 0
    #         props[1, user] = 0
    #     else:
    #         props[0, user] += props[2, user]
    #         props[1, user] += props[3, user]
    #         props[2, user] = 0
    #         props[3, user] = 0
    # for j in range(4):
    #     props[j, :] /= np.sum(props[j, :])
    
    props = np.random.dirichlet([alpha] * num_clients, size=(1))
    n = np.array([[len(v)] for v in compas_data])
    props = np.outer(n/sum(n), props)

    props = n * props
    for user in range(num_clients):
        for j in range(4):
            num_samples = max(int(props[j, user]), 1)
            dict_users[user] = np.concatenate((dict_users[user], compas_data[j][idx[j]:idx[j] + num_samples]), axis=0)
            idx[j] += num_samples

    logger.debug("Clients sample size: {}".format([len(v) for v in dict_users.values()]))
    return dataset, compas_testset, group, dict_users


def compas(num_clients=5, num_iter=10, gamma=0.2, group='Female', fair='SP', optimizer='FedGFT', pre=False, alpha=5.,
          save_model=False, device='cpu', lr=2e-3, epochs=1):
    logger.info("Running Compas example.")
    compas_dataset, compas_testset, group, client_groups = compas_split(num_clients, group, alpha=alpha)
    num_features = compas_dataset.dataset.shape[1]
    batch_size = 256

    logger.info(f"Initializing server and {num_clients} clients.")
    logger.info(f"Penalty parameter gamma: {gamma}")
    logger.info(f"Fairness metric: {fair}")
    logger.info(f"Optimizer: {optimizer}")
    optimizer_args = {'gamma': gamma, 'beta': 1, 'reg': 'l2'}
    reweigh = True if optimizer in ['FairFed', 'LRW', 'RW'] else False
    algorithm = get_optimizer(optimizer, **optimizer_args)

    testloader = torch.utils.data.DataLoader(compas_testset, batch_size=batch_size, shuffle=True)
    if pre:
        with open(f'../saved_models/compas/{fair}_FedAvg_{alpha}.pkl', "rb") as input_file:
            logger.info("Loading pre-trained model.")
            pre_model = load(input_file)
    else:
        pre_model = LogisticRegression(in_features=num_features)
    server = Server(pre_model,
                    device=device, test_loader=testloader, fair=fair, algorithm=algorithm)
    clients = []
    for i in range(num_clients):
        client = Trainer(LogisticRegression(in_features=num_features), fair=fair, algorithm=algorithm, reweigh=reweigh,
                         lr=lr,
                         dataset=compas_dataset, idx=client_groups[i], device=device, epochs=epochs, batch_size=batch_size)
        clients.append(client)

    if optimizer == 'RW':
        dataloaders = [client.train_loader for client in clients]
        weights = cal_weights(dataloaders, device=device)
        for client in clients:
            client.weights = weights

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
            # client.update_model(server)  # fetch model from server
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

    val = np.max([client.bias['local_val'] for client in clients])
    logger.info(f'max local bias: {val:.2f}')
    if save_model:
        with open(f'../saved_models/compas/{fair}_{optimizer}_{alpha}.pkl', 'wb') as outp:
            dump(server.model, outp)
    # server.plot_metrics()
    logger.info(f"FL training finished.")
    return server


def compare():
    nrep = 10 // size
    methods = ['FedAvg', 'FairFed', 'LRW', 'FedFairLocal', 'FedGFT']
    fair_list = ['SP', 'EOP'] # 'SP', 
    # methods = ['FedFairLocal']
    # alpha_list = [0.5, 5., 100.]
    alpha_list = [ .5,]
    v = 20
    epochs = 3
    # epoch_list = [1,3,5]
    epoch_list = [3]
    lr=0.01
    # lr_list = [0.002, 0.005, 0.01]
    lr_list = [0.01]
    clients = 10
    client_list = [20] # 20, 50, 
    # client_list = [10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f'Running on {device}')
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
            for lr in lr_list:
                # for epochs in epoch_list:
                for clients in client_list:
                    for fair in fair_list:
                        if rank == 0:
                            logger.warning('Alpha: {}, Method: {}, Fairness: {}'.format(alpha, method, fair))
                        acc = np.zeros((nrep, num_iter))
                        bias = np.zeros((nrep, num_iter))
                        for i in range(nrep):
                            server = compas(num_clients=clients, num_iter=num_iter, gamma=gamma[method][fair][alpha], lr=lr,
                                           fair=fair, optimizer=method, pre=False, alpha=alpha, device=device, epochs=epochs)
                            acc[i, :] = server.metrics['test-accuracy']
                            bias[i, :] = server.metrics['bias']

                        acc = comm.gather(acc, root=0)
                        bias = comm.gather(bias, root=0)
                        if rank == 0:
                            res = np.concatenate(acc, axis=0)
                            ans_err = np.concatenate(bias, axis=0)
                            with open(
                                    f'../saved_models/compas/{fair}_{method}_{alpha}_{gamma[method][fair][alpha]}_client_{clients}_epoch_{epochs}_lr_{lr}_uneven.pkl',
                                    'wb') as output:
                                dump({'test-accuracy': res, 'bias': ans_err}, output)


def ablation():
    nrep = 20 // size
    fair_list = ['SP', 'EOP']
    alpha_list = [0.5, 5., 100.]
    methods = [1,2,5]
    v = 20
    gamma = {'SP': dict(zip(alpha_list, [v, v, v])),
                         'EOP': dict(zip(alpha_list, [v, v, v]))}
    num_iter = 50
    for alpha in alpha_list:
        for m in methods:
            for fair in fair_list:
                acc = np.zeros((nrep, num_iter))
                bias = np.zeros((nrep, num_iter))
                for i in range(nrep):
                    server = compas(num_clients=10, num_iter=num_iter, gamma=gamma[fair][alpha],
                                   fair=fair, optimizer='FedGFT', pre=False, alpha=alpha, epochs=m)
                    acc[i, :] = server.metrics['test-accuracy']
                    bias[i, :] = server.metrics['bias']

                acc = comm.gather(acc, root=0)
                bias = comm.gather(bias, root=0)
                if rank == 0:
                    res = np.concatenate(acc, axis=0)
                    ans_err = np.concatenate(bias, axis=0)
                    with open(
                            f'../saved_models/compas/{fair}_epoch_{m}_{alpha}_client_{10}_metric_hatD.pkl',
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
    # compas(num_clients=10, num_iter=50, fair='SP', gamma=20, optimizer='FedGFT', pre=False, alpha=0.5)
    compare()
    # ablation()
    # pass
