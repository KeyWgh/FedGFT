import torch.utils.data

from utils import Server, Trainer
from optimizer import get_optimizer
from nn import *
import numpy as np
import torchvision.datasets as datasets
import logging
import sys, os
from datetime import datetime
from pickle import dump, load
from mpi4py import MPI
import PIL
from typing import Any, Tuple
from torchvision.models import resnet18

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
rng = np.random.RandomState(0)


class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        # create cnn model
        model = resnet18(pretrained=True)
        # remove fc layers and add a new fc layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # classifier
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        # Since NLL or KL divergence uses log-probability as input, we need to use log_softmax
        probs = F.log_softmax(logits, dim=1)
        return probs


class GroupCelebA(datasets.CelebA):
    def __init__(self, *args, group=None, **kwargs):
        self.group = group
        super(GroupCelebA, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            group = target[:, self.group] if len(target.shape) > 1 else target[self.group]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
            group = None

        return X, target, group


def my_excepthook(excType, excValue, traceback):
    logger.error("Logging an uncaught exception",
                 exc_info=(excType, excValue, traceback))


def celeba_split(num_clients, group, alpha=5.):
    path = '~/projects/Datasets/'

    # target = 31  # index of 'Smiling'
    target_transform = lambda x: x[:, 31] if len(x.shape) > 1 else x[31]
    image_size = 224
    # h, w = 218, 178  # the width and the hight of original images before resizing
    imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
    imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    celeba_dataset = GroupCelebA(path, target_type=["attr"], group=group, transform=transforms,
                          split='train', target_transform=target_transform)
    celeba_testset = GroupCelebA(path, target_type=["attr"], group=group, transform=transforms,
                                 split='test', target_transform=target_transform)
    test_size = int(len(celeba_testset) * .3)
    _, celeba_testset = torch.utils.data.random_split(celeba_testset, [len(celeba_testset) - test_size, test_size])

    logger.info("Splitting CelebA dataset.")
    logger.info('Heterogeneity parameter alpha: {}'.format(alpha))
    # celeba_data = [np.where(celeba_dataset.attr[:, group] == i)[0] for i in range(2)]
    celeba_data = [np.where((celeba_dataset.attr[:, group] == i) & (celeba_dataset.attr[:, 31] == j))[0]
                   for i in range(2) for j in range(2)]
    dict_users = {i: np.empty(0).astype(int) for i in range(num_clients)}
    idx = np.zeros(4, dtype=np.int64)
    prop = 0.05
    props = np.random.dirichlet([alpha] * num_clients, size=(4))
    props = np.array([[int(len(v)*prop)-10*num_clients] for v in celeba_data]) * props + 10

    for user in range(num_clients):
        for j in range(4):
            num_samples = int(props[j, user])
            dict_users[user] = np.concatenate((dict_users[user], celeba_data[j][idx[j]:idx[j] + num_samples]), axis=0)
            idx[j] += num_samples

    logger.debug("Clients sample size: {}".format([len(v) for v in dict_users.values()]))
    return celeba_dataset, celeba_testset, dict_users


def celeba(num_clients=5, num_iter=10, gamma=0.2, group=20, fair='SP', optimizer='FedGFT', pre=False, alpha=5., lr=2e-3, epochs=3):
    logger.info("Running Celeba example.")
    celeba_dataset, celeba_testset, client_groups = celeba_split(num_clients, group, alpha=alpha)
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Initializing server and {num_clients} clients.")
    logger.info(f"Penalty parameter gamma: {gamma}")
    logger.info(f"Fairness metric: {fair}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f'Device: {device}')
    optimizer_args = {'gamma': gamma, 'beta': 1, 'reg': 'l2'}
    reweigh = True if optimizer in ['FairFed', 'LRW', 'RW'] else False
    algorithm = get_optimizer(optimizer, **optimizer_args)

    testloader = torch.utils.data.DataLoader(celeba_testset, batch_size=batch_size, shuffle=True)
    if pre:
        with open(f'../saved_models/celeba/SP_FedAvg.pkl', "rb") as input_file:
            logger.info("Loading pre-trained model.")
            pre_model = load(input_file)
    else:
        pre_model = Classifier()
    server = Server(pre_model,
                    device=device, test_loader=testloader, fair=fair, algorithm=algorithm)
    clients = []
    logger.info(f"Learning rate: {lr}")
    for i in range(num_clients):
        client = Trainer(Classifier(), fair=fair, algorithm=algorithm, reweigh=reweigh, lr=lr, epochs=epochs,
                         dataset=celeba_dataset, idx=client_groups[i], device=device, batch_size=batch_size)
        clients.append(client)

    server.num_sample = sum([client.dataset_size for client in clients])
    for letter in 'abcd':
        server.bias[letter] = sum([client.bias[letter] * client.dataset_size / server.num_sample for client in clients])

    if optimizer == 'RW':
        dataloaders = [client.train_loader for client in clients]
        weights = cal_weights(dataloaders, device=device)
        for client in clients:
            client.weights = weights

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

    logger.info(f"FL training finished.")
    return server


def compare():
    nrep = 9 // size
    methods = ['FedAvg', 'LRW', 'FairFed', 'FedFairLocal', 'FedGFT']
    fair_list = ['SP', 'EOP']
    # g_list = [1, 1]
    # fair_list = [, 'EOP', 'CAL']
    # alpha_list = [0.5, 5., 100.]
    alpha_list = [0.5,]
    # g_list = [5, 5]
    # gamma = {k: v for k, v in zip(fair_list, g_list)}
    v = 10
    gamma = {'FedAvg': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'LRW': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'RW': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'FairFed': {'SP': {x: 1 for x in alpha_list}, 'EOP': {x: 1 for x in alpha_list}},
             'FedGFT': {'SP': dict(zip(alpha_list, [v, v, v])),
                         'EOP': dict(zip(alpha_list, [v, v, v]))},
             'FedFairLocal': {'SP': dict(zip(alpha_list, [10, 10, 10])),
                              'EOP': dict(zip(alpha_list, [10, 10, 10]))}}
    num_iter = 50
    epochs = 1
    lr = 5e-4
    for alpha in alpha_list:
        for method in methods:
            for fair in fair_list:
                if rank == 0:
                    logger.warning('Alpha: {}, Method: {}, Fairness: {}'.format(alpha, method, fair))
                acc = np.zeros((nrep, num_iter))
                bias = np.zeros((nrep, num_iter))
                for i in range(nrep):
                    server = celeba(num_clients=10, num_iter=num_iter, gamma=gamma[method][fair][alpha], group=20,
                                   fair=fair, optimizer=method, pre=False, alpha=alpha, epochs=epochs, lr=lr)
                    acc[i, :] = server.metrics['test-accuracy']
                    bias[i, :] = server.metrics['bias']

                acc = comm.gather(acc, root=0)
                bias = comm.gather(bias, root=0)
                if rank == 0:
                    res = np.concatenate(acc, axis=0)
                    ans_err = np.concatenate(bias, axis=0)
                    with open(
                            f'../saved_models/celeba/{fair}_{method}_{alpha}_{gamma[method][fair][alpha]}_client_{10}_epochs_{epochs}_lr_{lr}.pkl',
                            'wb') as output:
                        dump({'test-accuracy': res, 'bias': ans_err}, output)


if __name__ == '__main__':
    logPath = '../logs/'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    sys.excepthook = my_excepthook
    fileName = str(datetime.now()).split('.')[0]
    logging.basicConfig(
        level=logging.DEBUG,
        # level = logging.WARNING,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # logging.FileHandler("{0}{1}.log".format(logPath, fileName)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('celeba')
    logger.warning('Celeba')
    celeba(num_clients=10, num_iter=20, gamma=20, group=20, fair='EOP', optimizer='LRW', pre=False, alpha=.5, epochs=1, lr=5e-4)
    # compare()
