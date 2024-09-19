# Mitigating Group Bias in Federated Learning: Beyond Local Fairness

## Introduction

We develop a new algorithm named FedGFT to mitigate the group bias in the federated learning. FedGFT directly optimizes the fairness of global model by solving a regularized objective function consisting of the empirical prediction loss and a penalty term for fairness. Specifically, FedGFT aims to solve the following problem:

$\min_{f} \frac{1}{n}\sum_{i=1}^K \sum_{j=1}^{n_k} \ell(f(X_{ij}), Y_{ij})+\lambda J(F(f)),$

where $f$ is the global model, $F$ is a fairness metric, and $J$ is a penalty term. We found that $J(F(f))$ can also be decomposed to statistics that only depend on local clients, namely $J(F(f))=\sum_{i=1}^K F_k(f)$, thus the optimization of the above equation can be carried out similar to any classic FL algorithm such as FedAvg. More details can be found in our TMLR paper [here](https://openreview.net/pdf?id=ANXoddnzct?).

## Installation

### Prerequisites

See `requirements.txt`.

### Build from source

```bash
git clone https://github.com/KeyWgh/FedGFT.git
cd FedGFT
python setup.py install
```

## Explanation of Files

### Core files:
- `custom_abcmeta.py`: The custom class to implement the abstract class.
- `nn.py`: Define some neural network models and training procedures.
- `utils.py`: The utility module for federated learning.
- `optimizers.py`: Implement optimizers for different federated learning algorithms.

### Experiments Reproduction
- `adult.py`: Run the experiments on the adult dataset.
- `compas.py`: Run the experiments on the compas dataset.
- `celeba.py`: Run the experiments on the celeba dataset.

Before running them, users have to download the datasets and correct the path of datasets in each experiment script.
  

## Cite Our Work

```
@article{wang2024federated,
  title={Mitigating Group Bias in Federated Learning: Beyond Local Fairness},
  author={Wang, Ganghua and Payani, Ali and Lee, Myungjin and Kompella, Ramana},
  journal={Trans. Mach. Learn. Res.},
  year={2024}
}
```


## Contribution and License

This project is licensed under Apache-2.0. Feel free to submit pull requests to contribute or open an issue if you have any questions or suggestions.
