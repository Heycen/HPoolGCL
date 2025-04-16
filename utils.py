from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch.nn.functional as F
import os.path as osp
import os
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split

def currentTime():
    return datetime.now().strftime('%m-%d %H:%M:%S')

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default="A")
    parser.add_argument("--dataset", type=str, default="wikics", help="Name of the dataset. Supported names are: wikics, cs, computers, photo, and physics")
    parser.add_argument('--checkpoint_dir', type=str, default = './model_checkpoints', help='directory to save checkpoint')
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--task", type=str, default="node", help="Downstream task. Supported tasks are: node, clustering, similarity")
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument("--device", type=int, default=0)
    
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Warmup period for learning rate.")
    
    parser.add_argument("--epochs", type=int, default=1101)
    parser.add_argument("--eval_freq", type=float, default=10, help="The frequency of model evaluation") 
    
    parser.add_argument("--layers", nargs='?', default='[1024]', help="The number of units of each layer of the GNN. Default is [256]")
    parser.add_argument("--pred_hid", type=int, default=2048, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--dropout", type=float, default=0.0)

    
    parser.add_argument("--lr_cls", type=float, default=0.002, help="The learning rate for model training for node classification classifier.")
    parser.add_argument("--wd_cls", type=float, default=0.00001, help="The value of the weight decay for training for node classification classifier.")
    parser.add_argument("--epochs_cls", type=int, default=600, help="The number of training epochs for node classification classifier.")
    parser.add_argument("--norm_cls", action="store_true", help="Enable normalization (default: False)")
    

    parser.add_argument("--pool_dropout", type=float, default=0.2, help="投影层dropout")
    parser.add_argument("--pool_ratio", type=float, default=0.7)

    
    parser.add_argument("--a1", type=float, default=1, help="Muti-granularity loss weight,infoNCE loss")
    parser.add_argument("--a2", type=float, default=1, help="Muti-granularity loss weight,infoNCE loss")
    parser.add_argument("--a3", type=float, default=1, help="Muti-granularity loss weight,infoNCE loss")
    parser.add_argument("--tau", type=float, default=0.5, help="temperature of infoNCE loss")
    
    
    parser.add_argument("--p", type=float, default=0.0001, help="pool loss weight")
    parser.add_argument("--lam", type=float, default=1, help="weight of redundancy")
    
    return parser.parse_known_args()


def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Planetoid, "src": "pyg"}
    elif dataset == "computers":
        dataset = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "photo":
        dataset = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "cs" :
        dataset = "CS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "physics":
        dataset = "Physics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "wikics":
        dataset = "WikiCS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": WikiCS, "src": "pyg"}
    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics'")
    return params


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def create_masks(dataset, dataset_name='wikics', data_seed=0, mask_path=None):
    r"""Create train/val/test mask for each dataset."""
    data = dataset
    if dataset_name == 'wikics':
        train_mask = data.train_mask.T
        val_mask = data.val_mask.T
        test_mask = data.test_mask.repeat(20, 1)
        
    elif dataset_name in ['computers', 'photo', 'cs', 'physics']:
        idx = np.arange(len(data.y))
    
        train_mask = torch.zeros((20, data.y.size(0)), dtype=torch.bool)
        val_mask = torch.zeros((20, data.y.size(0)), dtype=torch.bool)
        test_mask = torch.zeros((20, data.y.size(0)), dtype=torch.bool)

        for i in range(20):
            train_idx, test_idx = train_test_split(idx, test_size=0.8, random_state=data_seed + i)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.5, random_state=data_seed + i)

            train_mask[i,train_idx] = True
            val_mask[i, val_idx] = True
            test_mask[i, test_idx] = True
    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    elif dataset_name in ['mag']:
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        train_mask[split_idx['train']['paper']] = True
        val_mask[split_idx['valid']['paper']] = True
        test_mask[split_idx['test']['paper']] = True        
    else:
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True

    
    if mask_path is not None:
        torch.save([train_mask, val_mask, test_mask], mask_path)

    return train_mask, val_mask, test_mask


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['embedder','device','root','epochs','isAnneal','pool_dropout','clus_num_iters','checkpoint_dir']:
            st_ = "{}{}_".format(name, val)
            st += st_

    return st[:-1]


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def fill_ones(x):
    n_data = x.shape[0]
    x = torch.sparse_coo_tensor(x._indices(), torch.ones(x._nnz()).to(x.device), [n_data, n_data])

    return x

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

