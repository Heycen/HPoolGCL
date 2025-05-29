from datetime import datetime
import random
from matplotlib import pyplot as plt
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import os
from utils import create_masks, enumerateConfig, init_weights, currentTime
import copy
from data import Dataset, get_wikics
from embedder import embedder
from utils import config2string
from embedder import Encoder
from .layers_pyg import SAGPooling as SAGPool
from .scheduler import CosineDecayScheduler
import logging



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HPoolGCL_ModelTrainer(embedder):

    def __init__(self, args):
        embedder.__init__(self, args)
        self._args = args
        set_seed(args.seed)

        self.config_str = config2string(args)

        current_time = datetime.now().strftime("%m%d_%H%M%S")

        
        log_dir = "runs/{}_{}".format(current_time, self.config_str)

        
        log_filename = "{dataset}_{task}_{time}_lr{lr}_pr{pool_ratio}_a1{a1}_a2{a2}_p{p}_tau{tau}.log".format(
            dataset=args.dataset,
            task=args.task,
            time=current_time,
            lr=args.lr,
            pool_ratio=args.pool_ratio,
            a1=args.a1,
            a2=args.a2,
            p=args.p,
            tau=args.tau,
        )
        log_dir_path = "log"
        log_filepath = os.path.join(log_dir_path, log_filename)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(name)s-%(levelname)s-%(message)s",
            handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("\n[Config] {}\n".format(self.config_str))
        args_names, args_vals = enumerateConfig(args)
        self.logger.info(args_names)
        self.logger.info(args_vals)
        self._init()

    def _init(self):
        args = self._args
        self._task = args.task
        self.logger.info("Downstream Task : {}".format(self._task))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)

        if args.dataset == "wikics":
            self._dataset = get_wikics(root="data/pyg/WikiCS", seed=args.seed)
        else:
            self._dataset = Dataset(root=args.root, dataset=args.dataset)

        maskdir = os.path.join(args.root, "mask")
        
        self.logger.info("Preset mask dir: {}".format(maskdir))
        os.makedirs(maskdir, exist_ok=True)
        mask_path = "{}/{}_mask.pt".format(maskdir, args.dataset)
        if not os.path.exists(mask_path):
            train_mask, val_mask, test_mask = create_masks(
                self._dataset.data, args.dataset, args.seed, mask_path
            )
            self.logger.info(
                "Preset mask for dataset {} not exists. Creatting Now.".format(
                    args.dataset
                )
            )
        else:
            train_mask, val_mask, test_mask = torch.load(mask_path)
            self.logger.info("Preset mask load from {}.".format(mask_path))

        self._dataset.data.train_mask = train_mask
        self._dataset.data.val_mask = val_mask
        self._dataset.data.test_mask = test_mask

        layers = [self._dataset.data.x.shape[1]] + self.hidden_layers

        self._encoder = Encoder(layers, args.dropout, args)
        self._model = HPoolGCL(self._encoder, layers, args).to(self._device)

        self._lr_scheduler = CosineDecayScheduler(
            args.lr, args.warmup_epochs, args.epochs
        )
        self._optimizer = optim.AdamW(
            params=self._model.parameters(), lr=args.lr, weight_decay=1e-5
        )

    def train(self):
        (
            self.best_test_acc,
            self.best_dev_acc,
            self.best_test_std,
            self.best_dev_std,
            self.best_epoch,
        ) = (0, 0, 0, 0, 0)
        self.best_dev_accs = []
        self.best_test_accs = []
        self._dataset.data.to(self._device)
        with torch.no_grad():
            target = self._model.online_representation(
                self._dataset.data.x, self._dataset.data.edge_index
            )

        self.infer_embeddings(self._encoder, 0)
        self.logger.info("initial accuracy ")
        self.evaluate(
            self._task,
            0,
            lr=self._args.lr_cls,
            epochs=self._args.epochs_cls,
            weight_decay=self._args.wd_cls,
            normalize=self._args.norm_cls,
        )

        f_final = open(
            "results/{}_{}.txt".format(self._args.dataset, self._args.task), "a"
        )

        
        self.logger.info("Training Start!")
        self._model.train()

        for epoch in range(self._args.epochs):

            
            if self._args.warmup_epochs != 0:
                lr = self._lr_scheduler.get(epoch)
                for param_group in self._optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = self._args.lr

            _, loss, fit_list = self._model(
                x=self._dataset.data.x,
                edge_index=self._dataset.data.edge_index,
                target=target,
                epoch=epoch,
            )

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            st = "[{}][Epoch {}/{}, lr:{:.6f}] Loss: {:.6f}".format(
                currentTime(), epoch, self._args.epochs, lr, loss.item()
            )
            self.logger.info(st)

            if epoch != 0 and (epoch) % self._args.eval_freq == 0:
                tmp_encoder = copy.deepcopy(self._model.online_encoder).eval()
                self.infer_embeddings(tmp_encoder, epoch)
                self.evaluate(
                    self._task,
                    epoch,
                    lr=self._args.lr_cls,
                    epochs=self._args.epochs_cls,
                    normalize=self._args.norm_cls,
                )
                self.logger.info(self.st_now)
                self.logger.info(self.st_best)

            
            self._model.eval()
            with torch.no_grad():
                target = self._model.online_representation(
                    self._dataset.data.x, self._dataset.data.edge_index
                )

        self.logger.info("\nTraining Done!")
        self.logger.info("[Final] {}".format(self.st_best))
        self.logger.info("Best dev_list: {}".format(self.best_dev_accs))
        self.logger.info("Best test_list: {}".format(self.best_test_accs))

        f_final.write(
            "{} -> {} -> {}\n".format(currentTime(), self.config_str, self.st_best)
        )


class HPoolGCL(nn.Module):
    def __init__(self, encoder, layer_config, args, **kwargs):
        super().__init__()
        self.online_encoder = encoder

        rep_dim = layer_config[-1]

        self.online_projection0 = create_projection_layer(
            rep_dim, args.pred_hid, rep_dim
        )
        self.online_projection0.apply(init_weights)

        self.online_projection1 = create_projection_layer(
            rep_dim, args.pred_hid, rep_dim
        )
        self.online_projection1.apply(init_weights)

        self.online_projection2 = create_projection_layer(
            rep_dim, args.pred_hid, rep_dim
        )
        self.online_projection2.apply(init_weights)

        pool_ratio = args.pool_ratio
        self.SAGPool1 = SAGPool(layer_config[-1], ratio=pool_ratio)
        self.SAGPool2 = SAGPool(layer_config[-1], ratio=pool_ratio)

        self.a1 = args.a1
        self.a2 = args.a2
        self.a3 = args.a3
        self.p = args.p

        self.tau = args.tau
        self.lam = args.lam
        self.dataset = args.dataset

    def coarsen(self, x, edge_index, epoch=None):
        x0 = self.online_encoder(x, edge_index)

        x_list = []  
        pos_list = []
        fit_list = []
        neg_list = []
        edge_list = []
        x_list.append(x0)

        x_p1, edge_index_p1, fitness1, perm1 = self.SAGPool1(x0, edge_index)

        x1_0 = x[perm1]

        x1 = x1_0

        x1 = self.online_encoder(x1, edge_index_p1)

        x_list.append(x1)
        fit_list.append(fitness1)  
        edge_list.append(edge_index_p1)
        pos_list.append(perm1)

        neg_indices1 = torch.ones(x0.size(0), dtype=bool)
        neg_indices1[perm1] = False
        neg_indices1 = torch.where(neg_indices1)[0]
        neg_list.append(neg_indices1)

        x_p2, edge_index_p2, fitness2, perm2 = self.SAGPool2(x1, edge_index_p1)

        x2_0 = x[perm1][perm2]

        x2 = x2_0

        x2 = self.online_encoder(x2, edge_index_p2)

        x_list.append(x2)
        fit_list.append(fitness2)  
        edge_list.append(edge_index_p2)
        pos_list.append(perm2)

        neg_indices2 = torch.ones(x1.size(0), dtype=bool)
        neg_indices2[perm2] = False
        neg_indices2 = torch.where(neg_indices2)[0]
        neg_list.append(neg_indices2)

        return x_list, pos_list, neg_list, fit_list, edge_list

    def online_representation(self, x, edge_index):
        target = self.online_encoder(x, edge_index)
        return target

    def forward(self, x, edge_index, target, epoch=None):

        x_list, pos_list, neg_list, fit_list, edge_list = self.coarsen(
            x, edge_index, epoch
        )
        pred0 = self.online_projection0(x_list[0])
        pred1 = self.online_projection1(x_list[1])
        pred2 = self.online_projection2(x_list[2])

        loss1 = loss_fn(pred0, target.detach(), fit_list[0], tau=self.tau)

        target_pos1 = target[pos_list[0]].detach()
        target_neg1 = target[neg_list[0]].detach()

        loss2 = info_nce_loss(pred1, target_pos1, target_neg1, tau=self.tau)

        neg_all_indices2 = torch.ones(pred0.size(0), dtype=bool)
        neg_all_indices2[pos_list[0][pos_list[1]]] = False
        neg_all_indices2 = torch.where(neg_all_indices2)[0]

        target_pos2 = target[pos_list[0][pos_list[1]]].detach()
        target_neg2 = target[neg_all_indices2].detach()

        loss3 = info_nce_loss(pred2, target_pos2, target_neg2, tau=self.tau)

        con_loss = self.a1 * loss1 + self.a2 * (loss2 + loss3) / 2

        pool_loss1 = loss_pool(
            x_list[0], x_list[0][pos_list[0]], x_list[1], fit_list[0], lam=self.lam
        )
        pool_loss2 = loss_pool(
            x_list[1], x_list[1][pos_list[1]], x_list[2], fit_list[1], lam=self.lam
        )

        pool_loss = (pool_loss1 + pool_loss2) / 2

        loss = con_loss + self.p * pool_loss

        return x_list[0], loss, fit_list


def create_projection_layer(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.PReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def loss_fn(x, y, fitness_scores, tau=0.5):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    sim = (x * y).sum(dim=-1)
    num = sim.size(0)
    sim = torch.exp(sim / tau)
    loss = -torch.log(sim / (num) + 1e-8)
    return loss.mean()


def info_nce_loss(pred, target_pos, target_neg, tau=0.5):

    pred_norm = F.normalize(pred, p=2, dim=-1)
    target_pos_norm = F.normalize(target_pos, p=2, dim=-1)
    target_neg_norm = F.normalize(target_neg, p=2, dim=-1)

    pos_sim_diag = (pred_norm * target_pos_norm).sum(dim=-1)
    neg_sim = torch.mm(pred_norm, target_neg_norm.t())  

    f = lambda x: torch.exp(x / tau)
    
    pos_sim_diag = f(pos_sim_diag)
    neg_sim = f(neg_sim)

    neg_sim_sum = neg_sim.sum(
        1
    )  

    
    loss = -torch.log(
        pos_sim_diag / (neg_sim_sum + pos_sim_diag + 1e-8)
    )  

    return loss.mean()


def loss_pool(x, x_pool, y, fitness, lam=1, gam=1, eps=1e-8):
    
    x_centered = x - x.mean(dim=0)  
    y_centered = y - y.mean(dim=0)  
    x_pool_centered = x_pool - x_pool.mean(dim=0)  

    n, d = x_centered.shape[0], x_centered.shape[1]
    m = y_centered.shape[0]
    C0 = x_centered.t() @ x_centered / (n - 1)  
    C1 = y_centered.t() @ y_centered / (m - 1)  

    diag_C0 = torch.diag(C0)  
    diag_C1 = torch.diag(C1)  

    var_norm_loss = F.mse_loss(diag_C1, diag_C0)

    C_xz = (
        x_pool_centered.t() @ y_centered / (n - 1)
    )  
    redundancy_loss = torch.mean(C_xz**2)
    loss = lam * var_norm_loss + gam * redundancy_loss

    return loss
