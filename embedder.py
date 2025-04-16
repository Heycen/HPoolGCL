import copy
import numpy as np
import torch
import torch.nn as nn
from utils import printConfig
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import random
import os

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise,homogeneity_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class embedder:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.hidden_layers = eval(args.layers)
        printConfig(args)

    def infer_embeddings(self,encoder,epoch):
        encoder.eval()
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None
        self._dataset.data.to(self._device)
        emb= encoder(x=self._dataset.data.x, edge_index=self._dataset.data.edge_index)
        emb = emb.detach()
        y = self._dataset.data.y.detach()
        self._embeddings, self._labels = emb, y

    def evaluate(self, task, epoch,lr=2e-2, epochs=300,weight_decay=1e-5, normalize=True):
        if task == "node":
            self.evaluate_node(epoch,lr=lr, epochs=epochs,weight_decay=weight_decay,normalize=normalize)
        elif task == "clustering":
            self.evaluate_clustering(epoch)
        elif task == "similarity":
            self.run_similarity_search(epoch)
    
    def evaluate_node(self, epoch,lr=2e-2, epochs=300,weight_decay=1e-5, normalize=True):
        
        if normalize:
            embeddings  = F.normalize(self._embeddings, dim=1)
        else:
            embeddings = self._embeddings

        emb_dim, num_class = embeddings.shape[1], self._labels.unique().shape[0]
        
        dev_accs, test_accs = [], []
        
        classifier = Classifier(emb_dim, num_class).to(self._device)

        for i in range(20):
            train_mask = self._dataset[0].train_mask[i]
            val_mask = self._dataset[0].val_mask[i]
            test_mask = self._dataset[0].test_mask[i]
                
            train_x, train_y = embeddings[train_mask], self._labels[train_mask]
            val_x, val_y = embeddings[val_mask], self._labels[val_mask]
            test_x, test_y = embeddings[test_mask], self._labels[test_mask]
                
            
            
            
            classifier.reset_parameters()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
            for _ in range(epochs):
                classifier.train()
                optimizer.zero_grad()
                
                output = classifier(train_x)
                loss = F.nll_loss(output, train_y)
                loss.backward()
                optimizer.step()
                
            dev_acc, test_acc = eval_acc(classifier, val_x, val_y), eval_acc(classifier, test_x, test_y)
        
            
            
            

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        
        

        dev_acc, dev_std = np.mean(dev_accs), np.std(dev_accs)
        test_acc, test_std = np.mean(test_accs), np.std(test_accs)
        
        self.st_now = '** [{}] [Epoch: {}/{}] Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(self.args.embedder, epoch,self.args.epochs, dev_acc, dev_std, test_acc, test_std)

        

        
        
        if epoch != 0:
            
            if test_acc > self.best_test_acc:
                self.best_dev_acc = dev_acc
                self.best_test_acc = test_acc
                self.best_dev_std = dev_std
                self.best_test_std = test_std
                self.best_epoch = epoch
                self.best_dev_accs = copy.deepcopy(dev_accs)
                self.best_test_accs = copy.deepcopy(test_accs)

            self.st_best = '**** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f}) ****\n'.format(
                self.best_epoch, self.best_dev_acc, self.best_dev_std, self.best_test_acc, self.best_test_std)        
        
        
        
        



    def evaluate_clustering(self, epoch):
        
        embeddings = F.normalize(self._embeddings, dim = -1, p = 2).detach().cpu().numpy()
        nb_class = len(self._dataset[0].y.unique())
        true_y = self._dataset[0].y.detach().cpu().numpy()

        estimator = KMeans(n_clusters = nb_class)

        NMI_list = []
        Hom_list = []
        for i in range(10):
            estimator.fit(embeddings)
            y_pred = estimator.predict(embeddings)

            s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
            s2 = homogeneity_score(true_y, y_pred) 
            
            NMI_list.append(s1)
            Hom_list.append(s2) 

        s1 = sum(NMI_list) / len(NMI_list)
        s2 = sum(Hom_list) / len(Hom_list)
        self.st_now = '** [{}] [Current Epoch {}] Clustering NMI: {:.4f} Hom: {:.4f} **'.format(self.args.embedder, epoch, s1,s2)
        

        if s1 > self.best_dev_acc:
            self.best_epoch = epoch
            self.best_dev_acc = s1

            if self._args.checkpoint_dir != '':
                print('Saving checkpoint...')
                torch.save(self._embeddings.detach().cpu(), os.path.join(self._args.checkpoint_dir, 'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))
        if s2 > self.best_test_acc:
            self.best_test_acc = s2            
        self.best_dev_accs.append(self.best_dev_acc)
        self.best_test_accs.append(self.best_test_acc)
        self.st_best = '** [Best epoch: {}] Best NMI: {:.4f} Best Hom: {:.4f} **\n'.format(self.best_epoch, self.best_dev_acc,self.best_test_acc)
        


    def run_similarity_search(self, epoch):

        test_embs = self._embeddings.detach().cpu().numpy()
        test_lbls = self._dataset[0].y.detach().cpu().numpy()
        numRows = test_embs.shape[0]

        cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
        st = []
        for N in [5, 10]:
            indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
            tmp = np.tile(test_lbls, (numRows, 1))
            selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
            original_label = np.repeat(test_lbls, N).reshape(numRows,N)
            st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4))

        self.st_now = '** [{}] [Current Epoch {}] sim@5 : {} | sim@10 : {} **'.format(self.args.embedder, epoch, st[0], st[1])
        

        if st[0] > self.best_dev_acc:
            self.best_dev_acc = st[0]
            self.best_test_acc = st[1]
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.best_test_accs.append(self.best_test_acc)
        self.st_best = '** [Best epoch: {}] Best @5 : {} | Best @10: {} **\n'.format(self.best_epoch, self.best_dev_acc, self.best_test_acc)
        

        return st

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
        self.linear.reset_parameters()

def eval_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        y_pred = torch.argmax(output, dim=1).squeeze(-1)
        
        return (y_pred == y).float().mean().item()


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        self.stacked_gnn = nn.ModuleList([GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList([nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=edge_weight)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)

        return x

