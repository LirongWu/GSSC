import argparse
from torch import nn
from torch.nn import Parameter
import dgl.function as fn
from torch.nn import functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import torch
from tqdm import trange
from utils import generate_random_seeds, set_random_state, evaluate

import csv
import dgl
import time
import pickle

class DAGNNConv(nn.Module):
    def __init__(self,
                 in_dim,
                 k):
        super(DAGNNConv, self).__init__()

        self.s = Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):

        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata['h'] = feats
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feats = graph.ndata['h']
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H


class MLPLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 activation=None,
                 dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.
        if self.activation is F.relu:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):

        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats


class DAGNN(nn.Module):
    def __init__(self,
                 k,
                 in_dim,
                 hid_dim,
                 out_dim,
                 bias=True,
                 activation=F.relu,
                 dropout=0, ):
        super(DAGNN, self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(MLPLayer(in_dim=in_dim, out_dim=hid_dim, bias=bias,
                                 activation=activation, dropout=dropout))
        self.mlp.append(MLPLayer(in_dim=hid_dim, out_dim=out_dim, bias=bias,
                                 activation=None, dropout=dropout))
        self.dagnn = DAGNNConv(in_dim=out_dim, k=k)

    def forward(self, graph, feats):
        for layer in self.mlp:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
        return feats


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    if args.dataset == 'cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'Pubmed':
        dataset = PubmedGraphDataset()
    elif args.dataset == 'coauthor-cs':
        dataset = dgl.data.CoauthorCSDataset()
    elif args.dataset == 'coauthor-phy':
        dataset = dgl.data.CoauthorPhysicsDataset() 
    else:
        print('Unknown dataset: {}'.format(args.dataset))

    if args.dataset == 'film':
        adj = pickle.load(open('../dataset/film/{}_adj.pkl'.format(args.dataset), 'rb'))
        row, col = np.where(adj.todense() > 0)
        U = row.tolist()
        V = col.tolist()
        graph = dgl.graph((U, V))
        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)
        graph.ndata['feat'] = torch.FloatTensor(pickle.load(open('../dataset/film/{}_features.pkl'.format(args.dataset), 'rb')))
        graph.ndata['label'] = torch.LongTensor(pickle.load(open('../dataset/film/{}_labels.pkl'.format(args.dataset), 'rb')))
    else:
        graph = dataset[0]

    if args.mode < -20 and args.mode >= -50:
        src = np.load("../dataset/{}/src_mask_r{}.npy".format(args.dataset, -args.mode))
        dst = np.load("../dataset/{}/dst_mask_r{}.npy".format(args.dataset, -args.mode))
        U = src.tolist()
        V = dst.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)

        if graph.number_of_nodes() - g.number_of_nodes() > 0:
            g.add_nodes(graph.number_of_nodes() - g.number_of_nodes())

        g = dgl.to_bidirected(g)

        g.ndata['feat'] = graph.ndata['feat']
        g.ndata['label'] = graph.ndata['label']
        g.ndata['train_mask'] = graph.ndata['train_mask']
        g.ndata['val_mask'] = graph.ndata['val_mask']
        g.ndata['test_mask'] = graph.ndata['test_mask']

        graph = g

    graph = graph.add_self_loop()

    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # retrieve the number of classes
    # n_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.ndata.pop('label').to(device).long()
    if args.mode < -50 and args.mode >= -60:
        labels = torch.LongTensor(np.load("../dataset/{}/label_mask_r{}.npy".format(args.dataset, -args.mode))).to(torch.device('cuda'))

    # Extract node features
    feats = graph.ndata.pop('feat').to(device)
    n_features = feats.shape[-1]

    # retrieve masks for train/validation/test
    if args.dataset == 'cora' or args.dataset == 'citeseer':
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)
        n_classes = dataset.num_classes

        if args.mode < 0 and args.mode >= -20:
            train_idx = torch.tensor(np.load("../dataset/{}/train_mask_r{}.npy".format(args.dataset, -args.mode))).to(device)
            val_idx = torch.tensor(np.load("../dataset/{}/val_mask_r{}.npy".format(args.dataset, -args.mode))).to(device)
            test_idx = torch.tensor(np.load("../dataset/{}/test_mask_r{}.npy".format(args.dataset, -args.mode))).to(device)
    else:
        train_idx = torch.tensor(np.load("../dataset/{}/train_mask.npy".format(args.dataset))).to(device)
        val_idx = torch.tensor(np.load("../dataset/{}/val_mask.npy".format(args.dataset))).to(device)
        test_idx = torch.tensor(np.load("../dataset/{}/test_mask.npy".format(args.dataset))).to(device)
        n_classes = int(labels.max().item() + 1)

    graph = graph.to(device)

    # Step 2: Create model =================================================================== #
    model = DAGNN(k=args.k,
                  in_dim=n_features,
                  hid_dim=args.hid_dim,
                  out_dim=n_classes,
                  dropout=args.dropout)
    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = F.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamb)

    # Step 4: training epochs =============================================================== #
    loss = float('inf')
    best_acc = 0
    no_improvement = 0
    epochs = trange(args.epochs, desc='Accuracy & Loss')

    for _ in epochs:
        model.train()

        logits = model(graph, feats)

        # compute loss
        train_loss = loss_fn(logits[train_idx], labels[train_idx])

        # backward
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = evaluate(model, graph, feats, labels,
                                                                                     (train_idx, val_idx, test_idx))

        # Print out performance
        epochs.set_description('Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}'.format(
            train_acc, train_loss.item(), valid_acc, valid_loss.item()))

        if valid_loss > loss:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print('Early stop.')
                break
        else:
            no_improvement = 0
            loss = valid_loss
            best_acc = test_acc

    print("Test Acc {:.4f}".format(best_acc))

    time_start = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(graph, feats)
    time_end = time.time()

    return best_acc, time_end - time_start


if __name__ == "__main__":
    """
        DAGNN Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='DAGNN')
    # data source params
    parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset.')
    parser.add_argument("--model_name", type=str, default="dagnn")
    # cuda params
    parser.add_argument("--mode", type=int, default=0, help="label mode")
    parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: -1, using CPU.')
    # training params
    parser.add_argument('--runs', type=int, default=1, help='Training runs.')
    parser.add_argument('--epochs', type=int, default=1500, help='Training epochs.')
    parser.add_argument('--early-stopping', type=int, default=100, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=0.005, help='L2 reg.')
    # model params
    parser.add_argument('--k', type=int, default=12, help='Number of propagation layers.')
    parser.add_argument("--hid-dim", type=int, default=64, help='Hidden layer dimensionalities.')
    parser.add_argument('--dropout', type=float, default=0.8, help='dropout')
    args = parser.parse_args()
    print(args)

    if args.dataset == 'citeseer':
        args.lamb = 0.02
        args.k = 16
    acc_lists = []
    time_lists = []
    random_seeds = generate_random_seeds(seed=1222, nums=args.runs)

    for run in range(args.runs):
        set_random_state(random_seeds[run])
        acc, infer_time = main(args)
        acc_lists.append(acc)
        time_lists.append(infer_time)

    acc_lists = np.array(acc_lists)

    mean = np.around(np.mean(acc_lists, axis=0), decimals=4)
    std = np.around(np.std(acc_lists, axis=0), decimals=4)

    outFile = open('PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in args.__dict__.items():
        results.append(k)
    
    results.append(str(acc_lists))
    results.append(str(mean))
    results.append(str(std))
    results.append(str(np.array(time_lists)))
    results.append(str(np.mean(np.array(time_lists))))
    writer.writerow(results)

    print('Total acc: ', acc_lists)
    print('mean', mean)
    print('std', std)