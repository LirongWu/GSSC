import dgl
import torch
import pickle
import numpy as np


def dataloader(param, device, mode):
    
    if param['dataset'] == 'cora':
        graph = dgl.data.CoraGraphDataset()[0]
    elif param['dataset'] == 'citeseer':
        graph = dgl.data.CiteseerGraphDataset()[0]
    elif param['dataset'] == 'coauthor-cs':
        graph = dgl.data.CoauthorCSDataset()[0]
    elif param['dataset'] == 'coauthor-phy':
        graph = dgl.data.CoauthorPhysicsDataset()[0]     
    elif param['dataset'] == 'film':
        adj = pickle.load(open('../dataset/film/{}_adj.pkl'.format(param['dataset']), 'rb'))
        row, col = np.where(adj.todense() > 0)
        U = row.tolist()
        V = col.tolist()
        graph = dgl.graph((U, V))
        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)
        graph.ndata['feat'] = torch.FloatTensor(pickle.load(open('../dataset/film/{}_features.pkl'.format(param['dataset']), 'rb')))
        graph.ndata['label'] = torch.LongTensor(pickle.load(open('../dataset/film/{}_labels.pkl'.format(param['dataset']), 'rb'))) 

    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)

    if param['dataset'] == 'cora' or param['dataset'] == 'citeseer':
        train_mask = graph.ndata['train_mask'].to(device)
        val_mask = graph.ndata['val_mask'].to(device)
        test_mask = graph.ndata['test_mask'].to(device)
    else:
        train_index = torch.tensor(np.load("../dataset/{}/train_mask.npy".format(param['dataset']))).to(device)
        val_index = torch.tensor(np.load("../dataset/{}/val_mask.npy".format(param['dataset']))).to(device)
        test_index = torch.tensor(np.load("../dataset/{}/test_mask.npy".format(param['dataset']))).to(device)
        train_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        val_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        test_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

    # Training with Limited Labeled Data
    if mode < 0 and mode >= -20:
        train_index = torch.tensor(np.load("../dataset/{}/train_mask_r{}.npy".format(param['dataset'], -mode))).to(device)
        val_index = torch.tensor(np.load("../dataset/{}/val_mask_r{}.npy".format(param['dataset'], -mode))).to(device)
        test_index = torch.tensor(np.load("../dataset/{}/test_mask_r{}.npy".format(param['dataset'], -mode))).to(device)
        train_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        val_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        test_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

    # Training with Structure Perturbation
    if mode < -20 and mode >= -50:
        src = np.load("../dataset/{}/src_mask_r{}.npy".format(param['dataset'], -mode))
        dst = np.load("../dataset/{}/dst_mask_r{}.npy".format(param['dataset'], -mode))
        U = src.tolist()
        V = dst.tolist()
        graph = dgl.graph((U, V))
        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)

    # Training with Label Noise
    if mode < -50 and mode >= -60:
        labels = torch.LongTensor(np.load("../dataset/{}/label_mask_r{}.npy".format(param['dataset'], -mode))).to(device)

    return graph, features, labels, train_mask, val_mask, test_mask