import numpy as np
import argparse
import os
import os.path as osp
import random
import nni
import time

import torch
from torch._C import wait
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pHNGCL.model import Encoder, HNGCL
from pHNGCL.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pHNGCL.eval import log_regression, MulticlassEvaluator
from pHNGCL.utils import common_loss, generate_feature_graph_edge_index, get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality, loss_dependence
from pHNGCL.dataset import get_dataset

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def train(model, x, edge_index, feature_graph_edge_index):
    model.train()
    optimizer.zero_grad()

    # topology contrastive graphs
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)

    # feature contrastive graphs
    edge_index_2 = dropout_adj(feature_graph_edge_index, p=drop_edge_rate_2)[0]
    x_2 = drop_feature(x, drop_feature_rate_2)

    # # topology contrastive graphs
    # edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    # x_2 = drop_feature(x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss_neg(z1, z2, batch_size=256)
    # loss = model.loss(z1, z2, batch_size=256)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc

def save_embedding():
    model.eval()
    z = model(data.x, data.edge_index)
    z = z.detach().cpu().numpy()
    path = osp.expanduser('~/HNGCL-Experiment/result')
    embedding_path = osp.join(path, "visulization", "embeddings", args.dataset)
    file_name = osp.join(embedding_path, args.dataset.lower() + "_" + str(param['k']) + "nn")
    check_dir(file_name)
    np.save(file_name, z)

def save_labels(labels):
    path = osp.expanduser('~/HNGCL-Experiment/result')
    labels_path = osp.join(path, "visulization", "labels", args.dataset.lower())
    check_dir(labels_path)
    np.save(labels_path, labels)

def plot_embedding(labels):
    path = osp.expanduser('~/HNGCL-Experiment/result')
    embedding_path = osp.join(path, "visulization", "embeddings", args.dataset)
    figure_path = osp.join(path, "visulization", "figures", args.dataset)
    check_dir(embedding_path)
    check_dir(figure_path)

    embeddings = np.load(osp.join(embedding_path, args.dataset.lower() + "_" + str(param['k']) + "nn.npy"))
    tsne = TSNE(init='pca', random_state=0)

    tsne_features = tsne.fit_transform(embeddings)

    xs = tsne_features[:, 0]
    ys = tsne_features[:, 1]

    plt.scatter(xs, ys, c = labels)
    figure_name = osp.join(figure_path, args.dataset.lower() + "_" + str(param['k']) + "nn.pdf")
    check_dir(figure_name)
    plt.savefig(figure_name)

def check_dir(file_name=None):
    dir_name = osp.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def record_hyper_parameter(result_file, param):
    fb = open(result_file, 'a+', encoding='utf-8')
    fb.write('\n'*5)
    fb.write('-'*30 + ' ' * 5 + 'Hyper parameters in training' + ' ' * 5 + '-'*30 + '\n\n')
    fb.write("total training epoches: {}\n".format(param['num_epochs']))
    fb.write("learning rate: {}\n".format(param['learning_rate']))
    fb.write("hidden num: {}\n".format(param['num_hidden']))
    fb.write("projection hidden num: {}\n".format(param['num_proj_hidden']))
    fb.write("activation function: {}\n".format(param['activation']))
    fb.write("drop edge rate1: {}\n".format(param['drop_edge_rate_1']))
    fb.write("drop edge rate2: {}\n".format(param['drop_edge_rate_2']))
    fb.write("drop feature rate1: {}\n".format(param['drop_feature_rate_1']))
    fb.write("drop feature rate2: {}\n".format(param['drop_feature_rate_2']))
    fb.write("temperature coefficient tau: {}\n".format(param['tau']))
    fb.write("alpha: {}\n".format(param['alpha']))
    fb.write('\n' + '-'*30 + ' ' * 5 + 'Hyper parameters in training' + ' ' * 5 + '-'*30 + '\n')
    fb.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--param', type=str, default='local:cora.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('-alpha', type=float, default=0.3)
    parser.add_argument('-beta', type=float, default=0.2)
    parser.add_argument('-gamma', type=float, default=0.2)
    parser.add_argument('-theta', type=float, default=0.3)
    parser.add_argument('-patience', type=int, default=100)
    default_param = {
        #学习率
        'learning_rate': 0.001,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.1,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree'
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)

    feature_graph_edge_index = generate_feature_graph_edge_index(data.x, param['k']).to(device)

    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)
    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = HNGCL(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau'], param['alpha']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    drop_edge_rate_1 = param['drop_edge_rate_1']
    drop_edge_rate_2 = param['drop_edge_rate_2']
    drop_feature_rate_1 = param['drop_feature_rate_1']
    drop_feature_rate_2 = param['drop_feature_rate_2']

    log = args.verbose.split(',')

    for epoch in range(1, param['num_epochs'] + 1):
        time_start = time.time()
        loss = train(model, data.x, data.edge_index, feature_graph_edge_index)
        time_end = time.time()
        time_c= time_end - time_start
        print('time cost', time_c, 's')
        if 'train' in log:
            print(f'(T) | Epoch={epoch:04d}, loss={loss:.4f}')

    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')