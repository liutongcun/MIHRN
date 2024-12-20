import argparse

from hgcn_utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'log': (None, 'None for no logging'),
        'lr': (0.0001, 'learning rate'),
        'batch_size': (512, 'batch size'),
        # 'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (15, 'maximum number of epochs to train for'),
        'weight_decay': (0, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'seed': (42, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
        'n_heads':(8, 'number of attention heads')
    },
    'model_config': {
        # 'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'embedding_dim': (100, 'user item embedding dimension'),
        'scale': (0.1, 'scale for init'),
        'dim': (100, 'embedding dimension'),
        'network': ('denseGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'), #denseGCN:19.0189;19.4617
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
        'num_layers': (3,  'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'beta':(1.0,'ssl task maginitude'),
        'dropout':(0.5,'dropout rate'),
        'temperature':(0.1,'the temperature of CL'),
        'wk':(20,'the normalized weight'),
        'activate':('relu','activate function'),
        'manifold': ('Hyperboloid', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        # 'use-att': (0, 'whether to use hyperbolic attention or not'),
        # 'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'dataset': ('Tmall', 'which dataset to use'), #Tmall，diginetica，Nowplaying
        # 'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
        'n_pos':(100,'number of position')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
