# -*- coding: utf-8 -*-
import pickle
from util import Data
from models.base_models import *
import os, sys
from config import parser
import torch
import datetime
import numpy as np
from utils.helper import *
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def train(args):

    '''
    load dataset
    '''
    base_url = sys.path[0]
    print(base_url)
    train_data = pickle.load(open('./data/' + args.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./data/' + args.dataset + '/test.txt', 'rb'))
    seq_data = pickle.load(open('./data/' + args.dataset + '/all_train_seq.txt', 'rb'))

    if args.dataset == 'diginetica':
        args.n_node = 43097
    elif args.dataset == 'Tmall':
        args.n_node = 40728
    elif args.dataset == 'Nowplaying':
        args.n_node = 60417
    else:
        print("dataset notfound")
        return

    train_data = Data(train_data, shuffle=True, n_node=args.n_node,graph_data=seq_data)
    test_data = Data(test_data, shuffle=True, n_node=args.n_node)
    '''
    model building and train
    '''
    model=MIHRN(args).cuda()
    top_K = [10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]
    start_time = time.time()
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        epo_stime = time.time()
        metrics, total_loss = train_test(model, train_data, test_data) # model train and test
        epo_entime = time.time()
        print(f'EPOCH time: {epo_entime - epo_stime}')

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


    end_time = time.time()
    print(f'Training time: {end_time - start_time}')

def forward(model, i, data):
    tar, reversed_sess_item, pos_id = data.get_slice(i)
    tar = trans_to_cuda(tar)
    pos_id = trans_to_cuda(pos_id)
    reversed_sess_item = trans_to_cuda(reversed_sess_item)

    if model.training is True:
        scores,loss=model(data.adjacency,reversed_sess_item,pos_id,tar)
        return tar, scores, loss
    else:
        scores = model(data.adjacency,reversed_sess_item, pos_id)
        return tar, scores


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    model.train()
    count=0
    for i in slices:
        _, scores, loss = forward(model, i, train_data)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        count+=1

    print('\tLoss:\t%.3f' % (total_loss / len(slices)))
    top_K = [10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(1 * model.batch_size)
    for i in slices:
        tar, scores = forward(model, i, test_data)
        index = trans_to_cpu(scores.topk(max(top_K))[1])
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)





