# -*- coding: utf-8 -*-
import datetime
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(nn.Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if self.dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)

        item_embeddings = sum(final, 0)
        return item_embeddings

class DualAttention(nn.Module):
    def __init__(self,adjacency, n_items, n_pos, wk, beta, temperature,layers,emb_size=100,
                dropout=0.2, activate='relu', dataset='Tmall', batch_size=512, lr=0.001):
        super(DualAttention, self).__init__()
        self.lr = lr
        self.emb_size=emb_size
        self.dim =2*emb_size
        self.n_items = n_items
        self.adjacency = adjacency
        self.layers = layers

        self.embedding = nn.Embedding(n_items, emb_size, padding_idx=0,max_norm=1.5)
        self.pos_embedding = nn.Embedding(n_pos, emb_size, padding_idx=0, max_norm=1.5)
        self.HyperGraph = HyperConv(self.layers, dataset)
        self.atten_w0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))
        self.w_f = nn.Linear(2*self.dim, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.self_atten_w1 = nn.Linear(self.dim, self.dim)
        self.self_atten_w2 = nn.Linear(self.dim, self.dim)
        self.batch_size = batch_size
        self.LN = nn.LayerNorm(self.dim)
        self.LN2 = nn.LayerNorm(self.dim)
        self.attention_mlp = nn.Linear(self.dim, self.dim)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.w = wk
        self.crietion = nn.CrossEntropyLoss()
        if activate == 'relu':
            self.activate = F.leaky_relu
        elif activate == 'selu':
            self.activate = F.selu
        self.num_heads = 10
        self.w0 = nn.Linear(self.dim, self.dim, bias=False)
        self.reduce = nn.Linear(self.dim * self.num_heads , self.dim)
        self.K = nn.Linear(self.dim, self.dim)
        self.V = nn.Linear(self.dim, self.dim)
        self.v2 = nn.Linear(self.dim, self.dim)
        self.beta = beta
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0, amsgrad=True)
        self.initial_()

    def initial_(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, pos, target=None):
        x_embeddings = self.embedding(x)  # B,seq,item_dim

        pos_embeddings = self.pos_embedding(pos)  # B, seq, pos_dim
        mask = (x > 0)  # B,seq
        x_ = torch.cat([x_embeddings, pos_embeddings], 2) # B, seq, dim(item_dim+pos_dim)
        h_t, d_i = self.DIM(x_, mask)
        result = self.predict(h_t, d_i) 

        if target is not None:
            mirror_h_t, _ = self.DIM(x_, mask)
            cl_loss = self.CL_Loss(h_t, mirror_h_t)
            loss = self.crietion(result, target)
            return result, loss + self.beta * cl_loss
        return result
    

    def DIM(self, x, mask=None):
        if mask is not None:
            full_mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1) # B,seq,seq
        m_s, x_n,x_d = self.self_attention(x, full_mask)
        global_c = self.global_attention(m_s, x_n, x, mask)
        return global_c,x_d


    def CL_Loss(self, session_embeddings, mirror_embeddings):
        session_embeddings, mirror_embeddings = session_embeddings.squeeze(1), mirror_embeddings.squeeze(1)
        batch_size = session_embeddings.shape[0]
        device = session_embeddings.device
        y_true = torch.cat([torch.arange(batch_size, 2*batch_size, dtype=torch.long), 
                        torch.arange(batch_size, dtype=torch.long)], 0).to(device)
        batch_emb = torch.cat([session_embeddings, mirror_embeddings], 0)
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        sim_score = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
        sim_score = sim_score / self.temperature
        sim_score = sim_score.fill_diagonal_(-np.inf)
        CL_loss = self.crietion(sim_score, y_true)
        return CL_loss

    def self_attention(self, sequence, mask=None):
        '''
        seqence: B, seq, dim
        mask: B, seq,seq
        '''
        input_shape = sequence.shape # B, Seq,dim
        device = sequence.device
        head_dim = input_shape[-1] // self.num_heads  # dim/n_head
        Q = self.attention_mlp(sequence) # B, seq, dim
        Q = Q.view(input_shape[0], -1, self.num_heads, head_dim).transpose(1, 2) # B,num_head，seq,head_dim，
        K = self.K(sequence).view(input_shape[0], -1, self.num_heads, head_dim).transpose(1, 2) # B num_head seq head_dim
        C = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(input_shape[-1] / self.num_heads) # B num_heads seq seq
        V = self.V(sequence).view(input_shape[0], -1, self.num_heads, head_dim).transpose(1, 2) # B num_head seq head_dim
        len_vec = torch.tensor(input_shape[1])
        full_mask = mask
        if mask is not None:
            mask = ~mask
            mask1 = (torch.eye(input_shape[1])==1).unsqueeze(0).repeat(input_shape[0], 1, 1).to(device) # (B, seq, seq)
            mask1 = mask | mask.transpose(-2, -1) | mask1
            mask1 = mask1.unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (B, num_head, seq, seq)
            C_ = C.masked_fill(mask1, 0)
            len_vec = torch.sum(~mask[:, 0], dim=-1).unsqueeze(-1) # batch, 1, 1
        alpha = torch.sum(C_, dim=-1).true_divide(len_vec.unsqueeze(-1)) # batch num_heads seq 
        if mask is not None:
            mask = mask[:,0] # batch 1
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1) # batch num_heads seq 
            alpha = alpha.masked_fill(mask, -np.inf)
        beta = F.softmax(alpha, dim=-1) # batch num_heads seq  batch seq dim
        out = torch.matmul(beta, sequence).view(input_shape[0], 1, -1) # batch num_heads dim -> batch  num_head * dim

        if full_mask is not None:
            scores = C.masked_fill(full_mask.unsqueeze(1) == 0, -np.inf)
        a = torch.softmax(scores, dim=-1)
        x_n = torch.matmul(a, V).transpose(1, 2).contiguous().view(input_shape)  #H
        x_n = self.dropout(self.w0(x_n)) + sequence #H~
        x_n = torch.cat([self.reduce(out), x_n], dim=1) # H~||out
        x_n = self.dropout(self.activate(self.self_atten_w2(self.activate(self.self_atten_w1(x_n))))) + x_n
        return x_n[:,0, :].unsqueeze(1), x_n[:, 1:, :],self.reduce(out)
    
    
    def global_attention(self, target, k, v, mask=None):
        alpha = torch.matmul(
            self.activate(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),
            self.atten_w0)  # (B,seq,1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = torch.softmax(alpha, dim=1)

        c = torch.matmul(alpha.transpose(1, 2), self.v2(v))  # (B, 1, dim)
        return c

    def predict(self, global_c, self_c):
        c = self.dropout(torch.selu(self.w_f(torch.cat((global_c, self_c), 2))))
        c = c.squeeze()
        l_c = (c/torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)
        z = self.w * torch.matmul(l_c, l_emb.T)

        return z


def forward(model, i, data):
    tar, reversed_sess_item, pos_id = data.get_slice(i)
    tar = trans_to_cuda(tar)
    pos_id = trans_to_cuda(pos_id)
    reversed_sess_item = trans_to_cuda(reversed_sess_item)
    if model.training is True:
        scores, loss = model(reversed_sess_item, pos_id, tar)
        return tar, scores, loss
    else:
        scores = model(reversed_sess_item, pos_id)
        return tar, scores
    

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    model.train()
    for i in slices:
        _, scores, loss = forward(model, i, train_data)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        total_loss += loss
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
                metrics['hit%d' %K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' %K].append(0)
                else:
                    metrics['mrr%d' %K].append(1 / (np.where(prediction == target)[0][0]+1))
    return metrics, total_loss
