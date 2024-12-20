# -*- coding: utf-8 -*-
import manifolds
import models.encoders as encoders
from utils.helper import trans_to_cuda,default_device
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rgd.rsgd import RiemannianSGD

class MIHRN(nn.Module):
    def __init__(self, args):
        super(MIHRN, self).__init__()
        self.lr = args.lr
        self.emb_size = args.embedding_dim
        self.dim = 2 * self.emb_size
        self.n_items = args.n_node  #物品个数
        self.embedding = nn.Embedding(self.n_items, self.emb_size, padding_idx=0, max_norm=1.5).to(default_device()) #Ietm embedding
        self.pos_embedding = nn.Embedding(args.n_pos, self.emb_size, padding_idx=0, max_norm=1.5).to(default_device()) #Pos embedding
        self.atten_w0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))
        self.w_f = nn.Linear(2 * self.dim, self.emb_size)
        self.dropout = nn.Dropout(args.dropout)
        self.self_atten_w1 = nn.Linear(self.dim, self.dim)
        self.self_atten_w2 = nn.Linear(self.dim, self.dim)
        self.batch_size = args.batch_size
        self.LN = nn.LayerNorm(self.dim)
        self.LN2 = nn.LayerNorm(self.dim)
        self.attention_mlp = nn.Linear(self.dim, self.dim)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.w = args.wk
        self.crietion = nn.CrossEntropyLoss()
        if args.activate == 'relu':
            self.activate = F.leaky_relu
        elif args.activate == 'selu':
            self.activate = F.selu
        self.num_heads = args.n_heads
        self.w0 = nn.Linear(self.dim, self.dim, bias=False)
        self.reduce = nn.Linear(self.dim * self.num_heads, self.dim)
        self.K = nn.Linear(self.dim, self.dim)
        self.V = nn.Linear(self.dim, self.dim)
        self.v2 = nn.Linear(self.dim, self.dim)
        self.beta = args.beta
        self.temperature = args.temperature
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=args.weight_decay, amsgrad=True)
        self.initial_()

        self.c = torch.tensor([args.c]).to(default_device())
        self.manifold = getattr(manifolds, "Hyperboloid")()

        self.encoder = getattr(encoders, "HGCN")(self.c, args)


    def initial_(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def encode(self, adj):

        x=self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c)

        if torch.cuda.is_available():
            adj = adj.to(default_device())
            x = x.to(default_device())
        h = self.encoder.encode(x, adj)
        return h


    def forward(self, adj,x, pos, target=None):
        '''
        x: B, seq_len
        pos:B, seq_len
        '''
        '''双曲卷积操作'''

        item_embeddings_hg = self.encode(adj)

        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embeddings_hg], 0)
        item_embedding=nn.Embedding.from_pretrained(item_embedding)
        x_embeddings=item_embedding(x)

        pos_embeddings = self.pos_embedding(pos)  # B, seq, pos_dim
        mask = (x > 0) # B,seq
        x_ = torch.cat([x_embeddings, pos_embeddings], 2)  # B, seq, dim(item_dim+pos_dim)

        h_t, d_i = self.DIM(x_, mask)
        result = self.predict(h_t, d_i) 

        if target is not None:
            mirror_h_t, _ = self.DIM(x_, mask)
            cl_loss = self.CL_Loss(h_t, mirror_h_t)
            loss = self.crietion(result, target)
            return result, loss + self.beta * cl_loss
            return result,loss
        return result

    def DIM(self, x, mask=None):
        '''
        x: B,seq, dim
        '''
        if mask is not None:
            full_mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1)  # B,seq,seq
        m_s, x_n, x_d = self.self_attention(x, full_mask)
        global_c = self.global_attention(m_s, x_n, x, mask)
        return global_c, x_d

    def CL_Loss(self, session_embeddings, mirror_embeddings):
        session_embeddings, mirror_embeddings = session_embeddings.squeeze(1), mirror_embeddings.squeeze(1)
        batch_size = session_embeddings.shape[0]
        device = session_embeddings.device
        y_true = torch.cat([torch.arange(batch_size, 2 * batch_size, dtype=torch.long),
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
        input_shape = sequence.shape  # B, Seq,dim
        device = sequence.device
        head_dim = input_shape[-1] // self.num_heads 
        Q = self.attention_mlp(sequence)  # B, seq, dim
        Q = Q.view(input_shape[0], -1, self.num_heads, head_dim).transpose(1, 2)  # B,num_head，seq,head_dim，
        K = self.K(sequence).view(input_shape[0], -1, self.num_heads, head_dim).transpose(1,2)  # B num_head seq head_dim
        C = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(input_shape[-1] / self.num_heads)  # B num_heads seq seq
        self.C=C #便于记录保存
        V = self.V(sequence).view(input_shape[0], -1, self.num_heads, head_dim).transpose(1,2)  # B num_head seq head_dim
        len_vec = torch.tensor(input_shape[1])
        full_mask = mask  # size of mask is B,seq
        if mask is not None:
            mask = ~mask
            mask1 = (torch.eye(input_shape[1]) == 1).unsqueeze(0).repeat(input_shape[0], 1, 1).to(device)  # (B, seq, seq)
            mask1 = mask | mask.transpose(-2, -1) | mask1
            mask1 = mask1.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (B, num_head, seq, seq)
            C_ = C.masked_fill(mask1, 0)
            len_vec = torch.sum(~mask[:, 0], dim=-1).unsqueeze(-1)  # batch, 1, 1
        alpha = torch.sum(C_, dim=-1).true_divide(len_vec.unsqueeze(-1))  # batch num_heads seq
        if mask is not None:
            mask = mask[:, 0]  # batch 1
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)  # batch num_heads seq
            alpha = alpha.masked_fill(mask, -np.inf)
        beta = F.softmax(alpha, dim=-1)  # batch num_heads seq  batch seq dim
        # print(beta.shape,sequence.shape)
        out = torch.matmul(beta, sequence).view(input_shape[0], 1,-1)  # batch num_heads dim -> batch  num_head * dim  # x_d吧？

        if full_mask is not None:
            scores = C.masked_fill(full_mask.unsqueeze(1) == 0, -np.inf)
        a = torch.softmax(scores, dim=-1)
        x_n = torch.matmul(a, V).transpose(1, 2).contiguous().view(input_shape)  # H
        x_n = self.dropout(self.w0(x_n)) + sequence  # H~
        x_n = torch.cat([self.reduce(out), x_n], dim=1)  # H~||out
        x_n = self.dropout(self.activate(self.self_atten_w2(self.activate(self.self_atten_w1(x_n))))) + x_n
        return x_n[:, 0, :].unsqueeze(1), x_n[:, 1:, :], self.reduce(out)

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
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = self.embedding.weight / torch.norm(self.embedding.weight, dim=-1).unsqueeze(1)
        z = self.w * torch.matmul(l_c, l_emb.T)

        return z


