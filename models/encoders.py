import torch.nn as nn

import layers.hyp_layers as hyp_layers
import manifolds
import numpy as np

class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x) 
        return self.manifold.logmap0(output, c=self.c)


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        in_dim=args.embedding_dim
        out_dim=args.dim
        hgc_layers = []
        hgc_layers.append(hyp_layers.HyperbolicGraphConvolution(self.manifold, in_dim, out_dim, self.c, args.network,
                                                                args.num_layers))

        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
            x_hyp = self.manifold.proj(x, c=self.c)
            return super(HGCN, self).encode(x_hyp, adj)
