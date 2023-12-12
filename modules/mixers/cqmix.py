import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class CausalMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(CausalMixer, self).__init__()

        self.args = args
        self.k = args.k
        self.n_agents = args.n_agents
        self.n_variables = args.n_variables
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")

        # hyper w0 b0
        self.hyper_w0_0 = nn.Sequential(nn.Linear(self.input_dim+self.n_variables, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.k))
        self.hyper_b0_0 = nn.Sequential(nn.Linear(self.input_dim+self.n_variables, 1))
        
        # self.hyper_w0_1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(args.hypernet_embed, self.n_agents))
        # self.hyper_b0_1 = nn.Sequential(nn.Linear(self.input_dim, 1))

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, (self.n_variables) * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, causal_relations, states, random=False):
        # causal_relations: b, t, 2, n_variables
        causal_relations = causal_relations.permute(0, 1, 3, 2)
        if qvals.shape[1] == causal_relations.shape[1]:
            causal_relations = causal_relations.unsqueeze(2)
        else:
            causal_relations = th.cat((causal_relations, causal_relations[:, -1].unsqueeze(1)), dim=1).unsqueeze(2)
        # reshape
        b, t, _ = qvals.size()
        # print(qvals.shape, causal_relations.shape) 
        # qvals = qvals.reshape(b * t, 1, self.n_agents)
        # states = states.reshape(-1, self.state_dim)

        group_qvals = []
        for variable in range(self.n_variables):
            qval_vars = [th.gather(qvals, dim=-1, index=causal_relations[:, :, :, variable, i]) for i in range(self.k)]
            # qval_v_0 = th.gather(qvals, dim=-1, index=causal_relations[:, :, :, variable, 0]) # b, t, 1
            # qval_v_1 = th.gather(qvals, dim=-1, index=causal_relations[:, :, :, variable, 1]) # b, t, 1
            # qval_v_2 = th.gather(qvals, dim=-1, index=causal_relations[:, :, :, variable, 2]) # b, t, 1
            group_qval = th.cat(qval_vars, dim=-1).reshape(b*t, 1, -1) # b*t, 1, 2
            var_onehot = th.zeros(self.n_variables).unsqueeze(0).unsqueeze(0).repeat(b, t, 1).to(group_qval.device)
            var_onehot[variable] = 1
            w0_0 = self.hyper_w0_0(th.cat((states, var_onehot), dim=-1)).view(-1, self.k, 1)
            if self.abs:
                w0_0 = self.pos_func(w0_0)
            b0_0 = self.hyper_b0_0(th.cat((states, var_onehot), dim=-1)).view(-1, 1, 1)
            group_qval = th.matmul(group_qval, w0_0)+b0_0 # b*t, 1, 1
            group_qvals.append(group_qval.reshape(b, t, 1))
            # group_qval: b, t, 1
        # other_val = th.sum(qvals, dim=-1).unsqueeze(-1)
        w0_1 = self.hyper_w0_1(states).view(-1, self.n_agents, 1)
        b0_1 = self.hyper_b0_1(states).view(-1, 1, 1)
        
        other_val = th.matmul(qvals.reshape(b*t, 1, self.n_agents), w0_1) + b0_1
        group_qvals.append(other_val.reshape(b, t, 1))
        
        group_qvals = th.stack(group_qvals, dim=-1) # b, t, m+1
        group_qvals = group_qvals.reshape(b*t, 1, -1)
        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_variables+1, self.embed_dim) # b * t, n_variables+1, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
            
        # Forward
        hidden = F.elu(th.matmul(group_qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return y.view(b, t, -1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
        

        
