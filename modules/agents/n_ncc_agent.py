import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from torch.cuda.amp import autocast

class NNCCAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NNCCAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        self.fc_cog_mean = nn.Linear(args.rnn_hidden_dim, args.cog_hidden_dim) 
        self.fc_cog_logstd = nn.Linear(args.rnn_hidden_dim, args.cog_hidden_dim)
        
        self.fc_decoder1 = nn.Linear(args.cog_hidden_dim, args.rnn_hidden_dim)
        self.fc_decoder2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_decoder3 = nn.Linear(args.rnn_hidden_dim, args.obs_dim)
        
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.cog_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def _cognition_module(self, hidden_state):
        c_mean = self.fc_cog_mean(hidden_state)
        c_logstd = self.fc_cog_logstd(hidden_state)
        c_hat = c_mean + torch.exp(c_logstd) * torch.normal(mean=0.0, std=1.0, size=c_logstd.shape).to(c_mean.device)
        return c_hat

    def _decoder(self, hidden_state):
        obs_hat = F.relu(self.fc_decoder1(hidden_state))
        obs_hat = F.relu(self.fc_decoder2(obs_hat))
        obs_hat = self.fc_decoder3(obs_hat)
        return obs_hat

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        c_hat = self._cognition_module(hh)
        obs_hat = self._decoder(c_hat)

        # TODO: check the size
        q_input = torch.cat([hh, c_hat], dim=1)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(q_input))
        else:
            q = self.fc2(q_input)

        return q.view(b, a, -1), c_hat.view(b, a, -1), obs_hat.view(b, a, -1), hh.view(b, a, -1)