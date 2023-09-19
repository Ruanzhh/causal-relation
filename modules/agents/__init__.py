REGISTRY = {}

from .n_rnn_agent import NRNNAgent
from .n_ncc_agent import NNCCAgent

REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["n_ncc"] = NNCCAgent

