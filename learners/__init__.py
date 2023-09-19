from .ppo_learner import PPOLearner
from .nq_learner import NQLearner
from .LA_SMAC import LA_SMAC_Learner
from .LA_SMAC_PPO import LA_SMAC_PPO
from .NCC_SMAC import NCC_SMAC_Learner
from .CNCC_SMAC import CNCC_SMAC_Learner
REGISTRY = {}


REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["LA_SMAC"] = LA_SMAC_Learner
REGISTRY["LA_SMAC_PPO"] = LA_SMAC_PPO
REGISTRY["NCC_SMAC"] = NCC_SMAC_Learner
REGISTRY["CNCC_SMAC"] = CNCC_SMAC_Learner