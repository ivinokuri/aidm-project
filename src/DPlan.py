
from torch import nn
import torch

from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

AVAIL_GPUS = min(1, torch.cuda.device_count())

class DPlan:

    def __init__(self, parsed_args, env):
        print("Init Dplan")
        self.features = parsed_args['features']
        self.hidden_size = parsed_args['hidden_size']
        self.max_epsilon = parsed_args['max_epsilon']
        self.min_epsilon = parsed_args['min_epsilon']
        self.greedy_course = parsed_args['epsilon_course']

        self.model = QNetwork(input_size=self.features,
                              n_actions=env.action_space.n,
                              hidden_size=self.hidden_size)
        self.policy = LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                      attr='eps',
                                      value_max=self.max_epsilon,
                                      value_min=self.min_epsilon,
                                      value_test=0.,
                                      nb_steps=self.greedy_course)


class QNetwork(nn.Module):

    def __init__(self, input_size: int, n_actions: int=2, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())
