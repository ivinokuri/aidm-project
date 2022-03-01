#!/usr/bin/env python3

import gym
from gym import spaces
import numpy as np

import DPlanCallback as call

class DPlanEnv(gym.Env):

    def __init__(self, data):
        print("Init Env")
        self.num_samples = 1000
        self.normal = 0
        self.anomaly = 1
        self.prob = 0.5

        self.samples = data.shape[0]
        self.features = data.shape[1] - 1 # last column are labels
        self.x = data[:, :self.features]
        self.y = data[:, self.features]
        self.data = data
        self.iu = np.where(self.y == self.normal)[0]
        self.ia = np.where(self.y == self.anomaly)[0]

        self.observation_space = spaces.Discrete(self.samples)
        self.action_space = spaces.Discrete(2)

        self.counts = None
        self.state = None
        self.DQN = None

    def sample_anomaly(self):
        return np.random.choice(self.ia)

    def sample_unlabeled(self, action, state):
        sample = np.random.choice(self.iu, self.num_samples)
        xs = self.x[np.append(sample, state)]
        dqns = call.penalti_output(xs, self.DQN)
        dqn = dqns[:-1]
        dqn_state = dqns[-1]

        distances = np.linalg.norm(dqn - dqn_state, axis=1)

        if action == 1:
            location = np.argmin(distances)
        elif action == 0:
            location = np.argmax(distances)
        return sample[location]

    def step(self, action):
        state = self.state
        generators = np.random.choice([self.sample_anomaly, self.sample_unlabeled])
        new_state = generators(action, state)

        self.state = new_state
        self.counts += 1

        reward = self.reward(action, state)
        done = False
        info = {"State t": state, "Action t": action, "State t+1": new_state}

        return self.state, reward, done, info

    def reward(self, action, state):
        if action == 1 and state in self.ia:
            return 1
        elif action == 0 and state in self.iu:
            return 0
        return  -1

    def reset(self):
        self.counts = 0
        self.state = np.random.choice(self.iu)
        return self.state
