#!/usr/bin/env python3

from rl.core import Processor


class DPLANProcessor(Processor):
    """
    Agent step override in the rl environment
    """
    def __init__(self, env):
        """
        Init
        :param env: simulation environment
        """
        self.x = env.x
        self.reward = None
        self.last_observation = None

    def process_step(self, observation, reward, done, info):
        """
        Override of environment step, it will save last observation and add intrinsic reward
        :param observation:
        :param reward:
        :param done:
        :param info:
        :return: (observation, reward, done, info)
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        self.last_observation = observation
        info = self.process_info(info)

        return observation, reward, done, info

    def process_observation(self, observation):
        """
        Override process observation step, it will take data point for specific observation
        :param observation:
        :return: data point
        """
        return self.x[observation, :]

    def process_reward(self, step_reward):
        """
        Override process reward function by customizing reward with intrinsic reward
        :param step_reward:
        :return:
        """
        r = self.reward[self.last_observation]
        return step_reward + r
