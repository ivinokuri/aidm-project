#!/usr/bin/env python3

from rl.callbacks import Callback
import keras.backend as K
from sklearn.ensemble import IsolationForest


class DPlanCallback(Callback):

    def __init__(self):
        super(DPlanCallback, self).__init__()
        self.iforest = IsolationForest()

    def on_action_begin(self, action, logs={}):
        self.env.DQN = self.model.model

    def on_train_begin(self, logs=None):
        """
        Updating intrinsic reward by iforest scores at the beginning of the training
        """
        self.model.processor.reward = self.calc_iforest_scores()

    def on_episode_end(self, episode, logs={}):
        """
            Updating internal reward by iforest scores at the end of the episode
        """
        self.model.processor.reward = self.calc_iforest_scores()

    def calc_iforest_scores(self):
        """
        Calculating scores of the current model state with trained weights of the network
        The score will define intrinsic reward
        :return:
        """
        x = self.env.x
        model = self.model.model
        output = self.penalti_output(x, model)
        fitted_iforest = self.iforest.fit(output)
        scores = fitted_iforest.score_samples(output)
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

        return norm_scores

    @staticmethod
    def penalti_output(x, model):
        penalti_function = K.function([model.input], [model.layers[-2].output])
        output = penalti_function(x)[0]
        return output
