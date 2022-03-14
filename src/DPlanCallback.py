#!/usr/bin/env python3

from rl.callbacks import Callback
import keras.backend as K
from sklearn.ensemble import IsolationForest


class DPlanCallback(Callback):
    """
    Keras callback for training process of the agent
    """
    def __init__(self):
        super(DPlanCallback, self).__init__()
        self.isolationForest = IsolationForest()

    def on_action_begin(self, action, logs={}):
        self.env.DQN = self.model.model

    def on_train_begin(self, logs=None):
        """
        Updating intrinsic reward by isolation forest scores at the beginning of the training
        """
        self.model.processor.reward = self.calc_isolation_forest_scores()

    def on_episode_end(self, episode, logs={}):
        """
        Updating internal reward by isolation forest scores at the end of the episode
        """
        self.model.processor.reward = self.calc_isolation_forest_scores()

    def calc_isolation_forest_scores(self):
        """
        Calculating scores of the current model state with trained weights of the network
        The score will define intrinsic reward
        :return: normalized scores
        """
        x = self.env.x
        model = self.model.model
        output = self.penalti_output(x, model)
        fitted_iforest = self.isolationForest.fit(output)
        scores = fitted_iforest.score_samples(output)
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

        return norm_scores

    @staticmethod
    def penalti_output(x, model):
        """
        Penalti calculation for model output and original input
        :param x: input
        :param model: dqn model
        :return: penalti results
        """
        penalti_function = K.function([model.input], [model.layers[-2].output])
        output = penalti_function(x)[0]
        return output
