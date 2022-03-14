
import numpy as np
import QNetwork as qn
import DPlanProcessor as proc
import DPlanCallback as call
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from keras.optimizers import adam_v2

class DPlan:
    """
    Complete implementation of DPlan by the instructions of authors of the article.
    We are using keras-rl library or policy, reply buffer and agent.
    """
    def __init__(self, parsed_args, env, features):
        """
        Init DPlan
        :param parsed_args: runtime parameters
        :param env: simulator environment
        :param features: number of features in the data
        """
        print("Init Dplan")
        self.env = env
        self.train_env = None
        self.features = features
        self.memory_size = parsed_args['memory_size']
        self.hidden_size = parsed_args['hidden_size']
        self.batch_size = parsed_args['batch_size']
        self.max_epsilon = parsed_args['max_epsilon']
        self.min_epsilon = parsed_args['min_epsilon']
        self.greedy_course = parsed_args['epsilon_course']
        self.epochs = parsed_args['epochs']
        self.steps = parsed_args['steps']
        self.lr = parsed_args['lr']
        self.model = qn.QNetwork(input_size=self.features,
                              n_actions=env.action_space.n,
                              hidden_size=self.hidden_size)
        self.policy = LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                      attr='eps',
                                      value_max=self.max_epsilon,
                                      value_min=self.min_epsilon,
                                      value_test=0.,
                                      nb_steps=self.greedy_course)
        self.memory = SequentialMemory(limit=self.memory_size,
                                  window_length=1)
        self.optimizer = adam_v2.Adam(learning_rate=self.lr)
        self.processor = proc.DPLANProcessor(self.env)
        self.agent = DQNAgent(model=self.model,
                              policy=self.policy,
                              nb_actions=self.env.action_space.n,
                              memory=self.memory,
                              processor=self.processor,
                              batch_size=self.batch_size)
        self.agent.compile(optimizer=self.optimizer)
        self.weights = self.agent.model.get_weights()
        self.agent.target_model.set_weights(np.zeros(self.weights.shape))

    def fit(self, weights_file=None):
        """
        Training fit function of the method
        :param weights_file: preloaded weights file
        """
        self.train_env = self.env
        callback = call.DPlanCallback()
        self.agent.fit(env=self.train_env,
                       nb_steps=self.epochs,
                       action_repetition=1,
                       callbacks=[callback],
                       nb_max_episode_steps=self.steps)
        if weights_file:
            self.agent.save_weights(weights_file, overwrite=True)

    def load_weights(self, weights_file):
        """
        Loading trained weights of the network
        :param weights_file: path to the file
        """
        self.agent.load_weights(weights_file)

    def predict(self, input):
        """
        Return anomaly score of input by trained weights
        :param input: input data
        :return: scores of output anomaly or normal
        """
        q_values = self.agent.model.predict(input[:, np.newaxis, :])
        scores = q_values[:, 1]
        return scores

    def predict_label(self, input):
        """
        Return labels of the input data
        :param input: input data
        :return: labels
        """
        q_values = self.agent.model.predict(input[:, np.newaxis, :])
        labels = np.argmax(q_values, axis=1)

        return labels
