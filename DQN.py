import copy
import math
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env import Game2048Env

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.99

class Epsilon():
    def __init__(self):
        super(Epsilon, self).__init__()
        self.value = EPS_START

    def update(self):
        self.value = max(self.value * EPS_DECAY, EPS_END)

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, n_hidden, epsilon=Epsilon()):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, output_size)
        self.epsilon = epsilon

    def forward(self, state):
        out = self.act1(self.fc1(state))
        out = self.fc2(out)
        print(out)
        return out


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, labe="2", frame_on=False)
    ax.plot(x,epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", colors="C0")
    ax.tick_params(axis="y", colors="C0")
    N=len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0,t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xasis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")
    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def train_network(env, agent, n_episode, epsilon, dqn_parameters, trainId, verbose=False, polyak_avg=False):
    scores = []
    avg100Scores = []
    avgScores = []
    last100Scores = deque(maxlen=100)
    memory = deque(maxlen=dqn_parameters.mem_size)
    eps_history = []
    for episode in range(n_episode):
        total_reward_episode = 0
        if polyak_avg:
            agent.soft_copy_target(0.1)
        elif episode % dqn_parameters.target_update == 0:
            agent.copy_target()
        policy = agent.gen_epsilon_greedy_policy(epsilon.value,
        dqn_parameters.action_size)
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode += reward
            memory.append((state, action, next_state, reward, is_done))
            if is_done:
                scores.append(total_reward_episode)
                avgScores.append(np.mean(scores))
                last100Scores.append(total_reward_episode)
                avg100Scores.append(np.mean(last100Scores))
                break
            agent.replay(memory, dqn_parameters.replay_size,
                             dqn_parameters.gamma)
            state = next_state
        eps_history.append(epsilon)
        epsilon.update()
        average100Score = np.mean(scores[-min(100, len(scores)):])
        if verbose:
            print("episode: {}, score: {}, memory length: {}, epsilon: {}, average score: {}" \
            .format(episode, total_reward_episode,
                    len(memory), epsilon.value, average100Score))
    x = [i+1 for i in range(n_episode)]
    filename = "game2048.png"
    plot_learning_curve(x, scores, eps_history, filename)

    agent.save_model(trainId, dqn_parameters.data_path)

class DQNParameters():
    def __init__(self, action_size, lr=1e-3, mem_size=30000, replay_size=20, gamma=0.9, n_hidden=30, target_update=30):
        super(DQNParameters, self).__init__()
        self.learning_rate = lr
        self.action_size = action_size
        self.mem_size = mem_size
        self.replay_size = replay_size
        self.gamma = gamma
        self.n_hidden = n_hidden
        self.target_update = target_update

class DQNAgent():
    def __init__(self, input_size, output_size, dqn_parameter, epsilon):
        self.criterion = torch.nn.MSELoss()
        self.model = QNetwork(input_size, output_size, dqn_parameter.n_hidden, epsilon=epsilon)
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), dqn_parameter.learning_rate)

    def update(self, s, y):
        y_pred = self.predict(s)
        loss = self.criterion(y_pred, torch.Tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def target_predict(self, s):
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values[action])
            print(len(td_targets))
            self.update(states, td_targets)

    def soft_copy_target(self, pa_tau=0.1):
        model_params = self.model.named_parameters()

        target_params = self.target_model.named_parameters()
        dict_target_params = dict(target_params)
        for name1, param1 in model_params:
            if name1 in dict_target_params:
                dict_target_params[name1].data.copy_(
                    pa_tau * param1.data + (1 - pa_tau) *
                    dict_target_params[name1].data)
        self.model_target.load_state_dict(dict_target_params)

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def gen_epsilon_greedy_policy(self, epsilon, n_action):
        def policy_function(state):
            if random.random() < epsilon:
                return random.randint(0, n_action - 1)

            else:
                q_values = self.predict(state)
                return torch.argmax(q_values).item()
        return policy_function

    def gen_greedy_policy(self):
        def policy_function(state):
            q_values = self.predict(state)
            return torch.argmax(q_values).item()
        return policy_function

    def save_model(self, id, data_path):
        torch.save(self.model.state_dict(), data_path +
        " / dqn_model_" + id + ".pt")

    def load_model(self, id, data_path):
        self.model.load_state_dict(torch.load(data_path +
        " / dqn_model_" + id + ".pt"))

def grid_train():
    env = Game2048Env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    n_episode = 600
    n_hidden_options = [30, 40, 50]
    lr_options = [0.001]
    replay_size_options = [20, 25]
    target_update_options = [30, 35]
    for n_hidden in n_hidden_options:
        for lr in lr_options:
            for replay_size in replay_size_options:
                for target_update in target_update_options:
                    trainId = str(n_hidden) + "_" + str(lr) + "_" + str(replay_size) + "_" + str(target_update)
                    print("Net params: hidden layer: {} , learning rate: {}, replay size {}, target update {}".format(n_hidden, lr, replay_size, target_update))
                    env.seed = 1
                    random.seed(1)
                    epsilon = Epsilon()
                    torch.manual_seed(1)
                    dqn_param = DQNParameters(action_size, \
                              lr=lr, \
                              mem_size=30000, \
                              replay_size=replay_size, \
                              gamma=0.9,
                              n_hidden = n_hidden, \
                              target_update = target_update)
                    agent = DQNAgent(state_size, action_size,
                            dqn_param, epsilon)
                    train_network(env, agent, n_episode, epsilon, dqn_param, trainId)
    env.close()

if __name__ == '__main__':
    grid_train()
