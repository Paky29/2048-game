#import numpy as np

from env import Game2048Env

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.2, gamma=0.9, epsilon=0.9, xi=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi
        self.cur_policy = np.random.randint(action_size, size=state_size)
        self.q_table = np.zeros([state_size, action_size])

    def actuate(self, s):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            print(self.q_table[s])
            return np.argmax(self.q_table[s])

    def percept(self, s, a, s_prime, r):
        q_prime = np.max(self.q_table[s_prime])
        old_q_value = self.q_table[s, a]
        learned_value = r + self.gamma * q_prime - old_q_value
        self.q_table[s, a] += self.alpha * learned_value
        self.cur_policy[s] = np.argmax(self.q_table[s])

    def update_episode(self):
        self.epsilon *= self.xi

# Puoi ora utilizzare l'agente addestrato per giocare al gioco
