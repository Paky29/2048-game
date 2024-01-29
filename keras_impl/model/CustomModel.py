import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from collections import namedtuple
from keras_impl.environment.Game2048Env import IllegalMove

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'new_state', 'done'))

'''
def process_log(observation):
    observation = np.asarray(observation)
    return observation.reshape(-1, 4, 4)
'''


class CustomModel:
    def __init__(self, env):
        self.env = env

        self.replay_memory_size = 200
        self.replay_memory = np.empty(self.replay_memory_size, dtype=object)
        self.replay_memory_index = 0
        self.batch_size = 30

        self.gamma = 0.90
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02
        self.learning_rate = 0.001

        self.loss_history = []

        self.main_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(4, 4)))

        model.add(Dense(units=1024, activation="relu"))
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=256, activation="relu"))

        model.add(Dense(units=4, activation='linear'))

        opt = SGD(learning_rate=self.learning_rate)

        model.compile(loss="huber", optimizer=opt)

        print(model.summary())

        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        random_move = np.random.random()

        if random_move < self.epsilon:
            print("Random move")
            return np.random.randint(0, 4)
        else:
            legal_moves = []
            for action in range(4):
                try:
                    # Check if the move is legal
                    self.env.move(action, trial=True)
                    legal_moves.append(action)
                except IllegalMove:
                    pass

            # q_values = self.main_model.predict(process_log(state))
            q_values = self.main_model.predict(state.reshape(-1, 4, 4))

            max_q_value = np.argmax([q_values[0][action] for action in legal_moves])
            return legal_moves[max_q_value]

    def remember(self, state, action, reward, new_state, done):
        transition = Transition(state, action, reward, new_state, done)
        self.replay_memory[self.replay_memory_index % self.replay_memory_size] = transition
        self.replay_memory_index += 1

    def replay(self):
        if self.replay_memory_index < self.batch_size:
            return

        print("Use memory")
        indices = np.random.choice(min(self.replay_memory_index, self.replay_memory_size), self.batch_size,
                                   replace=False)
        transitions = self.replay_memory[indices]

        states = np.array([transition.state for transition in transitions])
        actions = np.array([transition.action for transition in transitions])
        rewards = np.array([transition.reward for transition in transitions])

        new_states = np.array([transition.new_state for transition in transitions])
        new_states = np.squeeze(new_states)

        dones = np.array([transition.done for transition in transitions])

        # Converti gli stati in float e mantieni la forma corretta (32 stati di 4x4)
        # states = states.astype(np.float32).reshape(self.batch_size, 4, 4)
        # new_states = new_states.astype(np.float32).reshape(self.batch_size, 4, 4)
        states = states.reshape(self.batch_size, 4, 4)
        new_states = new_states.reshape(self.batch_size, 4, 4)

        # targets = self.main_model.predict(process_log(states))
        targets = self.main_model.predict(states)

        # new_state_values = np.max(self.main_model.predict(process_log(new_states)), axis=1)
        new_state_values = np.max(self.main_model.predict(new_states), axis=1)

        targets[np.arange(self.batch_size), actions] = rewards + (1 - dones) * self.gamma * new_state_values

        # Esegui il fit con gli stati convertiti
        # self.main_model.fit(process_log(states), targets, epochs=10, verbose=0)
        self.main_model.fit(states, targets, epochs=10, verbose=0)

    def save_model(self, fn):
        self.main_model.save(fn)
