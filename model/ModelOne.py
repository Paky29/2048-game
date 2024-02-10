import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers.legacy import SGD
from environment.Game2048Env import IllegalMove
from keras.utils import plot_model
from rl.memory import SequentialMemory


class ModelOne:
    def __init__(self, env):
        self.env = env

        self.memory_limit = 4000
        self.memory = SequentialMemory(limit=self.memory_limit, window_length=1)
        self.batch_size = 200

        self.gamma = 0.90
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02
        self.learning_rate = 0.001

        self.loss_history = []

        self.model = self.create_model()
        plot_model(self.model, to_file='../data/graphs/model1_plot.png', show_shapes=True, show_layer_names=True)

    def create_model(self):
        model = Sequential()

        # input layer
        model.add(Flatten(name="InputLayer", input_shape=(4, 4)))

        # hidden layer
        model.add(Dense(name="1", units=1024, activation="relu"))
        model.add(Dense(name="2", units=512, activation="relu"))
        model.add(Dense(name="3", units=256, activation="relu"))
        model.add(Dense(name="4", units=128, activation="relu"))
        model.add(Dense(name="5", units=64, activation="relu"))
        model.add(Dense(name="6", units=32, activation="relu"))
        model.add(Dense(name="7", units=16, activation="relu"))
        model.add(Dense(name="8", units=8, activation="relu"))

        # output layer
        model.add(Dense(name="OutputLayer", units=4, activation='linear'))

        model.compile(loss='huber', optimizer=SGD(learning_rate=self.learning_rate))

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

            q_values = self.model.predict(state.reshape(-1, 4, 4))

            max_q_value = np.argmax([q_values[0][action] for action in legal_moves])
            return legal_moves[max_q_value]

    def remember(self, state, action, reward, done):
        self.memory.append(state, action, reward, done)

    def replay(self):
        if self.memory.nb_entries < self.batch_size:
            return

        print("In replay\n")

        transitions = self.memory.sample(self.batch_size)

        states = np.array([transition.state0 for transition in transitions])
        actions = np.array([transition.action for transition in transitions])
        rewards = np.array([transition.reward for transition in transitions])

        new_states = np.array([transition.state1 for transition in transitions])
        new_states = np.squeeze(new_states)

        terminals = np.array([transition.terminal1 for transition in transitions])

        # Converti gli stati in float e mantieni la forma corretta (n=batch_size stati di 4x4)
        states = states.reshape(self.batch_size, 4, 4)
        new_states = new_states.reshape(self.batch_size, 4, 4)

        targets = self.model.predict(states)

        new_state_values = np.max(self.model.predict(new_states), axis=1)

        targets[np.arange(self.batch_size), actions] = rewards + (1 - terminals) * self.gamma * new_state_values

        # Esegui il fit con gli stati convertiti
        history = self.model.fit(states, targets, epochs=1, verbose=0)

        # Appendi il valore della perdita a loss_history
        self.loss_history.append(history.history['loss'][0])

    def save_model(self, fn):
        self.model.save(fn)
