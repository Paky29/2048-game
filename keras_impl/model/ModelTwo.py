import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl.memory import SequentialMemory
from keras_impl.environment.Game2048Env import IllegalMove


class DeepQLearningNetwork(nn.Module):
    def __init__(self, lr, action_space, momentum):
        super(DeepQLearningNetwork, self).__init__()

        # Assuming the state is a 4x4 grid, with a single 'channel'
        input_channels = 1

        # Layer 1: Two parallel convolutional layers
        self.conv1a = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(input_channels, 128, kernel_size=5, stride=1, padding=2)

        # Layer 2: Four parallel convolutional layers
        # Adjust kernel sizes and paddings to ensure the output dimensions are the same
        self.conv2a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv2c = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3)
        self.conv2d = nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4)
        self.conv2e = nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5)

        # Layer 3: Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Layer 4: Linear layer for output
        # Update the input features to the linear layer based on the output size
        self.fc = nn.Linear(128 * 5 * 4 * 4 * 2, action_space)
        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.loss = nn.HuberLoss()

        # Utilizzare la GPU se disponibile, altrimenti utilizzare la CPU
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # for macbook m1
        # self.device = T.device("mps")

        self.to(self.device)  # Trasferisce il modello sulla GPU o sulla CPU

    def forward(self, x):
        x = x.float().unsqueeze(1)  # Add a channel dimension

        x = x.squeeze()  # Rimuovi le dimensioni ridondanti (1x1) se presenti
        x = x.view(-1, 1, 4, 4)  # Aggiorna le dimensioni del batch

        # Apply convolutional layers
        x1 = F.relu(self.conv1a(x))
        x2 = F.relu(self.conv1b(x))

        x1a = F.relu(self.conv2a(x1))
        x1b = F.relu(self.conv2b(x1))
        x1c = F.relu(self.conv2c(x1))
        x1d = F.relu(self.conv2d(x1))
        x1e = F.relu(self.conv2e(x1))

        x2a = F.relu(self.conv2a(x2))
        x2b = F.relu(self.conv2b(x2))
        x2c = F.relu(self.conv2c(x2))
        x2d = F.relu(self.conv2d(x2))
        x2e = F.relu(self.conv2e(x2))

        # Concatenate outputs
        # Make sure the dimensions match for concatenation

        x = T.cat((x1a, x1b, x1c, x1d, x1e, x2a, x2b, x2c, x2d, x2e), 1)

        # Apply dropout
        x = self.dropout(x)

        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)

        # Apply the linear layer
        x = self.fc(x)

        return x


class ModelTwo:
    def __init__(self, env):
        self.env = env

        self.memory_limit = 2000
        self.memory = SequentialMemory(limit=self.memory_limit, window_length=1)
        self.batch_size = 100

        self.gamma = 0.90
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02
        self.learning_rate = 0.001

        self.loss_history = []

        self.model = DeepQLearningNetwork(lr=self.learning_rate, action_space=4, momentum=0.9)
        self.model.to(self.model.device)

        '''
        # Generare e salvare la visualizzazione del grafo del modello
        example_input = T.randn(1, 4, 4)
        output = self.model(example_input)
        dot = make_dot(output, params=dict(self.model.named_parameters()))
        dot.render("../data/graphs/model2_plot", format="png", cleanup=True)
        '''

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

            q_values = self.model.forward(T.tensor(state, dtype=T.float32).to(self.model.device).unsqueeze(0))

            q_values = q_values.squeeze().cpu().detach().numpy()

            max_q_value = np.argmax([q_values[action] for action in legal_moves])
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

        states = T.tensor(states, dtype=T.float32).to(self.model.device)
        new_states = T.tensor(new_states, dtype=T.float32).to(self.model.device)
        actions = T.tensor(actions, dtype=T.long).to(self.model.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.model.device)
        terminals = T.tensor(terminals, dtype=T.float32).to(self.model.device)

        targets = self.model.forward(states).to(self.model.device)

        new_state_values = T.max(self.model.forward(new_states).detach(), dim=1)[0]

        targets[T.arange(self.batch_size), actions] = rewards + (1 - terminals) * self.gamma * new_state_values

        loss = self.model.loss(targets, self.model.forward(states)).to(self.model.device)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        self.loss_history.append(loss.item())

    def save_model(self, fn):
        T.save(self.model.state_dict(), fn)
