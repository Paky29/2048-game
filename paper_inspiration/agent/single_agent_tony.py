import numpy as np
import torch as T

from collections import namedtuple

from gym_game.paper.paper_model import DeepQLearningNetwork

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'new_state', 'done'))

class SingleAgent:
    def __init__(self):
        self.lr = 0.01
        self.momentum = 0.9
        self.replay_memory_size = 10000
        self.replay_memory = np.empty(self.replay_memory_size, dtype=object)
        self.replay_memory_index = 0
        self.batch_size = 64

        self.gamma = 0.90
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # We use 2 models to prevent Bootstrapping
        self.main_model = DeepQLearningNetwork(lr=self.lr, action_space=4, momentum=self.momentum)
        # Aggiungi questa linea per inizializzare l'oggetto LossHistory
        #self.loss_history = LossHistory()
        #self.loss_history.losses = []

    #act senza le 4 mosse
    def choose_action(self, state):
        # azione che l'agente deve compiere in base allo stato

        # Epsilon value decays as model gains experience
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        #print("epsilon: ",self.epsilon)
        random = np.random.random()
        #print("Random epsilon: ", random)

        if random < self.epsilon:
            print("Random move")
            # Se un numero casuale è inferiore a epsilon, scegli un'azione casuale (esplorazione)
            return np.random.randint(0, 4)
        else:
            state = T.FloatTensor(state).unsqueeze(0).to(
                self.main_model.device)  # Transform it in tensor and add a dimension (batch dimension)
            actions = self.main_model(state)

            _, action = T.max(actions, 1)
            #print("Max Q value:", max_q_value)

            # Restituisci l'azione associata al massimo Q-value
            return action

    def store_transition(self, state, action, reward, new_state, done):
        # memorizza esperienza nell'archivio di replay
        # Replay Memory stores tuple(S, A, R, S')
        #self.memory.append([state, action, reward, new_state, done])
        transition = Transition(state, action, reward, new_state, done)
        self.replay_memory[self.replay_memory_index % self.replay_memory_size] = transition
        self.replay_memory_index += 1


    def learn(self):
        if self.replay_memory_index < self.batch_size:
            return

        print("Use memory")
        indices = np.random.choice(min(self.replay_memory_index, self.replay_memory_size), self.batch_size,
                                   replace=False)
        transitions = self.replay_memory[indices]

        states = T.tensor([transition.state for transition in transitions], dtype=T.float32)
        actions = T.tensor([transition.action for transition in transitions], dtype=T.long)
        rewards = T.tensor([transition.reward for transition in transitions], dtype=T.float32)
        new_states = T.tensor([transition.new_state for transition in transitions], dtype=T.float32)
        dones = T.tensor([transition.done for transition in transitions], dtype=T.float32)

        # Ottieni i valori Q predetti per gli stati attuali dal modello target
        targets = self.main_model(states).detach()
        # Calcoliamo i valori Q massimi per i nuovi stati
        new_state_values = self.main_model(new_states).detach().max(1)[0]

        # Aggiorniamo i valori Q target con la formula dell'equazione di Bellman
        targets[np.arange(self.batch_size), actions] = rewards + (1 - dones) * self.gamma * new_state_values

        # Zero gradients
        self.main_model.optimizer.zero_grad()
        # Calcolo delle perdite
        current_q_values = self.main_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.main_model.loss(current_q_values, targets.gather(1, actions.unsqueeze(1)).squeeze(1)).to(self.main_model.device)
        # Backpropagation
        loss.backward()
        # Aggiornamento dei parametri
        self.main_model.optimizer.step()

    def save_model(self, fn):
        # salva il modello su file
        T.save(self.main_model, 'model_tony.pth')
