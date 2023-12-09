import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQLAgent:
    def __init__(self, state_size, action_size):
        # Inizializzazione dell'agente con le dimensioni dello stato e dell'azione
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Memorizzazione delle esperienze della partita corrente
        # Parametri di apprendimento
        self.gamma = 0.95  # Fattore di sconto per il calcolo del reward futuro
        self.epsilon = 1.0  # Tasso di esplorazione iniziale
        self.epsilon_decay = 0.995  # Decadimento esponenziale del tasso di esplorazione
        self.epsilon_min = 0.01  # Tasso minimo di esplorazione
        self.learning_rate = 0.001  # Tasso di apprendimento per l'ottimizzatore
        # Creazione del modello della rete neurale
        self.model = self._build_model()

    def _build_model(self):
        # Costruzione del modello della rete neurale
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Memorizzazione dell'esperienza attuale nella memoria
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Selezione di un'azione basata sulla politica epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Azione casuale con probabilità epsilon
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Selezione dell'azione con il massimo valore Q

    def replay(self, batch_size):
        # Esecuzione del processo di "replay" solo alla fine di ogni partita
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Calcolo del target Q
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            # Aggiornamento dei valori Q nella rete neurale
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # Decadimento del tasso di esplorazione epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Pulizia della memoria alla fine della partita
        self.memory = []
