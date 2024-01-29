import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
import numpy as np
import random

class Game2048Environment(Env):

    def __init__(self):
        super(Game2048Environment, self).__init__()
        # Definisci lo spazio degli osservabili e delle azioni
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        # Inizializza lo stato di gioco
        self.state = np.zeros((4, 4), dtype=np.int32)
        for _ in range(2):
            empty_cells = np.where(self.state == 0)
            if len(empty_cells[0]) > 0:
                idx = np.random.choice(len(empty_cells[0]))

                self.state[empty_cells[0][idx], empty_cells[1][idx]] = np.random.choice([2, 4], p=[0.8, 0.2])

        self.point = 0
        self.reward = 0
        self.max_tile = 0



    def reset(self, seed=None, options=None):
        # Resetta lo stato di gioco
        self.state = np.zeros((4, 4), dtype=np.int32)
        #random.seed(self.seed)
        # Aggiungi due nuove tessere 2 o 4 in posizioni vuote
        for _ in range(2):
            empty_cells = np.where(self.state == 0)
            if len(empty_cells[0]) > 0:
                idx = np.random.choice(len(empty_cells[0]))
                self.state[empty_cells[0][idx], empty_cells[1][idx]] = np.random.choice([2, 4], p=[0.8, 0.2])
        self.point = 0
        self.reward = 0
        self.max_tile = 0
        return self.state, {"Info" : "No info"}

    def highest(self):
        highest = 0
        for y in range(4):
            for j in range(4):
                highest = max(highest, self.state[y][j])
        return highest

    def step_normale(self, action):
        # Esegui una mossa (azione) nel gioco
        prev_state = np.copy(self.state)
        merged, score = self._merge(action)

        # Aggiorna lo stato e ricompensa
        self.point += score  # Utilizza la somma dei valori come ricompensa
        self.state = np.copy(merged)

        # Verifica se lo stato precedente è diverso da quello ottenuto dopo aver eseguito l'azione (azione valida)
        # Solo se sono diversi aggiunge un nuovo 2 alla board
        if not np.array_equal(prev_state, self.state):
            self._add_new_tile()

        self.reward = score

        done = self._is_game_over()

        return self.state, self.reward, done, False, self.get_info()

    def render(self):
        # Implementa il rendering dell'ambiente
        print(self.state)

    def _merge(self, action):
        # Implementa la logica di unione delle celle dopo un'azione
        # Restituisce il nuovo stato unito e il punteggio ottenuto
        merged = np.copy(self.state)
        score = 0

        if action == 0:  # Sposta su
            for col in range(4):
                merged[:, col], col_score = self._merge_column(merged[:, col])
                score += col_score
        elif action == 1:  # Sposta giù
            for col in range(4):
                #prendiamo i valori di una singola colonna in ordine inverso
                merged[:, col], col_score = self._merge_column(merged[::-1, col])
                score += col_score
            #riinvertiamo i valori ottenuti così da ottenere la colonna in ordine corretto
            merged = merged[::-1, :]
        elif action == 2:  # Sposta a sinistra
            for row in range(4):
                merged[row, :], row_score = self._merge_row(merged[row, :])
                score += row_score
        elif action == 3:  # Sposta a destra
            for row in range(4):
                merged[row, :], row_score = self._merge_row(merged[row, ::-1])
                score += row_score
            merged = merged[:, ::-1]

        return merged, score

    def _merge_row(self, row):
        # Implementa la logica di unione delle celle per una riga
        # Restituisce la nuova riga unita e il punteggio ottenuto

        row = row[row > 0]
        score = 0
        for i in range(len(row)-1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                score += row[i]
                row[i + 1] = 0
                i = 0
        #prendo nuovamente solo i valori diversi da 0 così da eliminare gli spazi vuoti tra i numeri uniti e spostare
        #tutti i numeri verso sinistra o destra a seconda dell'azione eseguita
        row = row[row > 0]

        #aggiungo un pad di 0 alla fine della riga per ottenere 4 celle
        row = np.pad(row, (0, (4 - len(row))), mode='constant')

        return row, score

    def _merge_column(self, col):
        # Implementa la logica di unione delle celle per una colonna
        # Restituisce la nuova colonna unita e il punteggio ottenuto
        col = col[col > 0]
        score = 0
        for i in range(len(col)-1):
            if col[i] == col[i + 1] and col[i] != 0:
                col[i] *= 2
                score += col[i]
                col[i + 1] = 0
                i = 0
        col = col[col > 0]
        col = np.pad(col, (0, (4 - len(col))), mode='constant')

        return col, score

    def _add_new_tile(self):
        # Aggiungi una nuova tessera 2 o 4 in una posizione vuota
        empty_cells = np.where(self.state == 0)
        if len(empty_cells[0]) > 0:
            idx = np.random.choice(len(empty_cells[0]))
            self.state[empty_cells[0][idx], empty_cells[1][idx]] = np.random.choice([2, 4], p=[0.8, 0.2])

    def _is_game_over(self):
        # Verifica se ci sono celle con valore 2048
        if np.any(self.state == 2048):
            return True

        # Verifica se ci sono celle vuote (valore 0)
        if np.any(self.state == 0):
            return False

        # Verifica se ci sono almeno due celle che possono essere unite in ogni direzione
        for action in range(4):
            _, score = self._merge(action)
            if score > 0:
                return False

        # Se nessuna delle condizioni sopra è soddisfatta, il gioco è terminato
        return True

    def get_info(self):
        return {"around": self.observation_space}

    def step(self, action):
        # Esegui una mossa (azione) nel gioco
        prev_state = np.copy(self.state)
        merged, score = self._merge(action)
        # Aggiorna lo stato e ricompensa
        self.point += score
        self.state = np.copy(merged)

        # Inizializza reward
        self.reward = score

        # Reward aggiuntiva per raggiungere una nuova tessera massima
        #max_tile_reward = 0
        if np.max(self.state) > self.max_tile:
            self.max_tile = np.max(self.state)
            self.reward *= 2

        # Penalty per mosse non produttive
        non_prod_penalty = 0
        if np.array_equal(prev_state, self.state):
            non_prod_penalty = -1  # ad esempio, -1 punto per mossa non produttiva
            self.reward += non_prod_penalty

        # Verifica se lo stato precedente è diverso da quello ottenuto dopo aver eseguito l'azione
        if not np.array_equal(prev_state, self.state):
            self._add_new_tile()

        # Controlla il rischio di game over
        game_over_risk_penalty = 0
        if self._is_game_over_risk():
            game_over_risk_penalty = -5  # ad esempio, -5 punti se il rischio di game over è elevato
            self.reward += game_over_risk_penalty

        done = self._is_game_over()

        return self.state, self.reward, done, False, self.get_info()

    def _is_game_over_risk(self):
        # Determina il numero di celle vuote
        empty_cells = np.count_nonzero(self.state == 0)

        # Considera il gioco a rischio se ci sono meno di X celle vuote
        # X può essere un numero che scegli in base alla difficoltà che vuoi impostare
        if empty_cells > 5:
            return False

        # Controlla se ci sono mosse possibili che possono unire le tessere
        for i in range(4):  # Assumendo una griglia 4x4
            for j in range(4):
                if j + 1 < 4 and self.state[i][j] == self.state[i][j + 1]:
                    return False  # Esiste almeno una mossa possibile
                if i + 1 < 4 and self.state[i][j] == self.state[i + 1][j]:
                    return False  # Esiste almeno una mossa possibile

        # Se non ci sono celle vuote sufficienti e non ci sono mosse possibili, il gioco è a rischio
        return True
