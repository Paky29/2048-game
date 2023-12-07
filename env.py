import gym
from gym import spaces
import numpy as np

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        # Definisci lo spazio degli osservabili e delle azioni
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        # Inizializza lo stato di gioco
        self.state = np.zeros((4, 4), dtype=np.int32)
        for _ in range(2):
            empty_cells = np.where(self.state == 0)
            if len(empty_cells[0]) > 0:
                idx = np.random.choice(len(empty_cells[0]))
                self.state[empty_cells[0][idx], empty_cells[1][idx]] = np.random.choice([2, 4])

        self.point = 0

    def _add_new_tile(self):
        # Aggiungi una nuova tessera 2 o 4 in una posizione vuota
        empty_cells = np.where(self.state == 0)
        if len(empty_cells[0]) > 0:
            idx = np.random.choice(len(empty_cells[0]))
            self.state[empty_cells[0][idx], empty_cells[1][idx]] = 2

    def reset(self):
        # Resetta lo stato di gioco
        self.state = np.zeros((4, 4), dtype=np.int32)

        # Aggiungi due nuove tessere 2 o 4 in posizioni vuote
        for _ in range(2):
            empty_cells = np.where(self.state == 0)
            if len(empty_cells[0]) > 0:
                idx = np.random.choice(len(empty_cells[0]))
                self.state[empty_cells[0][idx], empty_cells[1][idx]] = np.random.choice([2, 4])
        self.point = 0
        return self.state

    def step(self, action):
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

        done = self._is_game_over()
        return self.state, self.point, done, {}

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

