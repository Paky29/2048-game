from tkinter import Tk, Frame, Label, messagebox
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from DQLAgent import DQLAgent
from gui.board import Board


class Game:
    def __init__(self, gamepanel, agent):
        self.gamepanel = gamepanel
        self.agent = agent
        self.end = False
        self.won = False

    def start(self):
        self.gamepanel.random_cell()
        self.gamepanel.random_cell()
        self.gamepanel.paintGrid()
        self.gamepanel.window.update()
        self.play_game()

    def start_new_game(self):
        self.gamepanel = Board()  # Crea una nuova istanza della classe Board
        self.agent = DQLAgent(state_size=16, action_size=4)
        self.end = False
        self.won = False
        self.start()  # Inizia la nuova partita

    def play_game(self):
        #print("i'm in play game")
        while not self.end and not self.won:
            state = self.get_game_state()
            action = self.agent.act(state)
            self.take_action(action)
            self.gamepanel.paintGrid()
            self.gamepanel.window.update()
            self.check_game_status()
            self.agent.remember(state, action, self.gamepanel.score, self.get_game_state(), self.end)
        self.agent.replay(batch_size=64)


    def get_game_state(self):
        return np.reshape(np.array(self.gamepanel.gridCell), (1, self.agent.state_size))


    def take_action(self, action):

        score_move = self.gamepanel.score

        if action == 0:  # Su
            self.gamepanel.transpose()
            self.gamepanel.compressGrid()
            self.gamepanel.mergeGrid()
            self.gamepanel.moved = self.gamepanel.compress or self.gamepanel.merge
            self.gamepanel.compressGrid()
            self.gamepanel.transpose()

        elif action == 1:  # Giù
            self.gamepanel.transpose()
            self.gamepanel.reverse()
            self.gamepanel.compressGrid()
            self.gamepanel.mergeGrid()
            self.gamepanel.moved = self.gamepanel.compress or self.gamepanel.merge
            self.gamepanel.compressGrid()
            self.gamepanel.reverse()
            self.gamepanel.transpose()

        elif action == 2:  # Sinistra
            self.gamepanel.compressGrid()
            self.gamepanel.mergeGrid()
            self.gamepanel.moved = self.gamepanel.compress or self.gamepanel.merge
            self.gamepanel.compressGrid()

        elif action == 3:  # Destra
            self.gamepanel.reverse()
            self.gamepanel.compressGrid()
            self.gamepanel.mergeGrid()
            self.gamepanel.moved = self.gamepanel.compress or self.gamepanel.merge
            self.gamepanel.compressGrid()
            self.gamepanel.reverse()
        else:
            pass

        self.gamepanel.paintGrid()
        self.gamepanel.window.update()

        #print("Punteggio totale " + str(self.gamepanel.score))
        #print("Punteggio mossa " + str(self.gamepanel.score - score_move) + "\n")

    def check_game_status(self):
        flag = 0
        for i in range(4):
            for j in range(4):
                if self.gamepanel.gridCell[i][j] == 2048:
                    flag = 1
                    break

        if flag == 1:
            self.won = True
            messagebox.showinfo('2048', message='Hai vinto')
            return

        # Controllo se ci sono caselle vuote o possibilità di unire
        for i in range(4):
            for j in range(4):
                if self.gamepanel.gridCell[i][j] == 0:
                    flag = 1
                    break

        flag = 0
        for i in range(4):
            for j in range(4):
                if self.gamepanel.gridCell[i][j] == 0:
                    flag = 1
                    break

        if not (flag or self.gamepanel.can_merge()):
            self.end = True
            #messagebox.showinfo('2048', message='Hai perso')
            self.gamepanel.window.destroy()
            print("Score: " + str(self.gamepanel.score))
            self.start_new_game()

        # Se il modello ha effettuato una mossa valida, genera una nuova casella
        if self.gamepanel.moved:
            self.gamepanel.random_cell()

        # Aggiornamento visivo dopo la mossa
        self.gamepanel.paintGrid()
        self.gamepanel.window.update()


# main
if __name__ == "__main__":
    game_panel = Board()
    agent = DQLAgent(state_size=16, action_size=4)
    game = Game(game_panel, agent)
    game.start()
