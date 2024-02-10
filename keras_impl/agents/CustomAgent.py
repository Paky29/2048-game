import keras
from keras_impl.environment.Game2048Env import *
from keras_impl.model.ModelOne import ModelOne
from keras_impl.model.ModelTwo import ModelTwo
from collections import Counter

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# hide 1/1 [==============================] - 0s 21ms/step
keras.utils.disable_interactive_logging()


class CustomAgent:
    def __init__(self):
        self.env = Game2048Env()
        self.max_tile = []
        self.match_score = []
        self.loss_history = []

        #self.dqn_agent = ModelOne(env=self.env)
        self.dqn_agent = ModelTwo(env=self.env)

    def train(self, num_games):
        cur_game = 0
        num_games = num_games

        for game in range(num_games):
            print("========================================\nGame: ", game)
            self.env.reset()
            cur_state = self.env.get_board().reshape(4, 4)
            print(cur_state)
            done = False
            cur_game += 1

            while not done:
                action = self.dqn_agent.act(cur_state)
                reward, done = self.env.step(action)

                self.dqn_agent.remember(cur_state, action, reward, done)

                if cur_game > 2:
                    self.dqn_agent.replay()

                new_state = self.env.get_board()
                print('\n', new_state)
                cur_state = new_state

            if self.env.highest() >= 512:
                if self.env.highest() == 2048:
                    self.dqn_agent.save_model("../data/saved_model/s2048_num{}.model".format(cur_game))
                else:
                    if self.env.highest() == 1024:
                        self.dqn_agent.save_model("../data/saved_model/s1024_num{}.model".format(cur_game))
                    else:
                        self.dqn_agent.save_model("../data/saved_model/s512_num{}.model".format(cur_game))

            self.max_tile.append(self.env.highest())
            self.match_score.append(int(self.env.score))

            if self.dqn_agent.loss_history:
                self.loss_history.append(self.dqn_agent.loss_history[-1])
            else:
                pass

        print(f"\nMax Tile: ", self.max_tile)
        self.stampa_occorrenze(self.max_tile)
        print(f"\nScore: ", self.match_score)
        self.plot_results(self.max_tile, self.match_score, self.loss_history)

        print("Sum of match score: ", sum(self.match_score))

    @staticmethod
    def plot_results(max_tile, match_score, loss):
        # Plot Max Tile
        plt.figure(figsize=(12, 12))
        plt.subplot(311)
        plt.plot(range(1, len(max_tile) + 1), max_tile)
        plt.title('Max Tile per Game')
        plt.xlabel('Game')
        plt.ylabel('Max Tile')

        # Plot Match Score
        plt.subplot(312)
        plt.plot(range(1, len(match_score) + 1), match_score)
        plt.title('Score per Game')
        plt.xlabel('Game')
        plt.ylabel('Score')

        # Plot Loss
        plt.subplot(313)
        plt.plot(range(1, len(loss) + 1), loss)
        plt.title('Loss per Game')
        plt.xlabel('Game')
        plt.ylabel('Loss')

        plt.subplots_adjust(hspace=0.5)
        plt.savefig('../data/graphs/result.png')

    @staticmethod
    def stampa_occorrenze(array):
        occorrenze = Counter(array)
        for valore, conteggio in occorrenze.items():
            print(f"Punteggio: {valore}, Occorrenze: {conteggio}")


if __name__ == "__main__":
    agent = CustomAgent()
    agent.train(num_games=2)
