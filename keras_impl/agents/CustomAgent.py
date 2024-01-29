import keras
from keras_impl.environment.Game2048Env import *
from keras_impl.model.CustomModel import CustomModel
from collections import Counter

# hide 1/1 [==============================] - 0s 21ms/step
keras.utils.disable_interactive_logging()


class CustomAgent:
    def __init__(self):
        self.env = Game2048Env()
        self.dqn_agent = CustomModel(env=self.env)

    def train(self, num_games):
        cur_game = 0
        max_tile = []
        match_score = []
        num_games = num_games

        for game in range(num_games):
            self.env.reset()
            cur_state = self.env.get_board().reshape(4, 4)
            print(cur_state)
            done = False
            cur_game += 1

            while not done:
                action = self.dqn_agent.act(cur_state)
                reward, done = self.env.step(action)
                new_state = self.env.get_board()
                print('\n', new_state)

                self.dqn_agent.remember(cur_state, action, reward, new_state, done)
                self.dqn_agent.replay()

                cur_state = new_state

            if self.env.highest() == 2048:
                print("Game:", cur_game, "WIN")
                self.dqn_agent.save_model("success.model")
            else:
                print(f"Game {cur_game} Failed to complete")
                if self.env.highest() >= 512:
                    self.dqn_agent.save_model("../data/saved_model/mag512_num{}.model".format(cur_game))

            max_tile.append(self.env.highest())
            match_score.append(int(self.env.score))

        print(f"\nMax Tile: ", max_tile)
        self.stampa_occorrenze(max_tile)
        print(f"\nScore: ", match_score)

    @staticmethod
    def stampa_occorrenze(array):
        occorrenze = Counter(array)
        for valore, conteggio in occorrenze.items():
            print(f"Punteggio: {valore}, Occorrenze: {conteggio}")


if __name__ == "__main__":
    agent = CustomAgent()
    agent.train(num_games=5)
