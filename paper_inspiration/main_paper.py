from gym_game.envs.plot_poi import RealTimeScorePlotter
from gym_game.paper.agent import MyAgent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gym_game.paper.double_agent_tony import Agent_tony
from gym_game.paper.single_agent_tony import SingleAgent


def play_agent():
    env = gym.make("gym_game:gym_game/Game2048-v0")
    print("Spazio delle osservazioni:", env.observation_space.shape)
    print("Spazio delle azioni: ", env.action_space.n)

    input_dim = (4, 4)

    #agent = MyAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dim=input_dim, lr=0.001)
    agent = Agent_tony()
    scores, eps_history = [], []

    n_games = 100
    plotter = RealTimeScorePlotter()
    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        env.render()
        while not done:
            obs = np.array(observation)  # Transform the observation in numpy array
            action = agent.choose_action(obs)
            print("azione intrapresa: ", action)
            new_observation, reward, done, _, info = env.step(action)
            env.render()
            score += reward

            agent.store_transition(observation, action, reward, new_observation, done)
            agent.learn()
            if i%20 == 0:
                agent.target_train()

            observation = new_observation

        plotter.update_plot(env.unwrapped.highest())
        plotter.save()
        scores.append(score)
        eps_history.append(agent.epsilon)

    print("Max score: ", max(scores))
    agent.save_model(1)
    plotter.close_plot()

    env.close()

if __name__ == '__main__':
    play_agent()