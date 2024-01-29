import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.envs.registration import register

from gym_game.ddqn import DoubleDQN
from gym_game.envs.plot_poi import RealTimeScorePlotter

register(
    id="Game2048-v0",
    entry_point="gym_game.envs:Game2048Environment",
    #max_episode_steps=2000
)

TRAIN = True

def create_modeld():
    n_cpu = 8
    env = make_vec_env("Game2048-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = DoubleDQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=1e-4,
                buffer_size=15000,
                learning_starts=100000,
                batch_size=32,
                gamma=0.90,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.1,
                exploration_final_eps=0.2,
                exploration_initial_eps=1.0,
                verbose=1)
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=1000000)
        model.save("gym_game/model1")
        del model

def create_model():
    n_cpu = 8
    env = make_vec_env("Game2048-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=1e-4,
                buffer_size=15000,
                learning_starts=100000,
                batch_size=32,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.1,
                exploration_final_eps=0.2,
                exploration_initial_eps=1.0,
                verbose=1)
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=10000000)
        model.save("gym_game/model1")
        del model

    # Run the algorithm
def play_model():
    env = gym.make("Game2048-v0")
    model = DQN.load("gym_game/model1", env=env)

    scores, eps_history, highest = [], [], []
    n_games = 1000
    plotter = RealTimeScorePlotter()
    for i in range(n_games):
        score = 0
        done = truncated = False
        obs, info = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs)
            print("Action:", action)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            print("Main",reward)
            # Render
            env.render()
            if reward > 0:
                score += reward

            # Start the thread to update weather and grass

        plotter.update_plot(env.unwrapped.highest())
        plotter.save()
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # average of the last 100 scores
        highest.append(env.highest())
        print("Partita: {}, Score: {}".format(i, score, avg_score))

    print("Highest scores:", highest)
    print("Average score: {}".format(avg_score))
    print("Max score: ", max(scores))
    plotter.close_plot()
    model.save('model/model_sb.zip')

if __name__ == '__main__':
    play_model()