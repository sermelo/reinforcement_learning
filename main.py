import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt

from lib.agent import Agent
gym.logger.set_level(40)

def train(env_name, num_of_episodes):
    env = gym.make(env_name)
    batch_size = 200
    agent = Agent(env, batch_size)

    all_episodes_rewards = []
    for episode in range(num_of_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(200):
            if episode % 5 == 0:
                env.render()
            action = agent.get_action(state)
            new_state, reward, done, info = env.step(action)
            agent.save(state, action, reward, new_state)
            state = new_state
            episode_reward += reward
            if done:
                print("episode: {}, step: {}, reward: {}".format(episode, step, episode_reward))
                break
        all_episodes_rewards.append(episode_reward)
    env.close()

    avg_rewards = []
    avg_rewards.append(np.mean(all_episodes_rewards[-10:]))

    plt.plot(all_episodes_rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

supported_environments = ['Pendulum-v0', 'Ant-v3']
parser = argparse.ArgumentParser(description='Train for openai with DDPG algoritm.')
parser.add_argument('--env', dest='environment_name', type=str, choices=supported_environments,
                    required=True, help='Openai environment')
args = parser.parse_args()
num_of_episodes = 75
train(args.environment_name, num_of_episodes)