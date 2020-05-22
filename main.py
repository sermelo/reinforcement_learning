import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch

from lib.agent import Agent
gym.logger.set_level(40)

def test_one_episode(agent, env, render):
    max_steps = env.spec.max_episode_steps
    if max_steps == None:
        max_steps = 1000
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        if render:
            env.render()
        action = agent.get_test_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            break
    return episode_reward, step

def test(agent, env, num_of_episodes):
    all_episodes_rewards = 0
    for episode in range(num_of_episodes):
        episode_reward, step = test_one_episode(agent, env, True)
        print("episode: {}, step: {}, reward: {}".format(episode, step, episode_reward))
        all_episodes_rewards += episode_reward
    return all_episodes_rewards/num_of_episodes

def train(agent, env, num_of_episodes, update_rate):
    episodes_show = 50
    max_steps = env.spec.max_episode_steps
    if max_steps == None:
        max_steps = 1000

    all_episodes_rewards = []
    avg_rewards = []
    test_episodes_rewards = []
    test_avg_rewards = []
    for episode in range(num_of_episodes):
        show = False
        if episode % episodes_show == 0:
            show = True
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # env.render()
            action = agent.get_action(state)
            new_state, reward, done, info = env.step(action)
            fail = done if (step+1) < max_steps else False
            agent.save(state, action, reward, new_state, fail)
            state = new_state
            episode_reward += reward
            if done:
                print("episode: {}, step: {}, reward: {}".format(episode, step, episode_reward))
                break
        agent.update(update_rate)

        all_episodes_rewards.append(episode_reward)


        test_episode_reward, test_step = test_one_episode(agent, env, show)
        test_episodes_rewards.append(test_episode_reward)

        if show:
            plot_rewards('Test', test_episodes_rewards)
            plot_rewards('Training', all_episodes_rewards)

def plot_rewards(name, train_rewards):
    plt.figure(name)
    plt.clf()
    avg_of = 100
    rewards = torch.tensor(train_rewards, dtype=torch.float)
    plt.title(f'{name} rewards')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards.numpy(), label='Episode reward')
    if len(rewards) >= avg_of:
        train_means = rewards.unfold(0, avg_of, 1).mean(1).view(-1)
        means_x = list(range(avg_of -1, len(rewards)))
        plt.plot(means_x, train_means.numpy(), label=f'{avg_of} episodes avg reward')
    plt.legend()
    plt.pause(0.001)

supported_environments = ['Pendulum-v0', 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'Hopper-v3', 'Walker2d-v3', 'HalfCheetah-v2', 'Ant-v3']
parser = argparse.ArgumentParser(description='Train for openai with DDPG algoritm.')
parser.add_argument('--env', dest='environment_name', type=str, choices=supported_environments,
                    required=True, help='Openai environment')
parser.add_argument('--episodes', dest="episodes", type=int, default=100, required=False,
                    help='Number of episodes to run')
parser.add_argument('--nn-update-rate', dest="nn_update_rate", type=int, default=100, required=False,
                    help='Number of neuronal networks training cicles per episode')

args = parser.parse_args()

batch_size = 100
env = gym.make(args.environment_name)
agent = Agent(env, batch_size)
print('****TRAINING****')
train(agent, env, args.episodes, args.nn_update_rate)
input("Press Enter to see the testing...")
print('****TESTING****')
test(agent, env, 5)
env.close()
input("Press Enter to end...")
