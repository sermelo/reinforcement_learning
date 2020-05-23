import gym
import safety_gym

import sys
import os
import time
import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch

from lib.ddpg_agent import DdpgAgent
from lib.sac_agent import SacAgent

gym.logger.set_level(40)

def test_one_episode(agent, env, render, max_steps):
    env_max_steps = env.spec.max_episode_steps
    if env_max_steps != None and env_max_steps < max_steps:
        max_steps = env_max_steps
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

def test(agent, env, num_of_episodes, max_steps):
    all_episodes_rewards = 0
    for episode in range(num_of_episodes):
        episode_reward, step = test_one_episode(agent, env, True, max_steps)
        print("Episode: {}, step: {}, reward: {}".format(episode, step, episode_reward))
        all_episodes_rewards += episode_reward
    return all_episodes_rewards/num_of_episodes

def train(data_dir, agent, env, num_of_episodes, max_steps, episodes_show=50):
    training_data_file = os.path.join(data_dir, 'training.csv')
    test_data_file = os.path.join(data_dir, 'test.csv')
    # Define every how many episodes we will show a test and update the graphs
    min_episode_show = int(num_of_episodes / 10)
    if episodes_show > min_episode_show:
        episodes_show = min_episode_show

    env_max_steps = env.spec.max_episode_steps
    if env_max_steps != None and env_max_steps < max_steps:
        max_steps = env_max_steps

    all_episodes_rewards = []
    avg_rewards = []
    test_episodes_rewards = []
    test_avg_rewards = []

    with open(training_data_file, 'w', newline='') as csvfile:
        train_writer = csv.writer(csvfile, delimiter=',',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with open(test_data_file, 'w', newline='') as csvfile:
            test_writer = csv.writer(csvfile, delimiter=',',
                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
            total_steps = 0
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
                        break
                print("Episode: {}, step: {}, reward: {}".format(episode, step, episode_reward))
                agent.update(step)

                total_steps += step
                all_episodes_rewards.append(episode_reward)
                train_writer.writerow([episode, step, total_steps, episode_reward])

                if show:
                    test_episode_reward, test_step = test_one_episode(agent, env, show, max_steps)
                    test_episodes_rewards.append(test_episode_reward)
                    test_writer.writerow([episode, test_step, total_steps, test_episode_reward])
                    plot_rewards('Test', test_episodes_rewards, episodes_show, episodes_show)
                    plot_rewards('Training', all_episodes_rewards, episodes_show)


def plot_rewards(name, train_rewards, avg_of, step=1):
    plt.figure(name)
    plt.clf()
    rewards = torch.tensor(train_rewards, dtype=torch.float)
    plt.title(f'{name} rewards')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(range(0, step * rewards.size()[0], step), rewards.numpy(), label='Episode reward')
    if len(rewards) >= avg_of:
        train_means = rewards.unfold(0, avg_of, 1).mean(1).view(-1)
        means_x = list(range(avg_of -1, len(rewards)))
        plt.plot(means_x, train_means.numpy(), label=f'{avg_of} episodes avg reward')
    plt.legend()
    plt.pause(0.001)

## Prepare the input arguments
supported_algorithms = ['DDPG', 'SAC']

supported_environments = [
        'Pendulum-v0',
        'LunarLanderContinuous-v2',
        'BipedalWalker-v3',
        'Hopper-v3',
        'Walker2d-v3',
        'HalfCheetah-v2',
        'Ant-v3',
]
# Add all safetygym environments
for aim in ['Goal', 'Button', 'Push']:
    for level in [0, 1, 2]:
        for robot in ['Point', 'Car', 'Doggo']:
            supported_environments.append(f'Safexp-{robot}{aim}{level}-v0')

parser = argparse.ArgumentParser(description='Train for openai with DDPG algoritm.')
parser.add_argument('--env', dest='environment_name', type=str, choices=supported_environments,
                    required=True, help='Openai environment')
parser.add_argument('--alg', dest='algorithm', type=str, choices=supported_algorithms,
                    required=True, help='Algorithm to use to resolve the environment')
parser.add_argument('--episodes', dest="episodes", type=int, default=100, required=False,
                    help='Number of episodes to run')
parser.add_argument('--load-model', dest="model_dir", type=str, required=False,
                    help='Load model from dir')
parser.add_argument('--max-steps', dest="max_steps", type=int, required=False, default=500,
                    help='Max steps per episode')
parser.add_argument('--test', dest="just_test", required=False, default=False, action='store_true',
                    help='Execute just tests. This option is remcomended with load-model option')


args = parser.parse_args()

## Define the environment
env = gym.make(args.environment_name)

# Create data dir
data_dir_name = f'data/{args.algorithm}_{args.environment_name}_{time.strftime("%Y_%m_%d_%H_%M_%S")}'
execution_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(execution_path, data_dir_name)
os.makedirs(data_dir)
print(f'Data dir: {data_dir}')

## Define the agent
batch_size = 100
if (args.algorithm == 'DDPG'):
    agent = DdpgAgent(env, batch_size)
elif (args.algorithm == 'SAC'):
    agent = SacAgent(env, batch_size)
else:
    print('Algorithm not suported')
    sys.exit(1)

if args.model_dir:
    agent.load_model(args.model_dir)

if args.just_test:
    test(agent, env, args.episodes, args.max_steps)
else:
    ## Train
    print('****TRAINING****')
    train(data_dir, agent, env, args.episodes, args.max_steps)
    print('Saving the model')
    agent.save_model(data_dir)
    print(f'All data saved in {data_dir}')
    input("Press Enter to see the testing...")
    print('****TESTING****')
    test(agent, env, 5, args.max_steps)

env.close()
input("Press Enter to end...")
