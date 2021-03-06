#!/usr/bin/env python3

import gym
import safety_gym
from gym import wrappers

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

def get_max_steps(max_steps, env):
    env_max_steps = env.spec.max_episode_steps
    print(f'Environment max steps: {env_max_steps}')
    if env_max_steps != None and env_max_steps < max_steps:
        max_steps = env_max_steps
    print(f'Configured max steps value to: {max_steps}')
    return max_steps

def run_one_episode(agent, env, render, test, max_steps, data_file=None):
    old_state = env.reset()
    episode_reward = 0
    episode_cost = 0
    cost = 0
    for step in range(max_steps):

        if render:
            env.render()
        if test:
            action = agent.get_test_action(old_state)
        else:
            action = agent.get_action(old_state)

        new_state, reward, done, info = env.step(action)
        if 'cost' in info:
            cost = info['cost']

        episode_cost += cost
        fail = done if (step+1) < max_steps else False
        if not test:
            agent.save(old_state, action, reward, new_state, cost, fail)
            agent.update()
        if data_file:
            data_file.writerow([step, old_state, action, reward, new_state, cost, fail])
        old_state = new_state
        episode_reward += reward
        if done:
            break
    return episode_reward, step, episode_cost, fail

def test(data_dir, agent, env, num_of_episodes, max_steps):
    episode_test_data_file = os.path.join(data_dir, 'episode_test.csv')
    all_rewards = 0
    max_steps = get_max_steps(max_steps, env)
    with open(episode_test_data_file, 'w', newline='') as episode_test_csvfile:
        episode_test_writer = csv.writer(episode_test_csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for episode in range(num_of_episodes):
            reward, step, cost, fail = run_one_episode(agent, env, True, True, max_steps)
            print(f'Episode: {episode}, step: {step}, reward: {reward}, cost: {cost}, fail: {fail}')
            all_rewards += reward
            episode_test_writer.writerow([episode, reward, cost])
    return all_rewards/num_of_episodes

def train(data_dir, agent, env, num_of_episodes, max_steps, episodes_show=50):
    episode_training_data_file = os.path.join(data_dir, 'episode_training.csv')
    training_data_file = os.path.join(data_dir, 'step_training.csv')
    test_data_file = os.path.join(data_dir, 'step_test.csv')
    # Define every how many episodes we will show a test and update the graphs
    min_episode_show = int(num_of_episodes / 10)
    if episodes_show > min_episode_show:
        episodes_show = min_episode_show

    max_steps = get_max_steps(max_steps, env)

    all_rewards = []
    all_costs = []
    all_test_rewards = []
    all_test_costs = []

    with open(episode_training_data_file, 'w', newline='') as episode_train_csvfile, \
         open(training_data_file, 'w', newline='') as train_csvfile, \
         open(test_data_file, 'w', newline='') as test_csvfile:

        episode_train_writer = csv.writer(episode_train_csvfile, delimiter=',',
                                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
        train_writer = csv.writer(train_csvfile, delimiter=',',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        test_writer = csv.writer(test_csvfile, delimiter=',',
                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
        total_steps = 0
        for episode in range(num_of_episodes):
            show = False
            if episode % episodes_show == 0:
                show = True
            reward, step, cost, fail = run_one_episode(agent, env, False, False, max_steps, train_writer)
            print(f'Episode: {episode}, step: {step}, reward: {reward}, cost: {cost}, fail: {fail}, cost/reward ratio: {cost/reward}')
            episode_train_writer.writerow([episode, reward, cost])

            total_steps += step
            all_rewards.append(reward)
            all_costs.append(cost)

            if show:
                reward, step, cost, fail = run_one_episode(agent, env, show, True, max_steps, test_writer)
                all_test_rewards.append(reward - cost)
                all_test_costs.append(cost)
                plot_rewards('Test rewards', all_test_rewards, episodes_show, episodes_show)
                plot_rewards('Test costs', all_test_costs, episodes_show, episodes_show)
                plot_rewards('Training rewards', all_rewards, episodes_show)
                plot_rewards('Training costs', all_costs, episodes_show)

def plot_rewards(name, train_rewards, avg_of, step=1):
    plt.figure(name)
    plt.clf()
    rewards = torch.tensor(train_rewards, dtype=torch.float)
    plt.title(f'{name}')
    plt.xlabel('Episode')
    plt.ylabel('Points')
    plt.plot(range(0, step * rewards.size()[0], step), rewards.numpy(), label='Episode reward')
    if len(rewards) >= avg_of:
        train_means = rewards.unfold(0, avg_of, 1).mean(1).view(-1)
        means_x = list(range(step * (avg_of - 1), step * (len(rewards)), step))
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

parser = argparse.ArgumentParser(description='Train for openai with DDPG or SAC algorithm.')
parser.add_argument('--env', dest='environment_name', type=str, choices=supported_environments,
                    required=True, help='Openai environment')
parser.add_argument('--alg', dest='algorithm', type=str, choices=supported_algorithms,
                    required=True, help='Algorithm to use to resolve the environment')
parser.add_argument('--episodes', dest="episodes", type=int, default=100, required=False,
                    help='Number of episodes to run')
parser.add_argument('--load-model', dest="model_dir", type=str, required=False,
                    help='Load model from dir')
parser.add_argument('--max-steps', dest="max_steps", type=int, required=False, default=1000,
                    help='Max steps per episode')
parser.add_argument('--test', dest="just_test", required=False, default=False, action='store_true',
                    help='Execute just tests. This option is remcomended with load-model option')
parser.add_argument('--video', dest="record_video", required=False, default=False, action='store_true',
                    help='Record the episodes to video')

args = parser.parse_args()

# Create data dir
data_dir_name = f'data/{args.algorithm}_{args.environment_name}_{time.strftime("%Y_%m_%d_%H_%M_%S")}'
execution_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(execution_path, data_dir_name)
os.makedirs(data_dir)
print(f'Data dir: {data_dir}')

## Define the environment
env = gym.make(args.environment_name)
if (args.record_video):
    env = wrappers.Monitor(env, f'{data_dir}/videos/')

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
    test(data_dir, agent, env, args.episodes, args.max_steps)
    print(f'All data saved in {data_dir}')
else:
    ## Train
    print('****TRAINING****')
    try:
        train(data_dir, agent, env, args.episodes, args.max_steps)
    finally:
        print('Saving the model')
        agent.save_model(data_dir)
        print(f'All data saved in {data_dir}')

    input("Press Enter to see the testing...")
    print('****TESTING****')
    test(data_dir, agent, env, 5, args.max_steps)

env.close()
input("Press Enter to end...")
