Implementation of reinforcement learning algorithms integrted with OpenAI gym and OpenAI safety gym.

# Algorithms supported
* DDPG(Deep Deterministic Policy Gradient)
* SAC(Soft Actor Critic)

# Requirements
* Python3
* OpenAI gym
* OpenAI safety gym
* Mujoco
* PyTorch

# Usage
```
usage: main.py [-h] --env
               {Pendulum-v0,LunarLanderContinuous-v2,BipedalWalker-v3,Hopper-v3,Walker2d-v3,HalfCheetah-v2,Ant-v3,Safexp-PointGoal0-v0,Safexp-CarGoal0-v0,Safexp-DoggoGoal0-v0,Safexp-PointGoal1-v0,Safexp-CarGoal1-v0,Safexp-DoggoGoal1-v0,Safexp-PointGoal2-v0,Safexp-CarGoal2-v0,Safexp-DoggoGoal2-v0,Safexp-PointButton0-v0,Safexp-CarButton0-v0,Safexp-DoggoButton0-v0,Safexp-PointButton1-v0,Safexp-CarButton1-v0,Safexp-DoggoButton1-v0,Safexp-PointButton2-v0,Safexp-CarButton2-v0,Safexp-DoggoButton2-v0,Safexp-PointPush0-v0,Safexp-CarPush0-v0,Safexp-DoggoPush0-v0,Safexp-PointPush1-v0,Safexp-CarPush1-v0,Safexp-DoggoPush1-v0,Safexp-PointPush2-v0,Safexp-CarPush2-v0,Safexp-DoggoPush2-v0}
               --alg {DDPG,SAC} [--episodes EPISODES] [--load-model MODEL_DIR]
               [--max-steps MAX_STEPS] [--test] [--video]

Train for openai with DDPG or SAC algorithm.

optional arguments:
  -h, --help            show this help message and exit
  --env {Pendulum-v0,LunarLanderContinuous-v2,BipedalWalker-v3,Hopper-v3,Walker2d-v3,HalfCheetah-v2,Ant-v3,Safexp-PointGoal0-v0,Safexp-CarGoal0-v0,Safexp-DoggoGoal0-v0,Safexp-PointGoal1-v0,Safexp-CarGoal1-v0,Safexp-DoggoGoal1-v0,Safexp-PointGoal2-v0,Safexp-CarGoal2-v0,Safexp-DoggoGoal2-v0,Safexp-PointButton0-v0,Safexp-CarButton0-v0,Safexp-DoggoButton0-v0,Safexp-PointButton1-v0,Safexp-CarButton1-v0,Safexp-DoggoButton1-v0,Safexp-PointButton2-v0,Safexp-CarButton2-v0,Safexp-DoggoButton2-v0,Safexp-PointPush0-v0,Safexp-CarPush0-v0,Safexp-DoggoPush0-v0,Safexp-PointPush1-v0,Safexp-CarPush1-v0,Safexp-DoggoPush1-v0,Safexp-PointPush2-v0,Safexp-CarPush2-v0,Safexp-DoggoPush2-v0}
                        Openai environment
  --alg {DDPG,SAC}      Algorithm to use to resolve the environment
  --episodes EPISODES   Number of episodes to run
  --load-model MODEL_DIR
                        Load model from dir
  --max-steps MAX_STEPS
                        Max steps per episode
  --test                Execute just tests. This option is remcomended with
                        load-model option
  --video               Record the episodes on video files
```
