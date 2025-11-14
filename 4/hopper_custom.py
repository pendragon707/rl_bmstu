import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, DQN, PPO
import os
import argparse

from pathlib import Path

from gymnasium import RewardWrapper
import numpy as np

class HopperCustomRewardWrapper(RewardWrapper):
    """
    Custom reward wrapper for Hopper-v5 to encourage different locomotion styles.
    
    Modes:
      - 'forward': default (positive x-velocity)
      - 'backward': negative x-velocity
      - 'jump_in_place': minimize x-velocity, encourage vertical motion or contact
    """
    def __init__(self, env, mode="forward"):
        super().__init__(env)
        self.mode = mode
        assert mode in ["forward", "backward", "jump_in_place"], f"Unknown mode: {mode}"

    def reward(self, original_reward):
        # Access MuJoCo simulation data via unwrapped environment
        # Hopper state: qpos = [x, z, angle, leg angles...]
        #               qvel = [x_vel, z_vel, angular_vel, ...]
        
        x_pos = self.unwrapped.data.qpos[0]      # horizontal position
        x_vel = self.unwrapped.data.qvel[0]      # horizontal velocity
        z_vel = self.unwrapped.data.qvel[1]      # vertical velocity
        
        # Health/survival info (from original env)
        is_healthy = self.unwrapped.is_healthy
        healthy_reward = 1.0 if is_healthy else 0.0

        if self.mode == "forward":
            # Original behavior: move forward fast
            behavior_reward = x_vel

        elif self.mode == "backward":
            # Encourage moving left (negative x direction)
            behavior_reward = -x_vel

        elif self.mode == "jump_in_place":
            # Goal: stay near origin + bounce vertically
            # Penalize horizontal movement
            x_penalty = -0.5 * np.abs(x_vel)          # discourage x-motion
            # Encourage vertical motion (bouncing)
            vertical_reward = 1.0 * np.abs(z_vel)     # reward up/down motion
            # Optional: reward staying near start (x ≈ 0)
            position_penalty = -0.1 * np.abs(x_pos)   # keep near origin
            
            behavior_reward = vertical_reward + x_penalty + position_penalty

        else:
            behavior_reward = x_vel  # fallback

        # Combine behavior + survival
        total_reward = behavior_reward + 0.1 * healthy_reward

        return float(total_reward)


def train(env, sb3_algo, timesteps = 25000):
    if sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir,learning_rate=0.01)
    elif sb3_algo == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    else:
        print('Algorithm not found!')
        return

    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{timesteps * iters}")

def test(env, sb3_algo, path_to_model):
    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    elif sb3_algo == 'DQN':
        # DQN works on discrete action space
        model = DQN.load(path_to_model, env=env)
    elif sb3_algo == 'PPO':
        model = PPO.load(path_to_model, env=env)
    else:
        print('Algorithm not found!')
        return

    # First observation in the env
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Test model.')
    parser.add_argument('-e', '--gymenv', default="Ant-v5", help='Gymnasium environment (Humanoid-v5, Ant-v5, HalfCheetah-v5 и т. д.)')
    parser.add_argument('-a', '--sb3_algo', default="PPO", help='StableBaseline3 RL-алгоритмы (SAC, TD3, A2C, DQN, PPO)')
    parser.add_argument('-n', '--num_timesteps', default="25000", help='Количество шагов обучения')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    REWARD_MODE = "jump_in_place"

    # Create directories to hold models and logs
    model_dir = "models/models_"+args.gymenv+REWARD_MODE+"/"+args.sb3_algo
    log_dir = "logs/logs_"+args.gymenv + REWARD_MODE
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.train:
        base_env = gym.make(args.gymenv, render_mode=None)
        gymenv = HopperCustomRewardWrapper(base_env, mode=REWARD_MODE)

        train(gymenv, args.sb3_algo)

    if (args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, args.sb3_algo, path_to_model=args.test)

        else:
            print(f'{args.test} not found')
