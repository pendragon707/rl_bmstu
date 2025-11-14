import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, DQN, PPO
import os
import argparse

from pathlib import Path

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
    parser.add_argument('-x', '--xml_file', default='./mujoco_menagerie/unitree_go1/scene.xml', help='Путь к xml-модели робота')
    parser.add_argument('-a', '--sb3_algo', default="PPO", help='StableBaseline3 RL-алгоритмы (SAC, TD3, A2C, DQN, PPO)')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    model_dir = "models/models_robodog"
    log_dir = "logs/logs_robodog"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.train:
        gymenv = gym.make(
            'Ant-v5',
            #xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
            xml_file=args.xml_file,
            forward_reward_weight=1,
            ctrl_cost_weight=0.05,
            contact_cost_weight=5e-4,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
            max_episode_steps=1000,
            render_mode=None,
        )

        train(gymenv, args.sb3_algo)

    if (args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make(
                'Ant-v5',
                #xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
                xml_file=args.xml_file,
                forward_reward_weight=1,
                ctrl_cost_weight=0.05,
                contact_cost_weight=5e-4,
                healthy_reward=1,
                main_body=1,
                healthy_z_range=(0.195, 0.75),
                include_cfrc_ext_in_observation=True,
                exclude_current_positions_from_observation=False,
                reset_noise_scale=0.1,
                frame_skip=25,
                max_episode_steps=1000,
                render_mode='human',
            )

            test(gymenv, args.sb3_algo, path_to_model=args.test)

        else:

            print(f'{args.test} not found')
