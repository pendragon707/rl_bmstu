import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, DQN, PPO
import os
import argparse

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # We expect `render()` to return a uint8 array with values in [0, 255] or a float array
                # with values in [0, 1], as described in
                # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.from_numpy(np.asarray([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


def train(env, sb3_algo, timesteps = 25000, video_recorder = None):
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

        if video_recorder:
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=video_recorder)
        else:
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

    # Create directories to hold models and logs
    model_dir = "models/models_"+args.gymenv+"/"+args.sb3_algo
    log_dir = "logs/logs_"+args.gymenv
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.train:
        gymenv = gym.make(args.gymenv, render_mode=None)
        video_recorder = VideoRecorderCallback(gymenv, render_freq=5000)
        train(gymenv, args.sb3_algo)

    if (args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, args.sb3_algo, path_to_model=args.test)

        else:
            print(f'{args.test} not found')
