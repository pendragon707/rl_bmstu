import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import imageio
from pyvirtualdisplay import Display

project_root = os.path.dirname(os.path.abspath(__file__))  # Путь к корню проекта для импорта DQN 
sys.path.append(project_root)
from gymnasium_env_RL_course.envs.cartpole_DQN_train import DQN 

gif_filename = 'cartpole_test_DQN.gif'
fps = 15
# Параметры
# ENV_NAME = 'CartPole-v1'  # оригинальна среда
ENV_NAME = 'gymnasium_env_RL_course/CartPoleEnv-v0' # кастомная среда
STATE_DIM = 4  # позиции и скорости тележки и палки
ACTION_DIM = 2  # втолкаем вправо или влево
MAX_STEPS = 500  # лимит шагов, чтобы избежать зависания среды при успешном агенте

# Создание сети
policy_net = DQN(STATE_DIM, ACTION_DIM) # создаем такой же DQN как при обучении
policy_net.eval() # режим оценки

# Загрузка модели, обученной ранее
checkpoint = torch.load('dqn_model.pth', map_location=torch.device('cpu'), weights_only=False) # cpu, с метаданными

policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
print("Модель загружена")

# Создание среды
env = gym.make(ENV_NAME, render_mode='rgb_array') # rgb array позволяет собирать кадры для gif
# env = NormalizeObservation(env)

# Функция для выбора действия без epsilon
def select_action(state, policy_net):
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0) # добавляем размерность батча
        q_values = policy_net(state)
        return q_values.argmax().item()

# Оценка модели
with Display(visible=0, size=(400, 300)):  # Виртуальный дисплей для фикса рендеринга в колабе

    NUM_EVAL_EPISODES = 1
    rewards = []
    all_frames = []
    for episode in range(NUM_EVAL_EPISODES):
        state, _ = env.reset() # сброс среды
        total_reward = 0
        done = False
        step_count = 0  #счётчик шагов для truncated
        while not done and step_count < MAX_STEPS:
            action = select_action(state, policy_net)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step_count += 1
            img = env.render()
            if isinstance(img, np.ndarray):
                all_frames.append(img[:, :, :3])  # RGB
        rewards.append(total_reward)
        print(f"Оценка эпизода {episode + 1}: Награда = {total_reward}")

    mean_reward = sum(rewards) / len(rewards)
    print(f"Средняя награда за {NUM_EVAL_EPISODES} эпизодов: {mean_reward:.2f}")

    imageio.mimsave(gif_filename, all_frames, fps=fps)
    print(f"GIF сохранён: {gif_filename}")

env.close()

