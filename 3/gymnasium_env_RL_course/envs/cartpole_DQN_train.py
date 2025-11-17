import gymnasium as gym
import gymnasium_env_RL_course
from gymnasium.wrappers import NormalizeObservation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

seed = 40000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Параметры
# ENV_NAME = 'CartPole-v1'
ENV_NAME = 'gymnasium_env_RL_course/CartPoleEnv-v0'
EPISODES = 300  # > 200 Количество эпизодов обучения
MAX_STEPS = 500  # 200-1000 Лимит шагов в эпизоде
GAMMA = 0.99  # 0.9-0.999 Коэффициент дисконтирования
EPSILON_START = 1.0  # 0.8-1 Начальный epsilon
EPSILON_END = 0.01   # 0.001-0.05 Минимальный epsilon
EPSILON_DECAY = 0.95  # 0.9-0.999 Уменьшение epsilon
BATCH_SIZE = 128  # 32-256 Размер батча
LR = 0.002  # 0.0001-0.005 Скорость обучения
MEMORY_SIZE = 20000  # 10000-100000, Размер replay buffer
TARGET_UPDATE = 2  # 1-10 Частота обновления target network

# Определяем Q-сеть (нейросеть - аппроксиматор Q функции)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Буфер
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # очередь фикс размера

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Выбор действия 
def select_action(state, epsilon, policy_net, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1) # случайное действие с вероятностью epsilon
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item() # действие с максимальным Q

# Обучение
def train_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size: # достаточно ли данных
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    # Преобразование в тензоры
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    # Вычисляем Q-values для текущих состояний и отбираем те которые соотв действиям агента
    q_values = policy_net(states).gather(1, actions)

    # Q-values для следующих состояний (используем target_net)
    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Loss и обновление весов
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Основной код
if __name__ == "__main__":
    env = gym.make(ENV_NAME) # создание среды
    normalizaton = 'Off'
    # env, normalizaton = NormalizeObservation(env), 'On'
    state_dim = env.observation_space.shape[0] # инициализация состояния
    action_dim = env.action_space.n # инициализация действия

    policy_net = DQN(state_dim, action_dim) # инициализация сетей
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START # сброс
    rewards_history = []

    # В каждом эпизоде:
    for episode in range(EPISODES):
        state, _ = env.reset() # сброс
        total_reward = 0

        # Среда рассчитывается до завершения эпизода:
        for step in range(MAX_STEPS):
            action = select_action(state, epsilon, policy_net, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # терминальное состояние или завершение по времени

            replay_buffer.push(state, action, reward, next_state, done)
            train_step(policy_net, target_net, optimizer, replay_buffer, BATCH_SIZE, GAMMA)

            state = next_state
            total_reward += reward

            if done:
                break

        # Уменьшение epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Обновление target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)
        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()

    # Сохранение модели
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim,
        # Опционально:
        # 'epsilon': epsilon,
        # 'GAMMA': GAMMA,
        # 'LR': LR
    }, 'dqn_model.pth')  # Сохраняет в файл dqn_model.pth

    # Графики обучения
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, linewidth=1.25, color='b')
    plt.title('Награда за эпизод')
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    plt.grid(True)
    plt.savefig('training_DQN.png', dpi=300)
    # plt.show()

    # Скользящее среднее для сглаживания графика
    window_size = 30
    if len(rewards_history) >= window_size:
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_history, linewidth=1, color='b', alpha=0.5)
        plt.plot(moving_avg, linewidth=2, color='b')
        plt.title(f'Скользящее среднее наград (M = {window_size}),\n$\\gamma = ${GAMMA}, $\\epsilon_d$ = {EPSILON_DECAY}, $learning\ rate = ${LR}, Normalization = {normalizaton}, seed = {seed}')#gamma = {GAMMA}, $\\epsilon_decay$ = {EPSILON_DECAY}, $\\lr = ${LR}')
        plt.xlabel('Эпизод')
        plt.ylabel('Средняя награда')
        plt.grid(True)
        plt.savefig('training_DQN_RunMean.png', dpi=300)
        plt.show()
