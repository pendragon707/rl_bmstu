import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class CartPoleEnv(gym.Env):
    """
    Кастомная среда CartPole (обратный маятник).
    Идентична CartPole-v1 (gymnasium)
    """
    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Параметры среды (идентичны CartPole-v1)
        self.gravity = 9.81     # Ускорение свободного падения
        self.masscart = 1.0     # Масса тележки (cart)
        self.masspole = 0.1     # Масса палки (pole)
        self.total_mass = self.masspole + self.masscart # Общая масса маятника
        self.length = 0.5       # Половина длины палки (центр масс в середине)
        self.polemass_length = self.masspole * self.length # Момент инерции палки
        self.force_mag = 10.0   # Сила, с которой мы толкаем тележку
        self.tau = 0.02         # Шаг времени (dt, дискретизация среды)
        self.kinematics_integrator = 'euler' # Метод аппроксимации производных

        # Границы состояний (переход в терминальное состояние)
        self.x_threshold = 2.4  # граница по горизонтальной оси
        self.theta_threshold_radians = 12 * 2 * np.pi / 360  # 12 градусов (в рад)

        # Пространства
        self.action_space = spaces.Discrete(2)  # дискретное для действий (0 влево, 1 вправо)
        self.observation_space = spaces.Box(    # непрерывное для состяния (в 2 раза больше для стабильности)
            # low=np.array([-self.x_threshold * 2, -np.inf, -self.theta_threshold_radians * 2, -np.inf]),
            # high=np.array([self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf]),
            # dtype=np.float32
            low=np.array([-self.x_threshold, -np.inf, -self.theta_threshold_radians, -np.inf]),
            high=np.array([self.x_threshold, np.inf, self.theta_threshold_radians, np.inf]),
            
        )

        # Внутреннее состояние
        self.state = None
        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}" # действие корректно
        assert self.state is not None, "Call reset() before step()" # состояние инициализировано

        x, x_dot, theta, theta_dot = self.state # извлечение текущего состояния

        # Наше действие - толкаем тележку право или влево
        force = self.force_mag if action == 1 else -self.force_mag

        # Физика среды (обратный маятник)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler': # расчёт нового состояния
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot) # обновление состояния

        # Проверка терминального состояния
        terminated = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )

        # Награда: +1 за каждый шаг пока эпизод не завершен
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print("Episode terminated!")
            self.steps_beyond_terminated += 1
            reward = 0.0

        truncated = False  # Без лимита шагов

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None): # сброс среды
        super().reset(seed=seed)
        # Инициализация близко к центру
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminated = None
        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"): # для визуализации
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            self._render_human()

    def _render_human(self):
        # Вывод положения в консоль
        x, _, theta, _ = self.state
        print(f"Cart position: {x:.2f}, Pole angle: {theta:.2f} rad")

    def _render_rgb_array(self):
        # Визуализация через matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.clear()
        
        ax.set_xlim(-self.x_threshold, self.x_threshold) # границы 
        ax.set_ylim(-1, 1)
        ax.axis('off') # убираем оси

        # Тележка (влияет только на визуализацию)
        cart_width, cart_height = 0.3, 0.1
        cart = Rectangle((self.state[0] - cart_width/2, -cart_height/2), cart_width, cart_height, color='blue')
        ax.add_patch(cart)

        # Палка
        pole_x = self.state[0]
        pole_y = 0
        pole_end_x = pole_x + self.length * np.sin(self.state[2])
        pole_end_y = pole_y + self.length * np.cos(self.state[2])
        ax.plot([pole_x, pole_end_x], [pole_y, pole_end_y], 'r-', linewidth=3)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return img[:, :, :3]  # RGB без alpha
