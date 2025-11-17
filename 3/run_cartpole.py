import gymnasium_env_RL_course
import gymnasium as gym
import imageio

gif_filename = 'cartpole_test.gif'

env_id = 'gymnasium_env_RL_course/CartPoleEnv-v0'
num_steps = 100
render_mode = 'rgb_array'

env = gym.make('gymnasium_env_RL_course/CartPoleEnv-v0', render_mode=render_mode)

obs, info = env.reset()
total_reward = 0
all_frames = []

for step in range(num_steps):
    action = env.action_space.sample() # рандомное действие
    # action = int(input("Action (0=left, 1=right): ")) # 0 - влево, 1 - вправо

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    frame = env.render()
    all_frames.append(frame)

    if terminated or truncated:
        print(f"Эпизод завершен, общая награда: {total_reward}")
        break

env.close()

if all_frames:
    imageio.mimsave(gif_filename, all_frames, duration=0.05, loop=0)
    print(f"GIF сохранен: {gif_filename}")