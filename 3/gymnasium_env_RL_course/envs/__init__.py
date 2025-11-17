from gymnasium.envs.registration import register

register(
    id='gymnasium_env_RL_course/CartPoleEnv-v0',
    entry_point='gymnasium_env_RL_course.envs.cartpole_env:CartPoleEnv',
)