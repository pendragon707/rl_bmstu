from gymnasium.envs.registration import register
from . import envs

register(
    id="gymnasium_env_RL_course/GridWorld-v0",
    entry_point="gymnasium_env_RL_course.envs:GridWorldEnv",
)
