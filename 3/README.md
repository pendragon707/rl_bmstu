# Введение в gymnasium. Создание кастомной среды и обучение DQN-агента

---
#### Сегодня мы:
- Создадим кастомную среду в Gymnasium, аналогичную классической CartPole (обратный маятник)
- Обучим агента с помощью Deep Q-Network (DQN)
- Увидим, как гиперпараметры влияют на обучение модели

Официальный гайд на создание кастомных сред в Gymnasium - https://gymnasium.farama.org/introduction/create_custom_env/

## Устанавливаем необходимые библиотеки

```
!pip install gymnasium
!pip install pyvirtualdisplay # нужен для рендеринга в колабе
```

Мы будем создавать среду, идентичную CartPole

Структура проекта:


```
dima@dima-TBk-14-G7-AHP:~/projects/RL_course_2$ tree
.
├── gymnasium_env_RL_course
│   ├── envs
│   │   ├── cartpole_DQN_train.py
│   │   ├── cartpole_env.py
│   │   ├── __init__.py
│   ├── __init__.py
├── pyproject.toml
├── run_DQN_cartpole.py
├── setup.py

```

Более подробные инструкции смотрите в ноутбуке [`3 - custom_env.ipynb`](https://colab.research.google.com/drive/1GeQuKU8kd89jakw9A1feJC59C-PROFY5?usp=sharing)