# RL алгоритмы. Q-learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1IX7Ei5l5wXioHXB8A6u-Ri-_pKnjnO0y/view?usp=sharing)

## Установка зависимостей

    ```bash
    python -m venv env
    . env/bin/activate
    pip install -r requirements.txt
    ```

## Использование

### 1) Тренировка
- **Start Training:**
  - Введите команду:
    ```bash
    python sb_rl.py -a [EnvironmentName] -a [AlgorithmName] -t
    ```
  - Пример :  `python sb_rl.py -e Hopper-v5 -a SAC -n 1000000 -t`
 
- **Monitor Training:**
  - С помощью TensorBoard вы можете отслеживать процесс обучения в режиме реального времени.
    ```bash
    tensorboard --logdir [LogFolder]

    ```

### 3) Тестирование
  - Введите команду:
    ```bash
    python sb_rl.py -e [EnvironmentName] -a [AlgorithmName] -s [PathToModel]
    ```
  - Пример :  `python sb_rl.py -e Hopper-v5 -a SAC -s models/models_Hopper-v5/SAC_1000000.zip`

## Обучение робособаки
Скрипт `robodog.py`

Нужно скачать urdf или mjcf модель вашего робота. Можно использовать модели из [этого репозитория](https://github.com/google-deepmind/mujoco_menagerie.git), например. Загрузите репозиторий с моделями одним из этих двух способов:

`git submodule update --init --recursive`

`git clone https://github.com/google-deepmind/mujoco_menagerie.git`

Обучение и тренировка аналогично

`python robodog.py -a SAC -n 10000000 -t`

`python robodog.py -a SAC -s models/models_robodog/SAC_10000000.zip`
