import os
import random
from pathlib import Path

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401; registers FlappyBird-v0
import numpy as np
import torch
import json
from dotenv import load_dotenv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _print_runtime_versions() -> None:
    import stable_baselines3 as sb3

    print("=== Runtime Versions ===")
    print(f"Python           : {os.sys.version.split()[0]}")
    print(f"pip              : run `pip --version` in shell")
    flappy_ver = getattr(flappy_bird_gymnasium, "__version__", "unknown")
    print(f"flappy-bird-gym  : {flappy_ver}")
    print(f"gymnasium        : {gym.__version__}")
    print(f"stable-baselines3: {sb3.__version__}")
    print(f"torch            : {torch.__version__}")
    print(f"numpy            : {np.__version__}")
    print("========================")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    load_dotenv()

    env_id = os.getenv("ENV_ID", "FlappyBird-v0")
    use_lidar = _as_bool(os.getenv("USE_LIDAR"), default=True)
    seed = _as_int(os.getenv("SEED"), default=42)

    logdir = Path(os.getenv("LOGDIR", "logs/dqn_flappy"))
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "artifacts/checkpoints"))
    model_dir = Path(os.getenv("MODEL_DIR", "artifacts/models"))
    metrics_dir = Path(os.getenv("METRICS_DIR", "artifacts/metrics"))

    for folder in [logdir, checkpoint_dir, model_dir, metrics_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    total_timesteps = _as_int(os.getenv("TOTAL_TIMESTEPS"), default=3_000_000)
    learning_rate = _as_float(os.getenv("LEARNING_RATE"), default=1e-4)
    buffer_size = _as_int(os.getenv("BUFFER_SIZE"), default=50_000)
    batch_size = _as_int(os.getenv("BATCH_SIZE"), default=64)
    learning_starts = _as_int(os.getenv("LEARNING_STARTS"), default=1_000)
    train_freq = _as_int(os.getenv("TRAIN_FREQ"), default=4)
    target_update_interval = _as_int(os.getenv("TARGET_UPDATE_INTERVAL"), default=1_000)
    gamma = _as_float(os.getenv("GAMMA"), default=0.99)

    checkpoint_freq = _as_int(os.getenv("CHECKPOINT_FREQ"), default=50_000)
    eval_freq = _as_int(os.getenv("EVAL_FREQ"), default=25_000)
    n_eval_episodes = _as_int(os.getenv("N_EVAL_EPISODES"), default=10)
    model_name = os.getenv("MODEL_NAME", "dqn_flappy_model")

    set_global_seed(seed)
    _print_runtime_versions()

    def make_env():
        env = gym.make(env_id, render_mode=None, use_lidar=use_lidar)
        env.reset(seed=seed)
        return Monitor(env, filename=str(metrics_dir / "monitor.csv"))

    vec_env = DummyVecEnv([make_env])
    eval_env_cb = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="dqn_ckpt",
    )
    eval_callback = EvalCallback(
        eval_env_cb,
        best_model_save_path=str(model_dir),
        log_path=str(metrics_dir),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        gamma=gamma,
        tensorboard_log=str(logdir),
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="dqn_flappy",
        callback=[checkpoint_callback, eval_callback],
    )

    model_path = model_dir / model_name
    model.save(str(model_path))
    print(f"Saved model to {model_path}")

    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    print(f"Evaluation mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_summary = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "n_eval_episodes": int(n_eval_episodes),
        "total_timesteps": int(total_timesteps),
        "seed": int(seed),
        "env_id": env_id,
        "use_lidar": bool(use_lidar),
    }
    with (metrics_dir / "eval_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(eval_summary, fp, indent=2)

    print(f"Saved evaluation summary to {metrics_dir / 'eval_summary.json'}")
    eval_env.close()

