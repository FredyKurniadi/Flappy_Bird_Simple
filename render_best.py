import os
from pathlib import Path

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401; registers FlappyBird-v0
import imageio.v2 as imageio
from dotenv import load_dotenv
from stable_baselines3 import DQN


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_model_path(model_dir: Path, model_name: str, model_path_override: str | None) -> Path:
    if model_path_override:
        override = Path(model_path_override)
        if override.exists():
            return override
        raise FileNotFoundError(f"Custom model path not found: {override}")

    best_model = model_dir / "best_model.zip"
    final_model = model_dir / f"{model_name}.zip"

    if best_model.exists():
        return best_model
    if final_model.exists():
        return final_model

    raise FileNotFoundError(
        "No model file found. Expected best_model.zip or final model zip in artifacts/models."
    )


if __name__ == "__main__":
    load_dotenv()

    env_id = os.getenv("ENV_ID", "FlappyBird-v0")
    use_lidar = _as_bool(os.getenv("USE_LIDAR"), default=True)
    seed = _as_int(os.getenv("SEED"), default=42)
    model_name = os.getenv("MODEL_NAME", "dqn_flappy_model")
    model_path_override = os.getenv("MODEL_PATH")
    output_video_name = os.getenv("OUTPUT_VIDEO_NAME", "flappy_best_model.mp4")

    model_dir = Path(os.getenv("MODEL_DIR", "artifacts/models"))
    video_dir = Path(os.getenv("VIDEO_DIR", "artifacts/videos"))
    video_fps = _as_int(os.getenv("VIDEO_FPS"), default=30)
    max_render_steps = _as_int(os.getenv("MAX_RENDER_STEPS"), default=5000)

    model_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    model_path = _resolve_model_path(
        model_dir=model_dir,
        model_name=model_name,
        model_path_override=model_path_override,
    )
    print(f"Loading model: {model_path}")
    model = DQN.load(str(model_path))

    env = gym.make(env_id, render_mode="rgb_array", use_lidar=use_lidar)
    obs, _ = env.reset(seed=seed)

    frames = []
    first_frame = env.render()
    if first_frame is not None:
        frames.append(first_frame)

    for _ in range(max_render_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    if not frames:
        raise RuntimeError("No frames were captured during rendering.")

    output_path = video_dir / output_video_name
    imageio.mimsave(output_path, frames, fps=video_fps)
    print(f"Saved rendered video to {output_path}")
