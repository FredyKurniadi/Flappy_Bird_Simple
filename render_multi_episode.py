import os
from pathlib import Path

import flappy_bird_gymnasium  # noqa: F401; registers FlappyBird-v0
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pygame
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
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


def _render_frame_native_lidar(env: gym.Env, use_lidar: bool) -> np.ndarray:
    """Capture frame using the library's own lidar drawing pipeline."""
    unwrapped = env.unwrapped
    unwrapped._draw_surface(show_score=False, show_rays=use_lidar)
    return np.transpose(pygame.surfarray.array3d(unwrapped._surface), axes=(1, 0, 2))


def _overlay_reward_text(
    frame: np.ndarray,
    episode_index: int,
    step_index: int,
    reward_t: float,
    reward_sum: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    lines = [
        f"Episode: {episode_index + 1}",
        f"Step: {step_index}",
        f"reward_t: {reward_t:.3f}",
        f"reward_sum: {reward_sum:.3f}",
    ]

    x = 8
    y = 8
    padding = 4
    line_height = 14
    max_text_width = 0

    for line in lines:
        left, _, right, _ = draw.textbbox((0, 0), line, font=font)
        max_text_width = max(max_text_width, right - left)

    box_right = x + max_text_width + (2 * padding)
    box_bottom = y + (line_height * len(lines)) + (2 * padding)
    draw.rectangle([x, y, box_right, box_bottom], fill=(0, 0, 0))

    text_y = y + padding
    for line in lines:
        draw.text((x + padding, text_y), line, fill=(255, 255, 255), font=font)
        text_y += line_height

    return np.asarray(image)


if __name__ == "__main__":
    load_dotenv()

    env_id = os.getenv("ENV_ID", "FlappyBird-v0")
    use_lidar = _as_bool(os.getenv("USE_LIDAR"), default=True)
    seed = _as_int(os.getenv("SEED"), default=42)

    model_name = os.getenv("MODEL_NAME", "dqn_flappy_model")
    model_path_override = os.getenv("MODEL_PATH")

    video_fps = _as_int(os.getenv("VIDEO_FPS"), default=30)
    max_render_steps = _as_int(os.getenv("MAX_RENDER_STEPS"), default=5000)
    render_episodes = _as_int(os.getenv("RENDER_EPISODES"), default=3)
    separator_frames = _as_int(os.getenv("SEPARATOR_FRAMES"), default=12)
    output_video_name = os.getenv("OUTPUT_VIDEO_NAME", "flappy_multi_episode.mp4")

    model_dir = Path(os.getenv("MODEL_DIR", "artifacts/models"))
    video_dir = Path(os.getenv("VIDEO_DIR", "artifacts/videos"))
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

    all_frames: list[np.ndarray] = []
    black_frame: np.ndarray | None = None

    for episode_idx in range(render_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        reward_sum = 0.0

        frame = _render_frame_native_lidar(env, use_lidar=use_lidar)
        all_frames.append(
            _overlay_reward_text(
                frame=frame,
                episode_index=episode_idx,
                step_index=0,
                reward_t=0.0,
                reward_sum=reward_sum,
            )
        )
        if black_frame is None:
            black_frame = np.zeros_like(frame)

        for step_idx in range(1, max_render_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_t, terminated, truncated, _ = env.step(action)
            reward_sum += float(reward_t)

            frame = _render_frame_native_lidar(env, use_lidar=use_lidar)
            all_frames.append(
                _overlay_reward_text(
                    frame=frame,
                    episode_index=episode_idx,
                    step_index=step_idx,
                    reward_t=float(reward_t),
                    reward_sum=reward_sum,
                )
            )

            if terminated or truncated:
                break

        if episode_idx < render_episodes - 1 and black_frame is not None:
            for _ in range(separator_frames):
                all_frames.append(black_frame)

    env.close()

    if not all_frames:
        raise RuntimeError("No frames were captured during rendering.")

    output_path = video_dir / output_video_name
    imageio.mimsave(output_path, all_frames, fps=video_fps)
    print(f"Saved rendered multi-episode video to {output_path}")
