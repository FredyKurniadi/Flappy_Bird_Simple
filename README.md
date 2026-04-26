# FLAPPY_BIRD - DQN (Stable-Baselines3) Reproducible Setup

Folder ini dibuat khusus untuk training agent Flappy Bird menggunakan:
- `flappy-bird-gymnasium`
- `stable-baselines3` (DQN)

Tujuan utama: setup yang terdokumentasi dan bisa direproduksi ulang.

## Struktur

- `requirements.txt`: dependency yang dipin (pinned)
- `.env.example`: konfigurasi training yang bisa diubah tanpa edit kode
- `train_dqn_sb3.py`: script training DQN
- `render_best.py`: render gameplay ke MP4 dari model terbaik/final
- `render_multi_episode.py`: render beberapa episode ke satu MP4
- `scripts/setup_env.ps1`: setup environment otomatis (Windows PowerShell)
- `scripts/run_train.ps1`: jalankan training
- `scripts/train_until_first_checkpoint.ps1`: train sampai checkpoint pertama
- `scripts/render_best_mp4.ps1`: render MP4 dari model terbaik
- `scripts/render_select_model_mp4.ps1`: render MP4 dengan memilih model (best/final/checkpoint/custom)
- `scripts/export_versions.ps1`: ekspor versi runtime + package terpasang
- `REPRODUCIBILITY.md`: penjelasan detail versi, env, dan prosedur reproduce

Output eksperimen disimpan terstruktur di:
- `artifacts/checkpoints`
- `artifacts/models`
- `artifacts/metrics`
- `artifacts/videos`

## Quickstart (Windows PowerShell)

Jalankan dari folder `FLAPPY_BIRD`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup_env.ps1
Copy-Item .env.example .env
.\scripts\train_until_first_checkpoint.ps1
.\scripts\render_best_mp4.ps1
```

## Render Multi-Episode Dengan Memilih Model

Contoh render 5 episode dari checkpoint 3M:

```powershell
.\scripts\render_select_model_mp4.ps1 -ModelSource checkpoint -CheckpointSteps 3000000 -Episodes 5
```

Contoh render dari best model:

```powershell
.\scripts\render_select_model_mp4.ps1 -ModelSource best -Episodes 3 -OutputName flappy_best_3ep.mp4
```

## Verifikasi Versi Runtime

Setelah install dependency:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\export_versions.ps1
```

File `VERSIONS_RUNTIME.md` akan berisi:
- versi Python
- versi pip
- hasil `pip freeze --all`

## Catatan Reproducibility

- Gunakan versi Python dan package sesuai `REPRODUCIBILITY.md`.
- Default seed = 42 (bisa diubah lewat `.env`).
- Hasil training RL tetap bisa memiliki variasi kecil antar mesin/driver/CUDA.
