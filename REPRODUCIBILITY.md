# Reproducibility Guide - Flappy Bird DQN (SB3)

Dokumen ini menjelaskan versi environment dan langkah reproducible untuk training DQN pada `FlappyBird-v0`.

## 1) Versi yang Direkomendasikan

### Python & Environment
- Python: **3.11.9**
- Environment type: **venv** (stdlib Python)
- pip (direkomendasikan): **24.3.1** atau versi terbaru stabil pada Python 3.11
- OS target utama: **Windows 10/11 (x64)**

### Library (Pinned)
Sesuai `requirements.txt`:
- flappy-bird-gymnasium==0.4.0
- gymnasium==0.29.1
- stable-baselines3==2.3.2
- torch==2.3.1
- numpy==1.26.4
- pygame==2.5.2
- tensorboard==2.16.2
- python-dotenv==1.0.1
- imageio==2.34.1
- imageio-ffmpeg==0.5.1

## 2) Kenapa versi dipin?

- Menghindari mismatch API antar versi gymnasium/SB3.
- Menjaga perilaku training lebih konsisten antar waktu.
- Mempermudah debugging saat hasil berubah.

## 3) Langkah Reproduce dari Nol

Masuk ke folder ini, lalu jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup_env.ps1
Copy-Item .env.example .env
.\scripts\train_until_first_checkpoint.ps1
.\scripts\render_best_mp4.ps1
```

## 3.1) Struktur Output Wajib

Pipeline ini menyimpan output di folder berikut:
- `artifacts/checkpoints`: file checkpoint periodik (`dqn_ckpt_*.zip`)
- `artifacts/models`: `best_model.zip` dari EvalCallback + model final (`dqn_flappy_model.zip`)
- `artifacts/metrics`: `monitor.csv`, `evaluations.npz`, `eval_summary.json`
- `artifacts/videos`: hasil render MP4

## 4) Kontrol Konfigurasi Training

Semua parameter utama dapat diubah dari `.env`:
- Environment: `ENV_ID`, `USE_LIDAR`, `SEED`
- Training: `TOTAL_TIMESTEPS`, `LEARNING_RATE`, `BUFFER_SIZE`, dll
- Callback: `CHECKPOINT_FREQ`, `EVAL_FREQ`, `N_EVAL_EPISODES`

Ini membuat eksperimen tetap terstruktur dan mudah dilacak.

## 5) Catat Versi Runtime Aktual

Setelah setup selesai, jalankan:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\export_versions.ps1
```

Simpan file `VERSIONS_RUNTIME.md` sebagai bukti versi aktual saat eksperimen dijalankan.

## 6) Faktor yang Tetap Bisa Menyebabkan Variasi

Walau seed sudah diatur, hasil RL tetap dapat sedikit berbeda karena:
- backend komputasi (CPU vs GPU)
- versi driver/CUDA/cuDNN
- urutan scheduling thread
- perbedaan hardware

Untuk eksperimen ilmiah, dokumentasikan juga:
- jenis CPU/GPU
- versi driver GPU
- apakah training CPU-only atau GPU
