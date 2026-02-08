# CARLA Closed-Loop (NuScenes-style)

This script runs CARLA 0.9.16 in synchronous mode and executes a closed-loop control/inference cycle with Epona. You can either start CARLA manually or let the script launch a Docker container.

## Key behavior
- CARLA sync mode, fixed delta `0.02` seconds.
- Capture ego pose + front camera every `0.1` seconds (10 Hz).
- First 1 second: Traffic Manager controls the ego to build the initial history.
- Then: run Epona inference on the last 1 second of data, warp ego along predicted trajectory for 1 second while rendering, repeat.
- No other vehicles.
- Traffic lights randomized each second.

## Usage

Manual CARLA:
```bash
python scripts/carla/closed_loop.py \
  --resume-path pretrained/epona_nuscenes.pkl \
  --config configs/dit_config_dcae_nuscenes.py
```

Docker CARLA:
```bash
python scripts/carla/closed_loop.py \
  --use-docker \
  --gpu \
  --render-offscreen \
  --resume-path pretrained/epona_nuscenes.pkl \
  --config configs/dit_config_dcae_nuscenes.py
```

Optional debug dump:

```bash
python scripts/carla/closed_loop.py \
  --use-docker \
  --gpu \
  --render-offscreen \
  --resume-path pretrained/epona_nuscenes.pkl \
  --dump-dir results/carla_loop
```

## Exact NuScenes calibration
If you have the nuScenes dataset locally, you can load the **exact** CAM_FRONT calibration (intrinsics + extrinsics) from the dataset and apply it automatically:

```bash
python scripts/carla/closed_loop.py \
  --use-docker \
  --gpu \
  --render-offscreen \
  --resume-path pretrained/epona_nuscenes.pkl \
  --nuscenes-dataroot /path/to/nuscenes \
  --nuscenes-version v1.0-trainval
```

This overrides `--cam-*` with the calibrated sensor values from `calibrated_sensor` and the camera resolution from `sample_data` for `CAM_FRONT`.
If you prefer to keep axes as-is (no Y flip), add `--no-nuscenes-axis-flip`.
Requires `nuscenes-devkit` installed in the Python environment.

## Default camera (approx)
- Resolution: `1600x900`
- FOV: `64.56` (computed from fx and width)
- Pose: `(x=1.5, y=0.0, z=1.5)` relative to ego

## Notes
- `--tm-speed-diff` slows the ego during warmup so relative motion stays within the model bounds.
- If your CARLA server runs on a different machine, set `--host` and `--port`.
- Docker mode uses `network_mode=host` (like gsedit) and `runtime=nvidia` when `--gpu` is set.
- The model expects NuScenes-style inputs (`image_size=(512,1024)`, `condition_frames=10`).
