import argparse
import os
import queue
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import cv2
import math
import docker

import carla

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.config_utils import Config


def _normalize_angle_deg(angle_deg: float) -> float:
    rad = np.deg2rad(angle_deg)
    return np.rad2deg(np.arctan2(np.sin(rad), np.cos(rad)))


def _rel_poses_from_abs_xyyaw(poses: List[Tuple[float, float, float]]):
    """Compute relative poses in ego frame and relative yaw in degrees.

    poses: list of (x, y, yaw_deg) in world frame
    Returns rel_poses (N, 2) and rel_yaws (N, 1)
    """
    n = len(poses)
    rel_poses = np.zeros((n, 2), dtype=np.float32)
    rel_yaws = np.zeros((n, 1), dtype=np.float32)
    for i in range(1, n):
        x0, y0, yaw0 = poses[i - 1]
        x1, y1, yaw1 = poses[i]
        dx = x1 - x0
        dy = y1 - y0
        yaw0_rad = np.deg2rad(yaw0)
        cos_t = np.cos(-yaw0_rad)
        sin_t = np.sin(-yaw0_rad)
        rel_x = cos_t * dx - sin_t * dy
        rel_y = sin_t * dx + cos_t * dy
        rel_poses[i, 0] = rel_x
        rel_poses[i, 1] = rel_y
        rel_yaws[i, 0] = _normalize_angle_deg(yaw1 - yaw0)
    return rel_poses, rel_yaws


def _carla_image_to_rgb(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb


def _resize_and_normalize(images: List[np.ndarray], target_h: int, target_w: int) -> torch.Tensor:
    resized = []
    for img in images:
        img_rs = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(img_rs)
    imgs = np.stack(resized, axis=0).astype(np.float32) / 255.0
    imgs = (imgs - 0.5) * 2.0
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # [T, C, H, W]
    return imgs


@dataclass
class CarlaContainerConfig:
    image: str
    name: str
    rpc_port: int
    render_offscreen: bool
    nosound: bool
    use_gpu: bool


class CarlaContainer:
    def __init__(self, cfg: CarlaContainerConfig):
        self.cfg = cfg
        self.client = docker.from_env()
        self.container = None

    def start(self):
        existing = None
        try:
            existing = self.client.containers.get(self.cfg.name)
        except docker.errors.NotFound:
            existing = None

        if existing is not None:
            self.container = existing
            if existing.status != "running":
                existing.start()
            return

        try:
            self.client.images.pull(self.cfg.image)
        except Exception:
            pass

        cmd = ["bash", "CarlaUE4.sh"]
        if self.cfg.render_offscreen:
            cmd.append("-RenderOffScreen")
        if self.cfg.nosound:
            cmd.append("-nosound")
        cmd += [
            f"-carla-rpc-port={self.cfg.rpc_port}",
            f"-world-port={self.cfg.rpc_port}",
        ]

        env = None
        runtime = None
        if self.cfg.use_gpu:
            env = {
                "NVIDIA_VISIBLE_DEVICES": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
            }
            runtime = "nvidia"

        self.container = self.client.containers.run(
            self.cfg.image,
            cmd,
            name=self.cfg.name,
            detach=True,
            network_mode="host",
            environment=env,
            runtime=runtime,
        )

    def stop(self):
        if self.container is None:
            return
        try:
            self.container.stop(timeout=10)
        finally:
            try:
                self.container.remove(force=True)
            except Exception:
                pass


class CarlaClosedLoop:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.connect_timeout)
        self.world = None
        self.tm = None
        self.vehicle = None
        self.camera = None
        self.image_queue = queue.Queue()
        self._should_stop = False

    def setup_world(self):
        deadline = time.time() + self.args.connect_wait_seconds
        while True:
            try:
                if self.args.town:
                    self.world = self.client.load_world(self.args.town)
                else:
                    self.world = self.client.get_world()
                break
            except Exception as exc:
                if time.time() > deadline:
                    raise RuntimeError("Failed to connect to CARLA") from exc
                time.sleep(1.0)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.args.delta_seconds
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(self.args.tm_port)
        self.tm.set_synchronous_mode(True)

    def spawn_vehicle_and_camera(self):
        blueprints = self.world.get_blueprint_library()
        vehicle_bp = blueprints.find(self.args.vehicle)
        vehicle_bp.set_attribute("role_name", "ego")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found")
        self.vehicle = None
        random.shuffle(spawn_points)
        for spawn in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn)
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.cam_width))
        cam_bp.set_attribute("image_size_y", str(self.args.cam_height))
        cam_bp.set_attribute("fov", str(self.args.cam_fov))
        cam_bp.set_attribute("sensor_tick", "0.0")

        cam_transform = carla.Transform(
            carla.Location(x=self.args.cam_x, y=self.args.cam_y, z=self.args.cam_z),
            carla.Rotation(
                pitch=self.args.cam_pitch,
                yaw=self.args.cam_yaw,
                roll=self.args.cam_roll,
            ),
        )
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self.image_queue.put)

    def destroy_actors(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()

    def randomize_traffic_lights(self):
        lights = self.world.get_actors().filter("traffic.traffic_light*")
        for tl in lights:
            state = random.choice(
                [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow, carla.TrafficLightState.Green]
            )
            tl.set_state(state)
            tl.set_green_time(random.uniform(3.0, 10.0))
            tl.set_red_time(random.uniform(3.0, 10.0))
            tl.set_yellow_time(random.uniform(1.0, 3.0))

    def _tick_and_get_latest_image(self, ticks: int) -> carla.Image:
        last_img = None
        for _ in range(ticks):
            self.world.tick()
            while True:
                try:
                    last_img = self.image_queue.get_nowait()
                except queue.Empty:
                    break
        if last_img is None:
            last_img = self.image_queue.get(timeout=self.args.sensor_timeout)
        return last_img

    def _get_pose_xyyaw(self) -> Tuple[float, float, float]:
        transform = self.vehicle.get_transform()
        loc = transform.location
        yaw = transform.rotation.yaw
        return (loc.x, loc.y, yaw)

    def warmup_sequence(self) -> Tuple[List[np.ndarray], List[Tuple[float, float, float]]]:
        self.vehicle.set_autopilot(True, self.tm_port)
        self.vehicle.set_simulate_physics(True)
        self.tm.vehicle_percentage_speed_difference(self.vehicle, self.args.tm_speed_diff)

        poses = [self._get_pose_xyyaw()]
        images = []
        sample_ticks = int(round(self.args.sample_period / self.args.delta_seconds))
        for _ in range(self.args.condition_frames):
            img = self._tick_and_get_latest_image(sample_ticks)
            images.append(_carla_image_to_rgb(img))
            poses.append(self._get_pose_xyyaw())
        return images, poses

    @property
    def tm_port(self):
        return self.args.tm_port

    def run(self, model, tokenizer):
        self.randomize_traffic_lights()
        images, poses = self.warmup_sequence()

        step_index = 0
        start_wall = time.time()
        while not self._should_stop:
            if self.args.run_seconds > 0 and (time.time() - start_wall) >= self.args.run_seconds:
                break
            self.randomize_traffic_lights()

            rel_pose, rel_yaw = _rel_poses_from_abs_xyyaw(poses)
            imgs_tensor = _resize_and_normalize(
                images, self.args.image_size[0], self.args.image_size[1]
            ).unsqueeze(0).cuda()

            rel_pose_t = torch.from_numpy(rel_pose).unsqueeze(0).float().cuda()
            rel_yaw_t = torch.from_numpy(rel_yaw).unsqueeze(0).float().cuda()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                start_latents = tokenizer.encode_to_z(imgs_tensor)
                predict_traj, _ = model.step_eval(
                    start_latents, rel_pose_t, rel_yaw_t, self_pred_traj=True, traj_only=True
                )

            traj = predict_traj[0].cpu().numpy()
            if self.args.clamp_traj:
                traj[:, 0] = np.clip(traj[:, 0], 0.0, 8.0)
                traj[:, 1] = np.clip(traj[:, 1], -0.5, 0.5)
                traj[:, 2] = np.clip(traj[:, 2], -8.0, 8.0)

            self.vehicle.set_autopilot(False, self.tm_port)
            self.vehicle.set_simulate_physics(False)

            sample_ticks = int(round(self.args.sample_period / self.args.delta_seconds))
            horizon_steps = int(round(self.args.control_horizon / self.args.sample_period))
            horizon_steps = min(horizon_steps, traj.shape[0])

            new_images = []
            new_poses = [poses[-1]]
            cur_x, cur_y, cur_yaw = poses[-1]

            for i in range(horizon_steps):
                dx, dy, dyaw = traj[i, 0], traj[i, 1], traj[i, 2]
                cur_yaw_rad = np.deg2rad(cur_yaw)
                wx = cur_x + np.cos(cur_yaw_rad) * dx - np.sin(cur_yaw_rad) * dy
                wy = cur_y + np.sin(cur_yaw_rad) * dx + np.cos(cur_yaw_rad) * dy
                wyaw = _normalize_angle_deg(cur_yaw + dyaw)

                transform = carla.Transform(
                    carla.Location(x=float(wx), y=float(wy), z=self.vehicle.get_transform().location.z),
                    carla.Rotation(pitch=0.0, yaw=float(wyaw), roll=0.0),
                )
                self.vehicle.set_transform(transform)
                img = self._tick_and_get_latest_image(sample_ticks)
                new_images.append(_carla_image_to_rgb(img))
                new_poses.append((wx, wy, wyaw))

                cur_x, cur_y, cur_yaw = wx, wy, wyaw

            if self.args.dump_dir:
                self._dump_step(step_index, new_images, new_poses)

            images = new_images
            poses = new_poses
            step_index += 1

    def _dump_step(self, step_index: int, images, poses):
        import cv2
        os.makedirs(self.args.dump_dir, exist_ok=True)
        step_dir = os.path.join(self.args.dump_dir, f"step_{step_index:04d}")
        os.makedirs(step_dir, exist_ok=True)
        for i, img in enumerate(images):
            cv2.imwrite(os.path.join(step_dir, f"{i:03d}.png"), img[:, :, ::-1])
        np.save(os.path.join(step_dir, "poses.npy"), np.array(poses))

    def stop(self):
        self._should_stop = True


def build_model(args):
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))
    local_rank = 0
    model = TrainTransformersDiT(
        cfg,
        load_path=args.resume_path,
        local_rank=local_rank,
        condition_frames=cfg.condition_frames,
    )
    tokenizer = VAETokenizer(cfg, local_rank)
    return model, tokenizer, cfg


def _quat_to_rotmat(qw, qx, qy, qz):
    # nuScenes uses [w, x, y, z]
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = 2.0 / n if n > 0 else 0.0
    x, y, z = qx, qy, qz
    w = qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - s * (yy + zz), s * (xy - wz), s * (xz + wy)],
            [s * (xy + wz), 1.0 - s * (xx + zz), s * (yz - wx)],
            [s * (xz - wy), s * (yz + wx), 1.0 - s * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _rotmat_to_carla_rpy(R):
    # CARLA uses degrees: roll (x), pitch (y), yaw (z)
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def apply_nuscenes_calibration(args):
    if not args.nuscenes_dataroot:
        args.nuscenes_dataroot = os.getenv("NUSCENES_DATAROOT", "")
    if not args.nuscenes_dataroot:
        return args
    try:
        from nuscenes.nuscenes import NuScenes
    except Exception as exc:
        raise RuntimeError(
            "nuscenes-devkit is required when --nuscenes-dataroot is set"
        ) from exc

    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    sample = nusc.sample[0]
    cam_token = sample["data"][args.nuscenes_camera]
    cam_data = nusc.get("sample_data", cam_token)
    calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

    K = np.array(calib["camera_intrinsic"], dtype=np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    width = int(cam_data["width"])
    height = int(cam_data["height"])
    fov = 2.0 * math.degrees(math.atan(width / (2.0 * fx)))

    # nuScenes ego frame is commonly treated as x-forward, y-left, z-up.
    # CARLA uses x-forward, y-right, z-up. We flip y by default.
    if args.no_nuscenes_axis_flip:
        S = np.eye(3, dtype=np.float32)
    else:
        S = np.diag([1.0, -1.0, 1.0]).astype(np.float32)

    t_nu = np.array(calib["translation"], dtype=np.float32)
    t_c = (S @ t_nu.reshape(3, 1)).reshape(3)

    qw, qx, qy, qz = calib["rotation"]
    R_nu = _quat_to_rotmat(qw, qx, qy, qz)
    R_c = S @ R_nu @ S
    roll, pitch, yaw = _rotmat_to_carla_rpy(R_c)

    args.cam_width = width
    args.cam_height = height
    args.cam_fov = fov
    args.cam_x, args.cam_y, args.cam_z = float(t_c[0]), float(t_c[1]), float(t_c[2])
    args.cam_roll, args.cam_pitch, args.cam_yaw = float(roll), float(pitch), float(yaw)

    args.nuscenes_intrinsic = (fx, fy, cx, cy)
    return args


def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use-docker", action="store_true")
    parser.add_argument("--carla-image", default="carlasim/carla:0.9.16")
    parser.add_argument("--carla-name", default="carla_container")
    parser.add_argument("--render-offscreen", action="store_true")
    parser.add_argument("--nosound", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--connect-wait-seconds", type=float, default=60.0)
    parser.add_argument("--town", default="Town03")

    # Simulation timing
    parser.add_argument("--delta-seconds", type=float, default=0.02)
    parser.add_argument("--sample-period", type=float, default=0.1)
    parser.add_argument("--control-horizon", type=float, default=1.0)

    # Vehicle / traffic
    parser.add_argument("--vehicle", default="vehicle.tesla.model3")
    parser.add_argument("--tm-speed-diff", type=float, default=50.0)

    # Camera (nuScenes-like)
    parser.add_argument("--cam-width", type=int, default=1600)
    parser.add_argument("--cam-height", type=int, default=900)
    parser.add_argument("--cam-fov", type=float, default=64.56)
    parser.add_argument("--cam-x", type=float, default=1.5)
    parser.add_argument("--cam-y", type=float, default=0.0)
    parser.add_argument("--cam-z", type=float, default=1.5)
    parser.add_argument("--cam-pitch", type=float, default=0.0)
    parser.add_argument("--cam-yaw", type=float, default=0.0)
    parser.add_argument("--cam-roll", type=float, default=0.0)

    # Epona model
    parser.add_argument("--config", default="configs/dit_config_dcae_nuscenes.py")
    parser.add_argument("--resume-path", required=True)
    parser.add_argument("--condition-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--batch-size", type=int, default=1)

    # Debug
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--clamp-traj", action="store_true")
    parser.add_argument("--sensor-timeout", type=float, default=2.0)
    parser.add_argument("--run-seconds", type=float, default=20.0)

    # nuScenes calibration
    parser.add_argument("--nuscenes-dataroot", default="")
    parser.add_argument("--nuscenes-version", default="v1.0-mini")
    parser.add_argument("--nuscenes-camera", default="CAM_FRONT")
    parser.add_argument("--no-nuscenes-axis-flip", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = add_arguments()
    args.dump_dir = args.dump_dir or None

    container = None
    if args.use_docker:
        cfg = CarlaContainerConfig(
            image=args.carla_image,
            name=args.carla_name,
            rpc_port=args.port,
            render_offscreen=args.render_offscreen,
            nosound=args.nosound,
            use_gpu=args.gpu,
        )
        container = CarlaContainer(cfg)
        container.start()
        time.sleep(5)

    args = apply_nuscenes_calibration(args)

    model, tokenizer, cfg = build_model(args)

    loop = CarlaClosedLoop(args)

    def _handle_sig(*_):
        loop.stop()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    try:
        loop.setup_world()
        loop.spawn_vehicle_and_camera()
        loop.run(model, tokenizer)
    finally:
        loop.destroy_actors()
        if container is not None:
            container.stop()
