import os
import json
import numpy as np
import torch

from torch.utils.data import Dataset
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets.pipelines import Compose


# ============================================================
# Geometry helpers (DO NOT register)
# ============================================================

def build_intrinsic_from_fov(image_size, fov_deg):
    """
    image_size: [W, H] | (W, H) | {'width': W, 'height': H}
    fov_deg: float
    """
    # ---- normalize image_size ----
    if isinstance(image_size, dict):
        W = image_size.get('width')
        H = image_size.get('height')
    else:
        W, H = image_size[:2]

    W = float(W)
    H = float(H)
    fov = float(fov_deg)

    fx = fy = 0.5 * W / np.tan(np.deg2rad(fov / 2.0))
    cx = W / 2.0
    cy = H / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K

def carla_pose_to_matrix(loc, rot):
    """
    CARLA camera pose -> 4x4 T_cam2ego
    rot: [roll, pitch, yaw] in degrees
    """
    x, y, z = loc
    roll, pitch, yaw = np.deg2rad(rot)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


# ============================================================
# Dataset
# ============================================================

@DATASETS.register_module()
class CarlaDataset(Dataset):

    def __init__(self,
                 data_root,
                 pipeline,
                 queue_length=3,
                 bev_size=(50, 50),
                 test_mode=True,
                 camera_names=None,
                 scene_token='carla_scene_0001',
                 **kwargs):

        self.data_root = data_root
        self.pipeline = Compose(pipeline)

        self.queue_length = int(queue_length)
        self.bev_size = tuple(bev_size)
        self.test_mode = bool(test_mode)
        self.scene_token = str(scene_token)

        self.camera_names = camera_names or [
            'cam_front',
            'cam_front_left',
            'cam_front_right',
            'cam_left',
            'cam_right',
            'cam_back',
        ]

        self.samples, self.timestamps = self._load_samples_and_timestamps()
        self.camera_system = self._load_camera_system()
        self.ego_motion = self._load_ego_motion()

        assert 'cams' in self.camera_system
        assert 'image_size' in self.camera_system

        self.cams = self.camera_system['cams']

        # align frames vs ego motion
        if len(self.samples) == len(self.ego_motion) + 1:
            self.samples = self.samples[1:]
            self.timestamps = self.timestamps[1:]
        elif len(self.samples) != len(self.ego_motion):
            raise ValueError('Frame / ego_motion length mismatch')

        missing = [c for c in self.camera_names if c not in self.cams]
        if missing:
            raise KeyError(f'Missing cameras: {missing}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        queue = []
        for i in range(idx - self.queue_length + 1, idx + 1):
            i = max(i, 0)
            queue.append(self._prepare_single(i))
        return self._union2one(queue)

    def _prepare_single(self, index):
        frame_id = self.samples[index]

        img_filenames = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam in self.camera_names:
            img_path = os.path.join(self.data_root, cam, f'{frame_id}.png')
            img_filenames.append(img_path)

            cam_info = self.cams[cam]

            # -------- intrinsic --------
            K = build_intrinsic_from_fov(
                image_size=self.camera_system['image_size'],
                fov_deg=cam_info.get('fov', self.camera_system.get('fov_deg', 90))
            )

            # -------- extrinsic (关键修正点) --------
            # -------- extrinsic from CARLA pose --------
            if 'position_xyz' not in cam_info or 'rotation_rpy_deg' not in cam_info:
                raise KeyError(
                    f"Camera '{cam}' missing pose fields. "
                    f"Expected ['position_xyz', 'rotation_rpy_deg'], "
                    f"got {list(cam_info.keys())}"
                )

            T_cam2ego = carla_pose_to_matrix(
                cam_info['position_xyz'],
                cam_info['rotation_rpy_deg']
            )

            if T_cam2ego.shape != (4, 4):
                raise ValueError(f'Extrinsic for {cam} must be 4x4, got {T_cam2ego.shape}')

            # ego -> cam
            T_ego2cam = np.linalg.inv(T_cam2ego).astype(np.float32)

            viewpad = np.eye(4, dtype=np.float32)
            viewpad[:3, :3] = K

            lidar2img = viewpad @ T_ego2cam
            lidar2img_rts.append(lidar2img)
            cam_intrinsics.append(viewpad)

        dx, dy, dyaw = self.ego_motion[index]

        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0] = float(dx)
        can_bus[1] = float(dy)
        dyaw_rad = float(dyaw)
        dyaw_deg = dyaw_rad / np.pi * 180.0
        can_bus[-2] = dyaw_rad
        can_bus[-1] = dyaw_deg

        input_dict = dict(
            img_filename=img_filenames,
            lidar2img=lidar2img_rts,
            cam_intrinsic=cam_intrinsics,
            can_bus=can_bus,
            scene_token=self.scene_token,
            frame_idx=index,
            timestamp=float(self.timestamps[index]),
            bev_size=self.bev_size,
        )

        self._pre_pipeline(input_dict)
        return self.pipeline(input_dict)

    def _union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_list = []

        for i, each in enumerate(queue):
            meta = each['img_metas'].data
            meta['prev_bev_exists'] = (i != 0)
            metas_list.append(meta)

        queue[-1]['img'] = DC(
            torch.stack(imgs_list),
            cpu_only=False,
            stack=True
        )

        # ⚠️ 关键：img_metas 必须是 list
        queue[-1]['img_metas'] = DC(
            metas_list,
            cpu_only=True
        )

        return queue[-1]


    # ---------------- loaders ----------------

    def _load_samples_and_timestamps(self):
        ts_path = os.path.join(self.data_root, 'timestamps.txt')
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f'Missing {ts_path}')

        frame_ids = []
        timestamps = []

        with open(ts_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 支持 "frame timestamp" 或 "frame,timestamp"
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                else:
                    parts = line.split()

                if len(parts) < 2:
                    raise ValueError(f'Invalid timestamp line: "{line}"')

                frame_int = int(parts[0])          # ← 强制转 int
                timestamp = float(parts[1])

                frame_str = f"{frame_int:06d}"     # ← 关键：补 0

                frame_ids.append(frame_str)
                timestamps.append(timestamp)

        return frame_ids, timestamps
    
    def _load_camera_system(self):
        with open(os.path.join(self.data_root, 'camera_system.json')) as f:
            return json.load(f)

    def _load_ego_motion(self):
        path = os.path.join(self.data_root, 'ego_motion_gt', 'ego_motion_gt.txt')
        motions = []
        with open(path) as f:
            for line in f:
                p = line.strip().split()
                try:
                    motions.append([float(p[-3]), float(p[-2]), float(p[-1])])
                except:
                    continue
        return motions

    def _pre_pipeline(self, results):
        results.update(dict(
            img_prefix=None,
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[],
        ))
