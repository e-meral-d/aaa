import torch.utils.data as data
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np
import torch
import os
import cv2
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125',
                    'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id


class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
            data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}


class _DefaultVideoFrameTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size
        mean = torch.tensor(OPENAI_DATASET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(OPENAI_DATASET_STD, dtype=torch.float32).view(3, 1, 1)
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


class _DefaultVideoMaskTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, mask_img: Image.Image) -> torch.Tensor:
        mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.array(mask_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor


def default_video_transforms(image_size: int) -> Tuple[callable, callable]:
    return _DefaultVideoFrameTransform(image_size), _DefaultVideoMaskTransform(image_size)


def _sorted_image_paths(directory: Path, extensions: Tuple[str, ...]) -> List[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in extensions and not p.name.startswith("._")]
    )


def _ucsd_video_index(root: Path) -> List[Dict]:
    entries: List[Dict] = []
    ped_dirs = [root / "UCSDped1" / "Test", root / "UCSDped2" / "Test"]
    for test_root in ped_dirs:
        if not test_root.exists():
            continue
        for sequence_dir in sorted([p for p in test_root.iterdir() if p.is_dir() and not p.name.endswith("_gt")]):
            frame_paths = _sorted_image_paths(sequence_dir, (".bmp", ".png", ".jpg", ".jpeg", ".tif"))
            if not frame_paths:
                continue
            mask_dir = sequence_dir.parent / f"{sequence_dir.name}_gt"
            mask_paths: Optional[List[Path]] = None
            if mask_dir.exists():
                mask_paths = _sorted_image_paths(mask_dir, (".bmp", ".png", ".jpg", ".jpeg", ".tif"))
            entries.append(
                {
                    "dataset": sequence_dir.parent.parent.name,
                    "video_id": sequence_dir.name,
                    "frame_paths": frame_paths,
                    "mask_paths": mask_paths,
                    "mask_type": "image" if mask_paths is not None else "zeros",
                }
            )
    return entries


def _shanghaitech_video_index(root: Path) -> List[Dict]:
    entries: List[Dict] = []
    frame_root = root / "testing" / "frames"
    mask_root = root / "testing" / "test_pixel_mask"
    if not frame_root.exists() or not mask_root.exists():
        return entries
    for sequence_dir in sorted([p for p in frame_root.iterdir() if p.is_dir()]):
        frame_paths = _sorted_image_paths(sequence_dir, (".bmp", ".png", ".jpg", ".jpeg"))
        if not frame_paths:
            continue
        mask_path = mask_root / f"{sequence_dir.name}.npy"
        entries.append(
                {
                    "dataset": "ShanghaiTech",
                    "video_id": sequence_dir.name,
                    "frame_paths": frame_paths,
                    "mask_npy": mask_path if mask_path.exists() else None,
                    "mask_type": "npy" if mask_path.exists() else "zeros",
                }
            )
    return entries


def build_video_metadata(dataset_name: str, root: str) -> List[Dict]:
    dataset_name = dataset_name.lower()
    path_root = Path(root)
    if dataset_name in {"ucsd", "ucsd-ped", "ucsd_ped"}:
        return _ucsd_video_index(path_root)
    if dataset_name in {"shanghaitech", "shanghaitechcampus", "sht", "shttech"}:
        return _shanghaitech_video_index(path_root)
    raise ValueError(f"Unsupported video dataset: {dataset_name}")


class VideoAnomalyDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        root: str,
        transform=None,
        target_transform=None,
        clip_length: int = 16,
        stride: int = 8,
        image_size: int = 336,
        mode: str = "train",
        preload_masks: bool = False,
    ):
        self.dataset_name = dataset_name
        self.root = root
        if transform is None or target_transform is None:
            frame_transform, mask_transform = default_video_transforms(image_size)
            if transform is None:
                transform = frame_transform
            if target_transform is None:
                target_transform = mask_transform
        self.transform = transform
        self.target_transform = target_transform
        self.clip_length = clip_length
        self.stride = stride
        self.mode = mode
        self.preload_masks = preload_masks
        self.image_size = image_size

        self.video_entries = build_video_metadata(dataset_name, root)
        self.samples: List[Dict] = []
        self._video_cache: Dict[str, np.ndarray] = {}
        self._build_samples()

    def _build_samples(self):
        for video_info in self.video_entries:
            frame_paths: List[Path] = video_info["frame_paths"]
            num_frames = len(frame_paths)
            if num_frames < self.clip_length:
                continue
            mask_type = video_info["mask_type"]
            cache_key = video_info["video_id"]
            if mask_type == "npy" and self.preload_masks and video_info.get("mask_npy") is not None:
                array = np.load(video_info["mask_npy"])
                self._video_cache[cache_key] = array
            if mask_type == "image" and self.preload_masks and video_info.get("mask_paths") is not None:
                stacked = []
                for mask_path in video_info["mask_paths"]:
                    with Image.open(mask_path) as mask_img:
                        stacked.append(np.array(mask_img.convert("L"), dtype=np.uint8))
                self._video_cache[cache_key] = np.stack(stacked, axis=0)

            for start in range(0, num_frames - self.clip_length + 1, self.stride):
                end = start + self.clip_length
                self.samples.append(
                    {
                        "video_id": video_info["video_id"],
                        "dataset": video_info["dataset"],
                        "frame_paths": frame_paths[start:end],
                        "mask_info": {
                            "type": mask_type,
                            "paths": video_info.get("mask_paths"),
                            "npy": video_info.get("mask_npy"),
                            "cache_key": cache_key,
                            "start": start,
                            "end": end,
                        },
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip_frames(self, frame_paths: List[Path]) -> torch.Tensor:
        frames = []
        for frame_path in frame_paths:
            img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to read frame {frame_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            frame_tensor = self.transform(pil_img) if self.transform is not None else torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            frames.append(frame_tensor)
        return torch.stack(frames, dim=0)

    def _prepare_mask_tensor(self, mask_array: np.ndarray) -> torch.Tensor:
        mask_frames: List[torch.Tensor] = []
        for mask_frame in mask_array:
            mask_img = Image.fromarray((mask_frame > 0).astype(np.uint8) * 255, mode="L")
            if self.target_transform is not None:
                mask_tensor = self.target_transform(mask_img)
            else:
                mask_tensor = torch.from_numpy((np.array(mask_img) > 0).astype(np.float32)).unsqueeze(0)
            mask_frames.append(mask_tensor)
        return torch.stack(mask_frames, dim=0)

    def _load_clip_masks(self, mask_info: Dict, spatial_size: Tuple[int, int]) -> torch.Tensor:
        mask_type = mask_info["type"]
        start, end = mask_info["start"], mask_info["end"]
        length = end - start

        if mask_type == "zeros":
            return torch.zeros(length, 1, spatial_size[0], spatial_size[1], dtype=torch.float32)

        if mask_type == "image":
            masks: List[torch.Tensor] = []
            mask_paths: Optional[List[Path]] = mask_info.get("paths")
            if not mask_paths:
                return torch.zeros(length, 1, spatial_size[0], spatial_size[1], dtype=torch.float32)
            for mask_path in mask_paths[start:end]:
                mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_arr is None:
                    raise RuntimeError(f"Failed to read mask {mask_path}")
                mask_arr = (mask_arr > 0).astype(np.uint8) * 255
                if self.target_transform is not None:
                    mask_img = Image.fromarray(mask_arr, mode="L")
                    mask_tensor = self.target_transform(mask_img)
                else:
                    mask_tensor = torch.from_numpy(mask_arr.astype(np.float32) / 255.0).unsqueeze(0)
                mask_tensor = (mask_tensor > 0.5).float()
                masks.append(mask_tensor)
            return torch.stack(masks, dim=0)

        if mask_type == "npy":
            cache_key = mask_info.get("cache_key")
            if cache_key in self._video_cache:
                mask_array = self._video_cache[cache_key][start:end]
            else:
                npy_path = mask_info.get("npy")
                if npy_path is None or not Path(npy_path).exists():
                    return torch.zeros(length, 1, spatial_size[0], spatial_size[1], dtype=torch.float32)
                mask_array = np.load(npy_path, mmap_mode="r")[start:end]
            return self._prepare_mask_tensor(mask_array)

        raise ValueError(f"Unsupported mask type: {mask_type}")

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        frame_paths = sample["frame_paths"]
        frames_tensor = self._load_clip_frames(frame_paths)

        mask_info = sample["mask_info"]
        spatial_size = (frames_tensor.shape[-2], frames_tensor.shape[-1])
        mask_tensor = self._load_clip_masks(mask_info, spatial_size)

        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)
        if mask_tensor.shape[-2] != frames_tensor.shape[-2] or mask_tensor.shape[-1] != frames_tensor.shape[-1]:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.float(),
                size=(frames_tensor.shape[-2], frames_tensor.shape[-1]),
                mode="nearest",
            )

        anomaly_label = 1 if mask_tensor.sum() > 0 else 0

        return {
            "clip": frames_tensor,
            "mask": mask_tensor,
            "video_id": sample["video_id"],
            "dataset": sample["dataset"],
            "frame_indices": (mask_info["start"], mask_info["end"]),
            "anomaly": anomaly_label,
        }
