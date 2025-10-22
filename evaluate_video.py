import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import AnomalyCLIP_lib
from dataset import VideoAnomalyDataset
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_video_stats(dataset: VideoAnomalyDataset):
    stats = {}
    for entry in dataset.video_entries:
        stats[entry["video_id"]] = len(entry["frame_paths"])
    return stats


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def evaluate(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess, target_transform = get_transform(args)
    test_dataset = VideoAnomalyDataset(
        dataset_name=args.dataset,
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        clip_length=args.clip_length,
        stride=args.clip_stride_test,
        image_size=args.image_size,
        mode="test",
        preload_masks=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    video_lengths = build_video_stats(test_dataset)
    h = args.image_size
    w = args.image_size

    frame_score_sums = {vid: np.zeros(length, dtype=np.float64) for vid, length in video_lengths.items()}
    frame_score_counts = {vid: np.zeros(length, dtype=np.float64) for vid, length in video_lengths.items()}
    frame_labels = {vid: np.zeros(length, dtype=np.int32) for vid, length in video_lengths.items()}

    pixel_score_sums = {vid: np.zeros((length, h, w), dtype=np.float32) for vid, length in video_lengths.items()}
    pixel_score_counts = {vid: np.zeros(length, dtype=np.float32) for vid, length in video_lengths.items()}
    pixel_gt_max = {vid: np.zeros((length, h, w), dtype=np.uint8) for vid, length in video_lengths.items()}

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
        "temporal_adapter_cfg": {
            "enabled": args.use_temporal_adapter,
            "layers": args.temporal_adapter_layers,
            "num_heads": args.temporal_adapter_heads,
            "mlp_ratio": args.temporal_adapter_mlp_ratio,
            "dropout": args.temporal_adapter_dropout,
        } if args.use_temporal_adapter else {"enabled": False},
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    if args.use_temporal_adapter and getattr(model.visual, "temporal_adapter", None) is not None:
        temporal_state = checkpoint.get("temporal_adapter", None)
        if temporal_state is not None:
            model.visual.temporal_adapter.load_state_dict(temporal_state)
        model.visual.temporal_adapter.eval()

    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for batch in tqdm(test_loader, desc="Evaluating"):
            clips = batch["clip"].to(device)
            masks = batch["mask"].to(device)
            video_ids = batch["video_id"]
            frame_indices = batch["frame_indices"]

            if isinstance(video_ids, str):
                video_ids = [video_ids]
            if torch.is_tensor(frame_indices):
                frame_indices = [tuple(idx.tolist()) for idx in frame_indices]

            B, T, C, H, W = clips.shape
            frames = clips.view(B * T, C, H, W)
            temporal_meta = {"batch_size": B, "num_frames": T} if args.use_temporal_adapter else None

            if args.use_temporal_adapter:
                _, patch_features = model.encode_image(frames, args.features_list, DPAM_layer=20, temporal_adapter=temporal_meta)
            else:
                _, patch_features = model.encode_image(frames, args.features_list, DPAM_layer=20)

            anomaly_maps_layers = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / (patch_feature.norm(dim=-1, keepdim=True) + 1e-7)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    anomaly_map = (similarity_map[:, 1, :, :] + 1 - similarity_map[:, 0, :, :]) / 2.0
                    anomaly_maps_layers.append(anomaly_map)

            if len(anomaly_maps_layers) == 0:
                anomaly_maps = torch.zeros(B * T, H, W, device=device)
            else:
                anomaly_maps = torch.stack(anomaly_maps_layers).sum(dim=0)

            anomaly_maps = anomaly_maps.view(B, T, H, W)
            frame_scores = anomaly_maps.view(B, T, -1).max(dim=2).values

            for b in range(B):
                vid = video_ids[b]
                start, end = frame_indices[b]
                frame_range = slice(start, end)

                scores_np = frame_scores[b].cpu().numpy()
                maps_np = anomaly_maps[b].cpu().numpy()
                mask_np = masks[b, :, 0].cpu().numpy()

                frame_score_sums[vid][frame_range] += scores_np
                frame_score_counts[vid][frame_range] += 1

                pixel_score_sums[vid][frame_range] += maps_np
                pixel_score_counts[vid][frame_range] += 1
                pixel_gt_max[vid][frame_range] = np.maximum(pixel_gt_max[vid][frame_range], (mask_np > 0).astype(np.uint8))

                frame_labels_clip = (mask_np.reshape(mask_np.shape[0], -1).sum(axis=1) > 0).astype(np.int32)
                frame_labels[vid][frame_range] = np.maximum(frame_labels[vid][frame_range], frame_labels_clip)

    all_frame_scores = []
    all_frame_labels = []
    all_pixel_scores = []
    all_pixel_labels = []

    for vid in video_lengths.keys():
        counts = frame_score_counts[vid]
        counts[counts == 0] = 1
        avg_scores = frame_score_sums[vid] / counts
        all_frame_scores.extend(avg_scores.tolist())
        all_frame_labels.extend(frame_labels[vid].tolist())

        pixel_counts = pixel_score_counts[vid]
        pixel_counts[pixel_counts == 0] = 1
        avg_maps = pixel_score_sums[vid] / pixel_counts[:, None, None]
        gt_maps = (pixel_gt_max[vid] > 0).astype(np.uint8)

        all_pixel_scores.append(avg_maps.reshape(-1))
        all_pixel_labels.append(gt_maps.reshape(-1))

    all_pixel_scores = np.concatenate(all_pixel_scores, axis=0)
    all_pixel_labels = np.concatenate(all_pixel_labels, axis=0)

    frame_auc = float("nan")
    frame_ap = float("nan")
    if len(np.unique(all_frame_labels)) > 1:
        frame_auc = roc_auc_score(all_frame_labels, all_frame_scores)
        frame_ap = average_precision_score(all_frame_labels, all_frame_scores)

    pixel_auc = float("nan")
    pixel_ap = float("nan")
    if len(np.unique(all_pixel_labels)) > 1:
        pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_scores)
        pixel_ap = average_precision_score(all_pixel_labels, all_pixel_scores)

    ensure_dir(args.save_path)
    summary_path = os.path.join(args.save_path, f"evaluation_{args.dataset}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Frame-level AUC: {frame_auc:.4f}\n")
        f.write(f"Frame-level AP : {frame_ap:.4f}\n")
        f.write(f"Pixel-level AUC: {pixel_auc:.4f}\n")
        f.write(f"Pixel-level AP : {pixel_ap:.4f}\n")

    print(f"[Evaluation] Frame AUC: {frame_auc:.4f}, Frame AP: {frame_ap:.4f}")
    print(f"[Evaluation] Pixel AUC: {pixel_auc:.4f}, Pixel AP: {pixel_ap:.4f}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video Evaluation", add_help=True)
    parser.add_argument("--data_path", type=str, required=True, help="path to dataset root directory")
    parser.add_argument("--dataset", type=str, choices=["ucsd", "shanghaitechcampus"], required=True, help="dataset name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--save_path", type=str, default="./results", help="directory to save evaluation summaries")

    parser.add_argument("--clip_length", type=int, default=16, help="number of frames per clip")
    parser.add_argument("--clip_stride_test", type=int, default=4, help="temporal stride during evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")

    parser.add_argument("--image_size", type=int, default=518, help="input spatial resolution")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="feature layers used")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="feature map indexes to fuse")

    parser.add_argument("--n_ctx", type=int, default=12, help="prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="prompt extra context")
    parser.add_argument("--depth", type=int, default=9, help="prompt depth")

    parser.add_argument("--use_temporal_adapter", action="store_true", help="enable temporal adapter during evaluation")
    parser.add_argument("--temporal_adapter_layers", type=int, default=1, help="temporal adapter layer count")
    parser.add_argument("--temporal_adapter_heads", type=int, default=8, help="temporal adapter attention heads")
    parser.add_argument("--temporal_adapter_mlp_ratio", type=float, default=2.0, help="temporal adapter MLP ratio")
    parser.add_argument("--temporal_adapter_dropout", type=float, default=0.0, help="temporal adapter dropout")

    parser.add_argument("--seed", type=int, default=111, help="random seed")

    args = parser.parse_args()
    evaluate(args)
