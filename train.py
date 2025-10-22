import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset, VideoAnomalyDataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    if args.use_temporal_adapter:
        AnomalyCLIP_parameters["temporal_adapter_cfg"] = {
            "enabled": True,
            "layers": args.temporal_adapter_layers,
            "num_heads": args.temporal_adapter_heads,
            "mlp_ratio": args.temporal_adapter_mlp_ratio,
            "dropout": args.temporal_adapter_dropout,
        }
    else:
        AnomalyCLIP_parameters["temporal_adapter_cfg"] = {"enabled": False}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    if args.input_type != "video":
        raise ValueError("当前训练脚本仅支持视频输入，请设置 --input_type video。")

    train_data = VideoAnomalyDataset(
        dataset_name=args.dataset,
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        clip_length=args.clip_length,
        stride=args.clip_stride_train,
        image_size=args.image_size,
        mode="train",
        preload_masks=args.video_preload_masks,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if args.debug_video_shapes:
        print("[Video Debug] fetching first batch...")
        try:
            first_batch = next(iter(train_dataloader))
        except Exception as exc:
            print(f"[Video Debug] failed to fetch batch: {exc.__class__.__name__}: {exc}")
            raise
        else:
            clip = first_batch["clip"]
            mask = first_batch["mask"]
            print(f"[Video Debug] clip shape: {tuple(clip.shape)}")
            print(f"[Video Debug] mask shape: {tuple(mask.shape)}")
            print(f"[Video Debug] video ids sample: {first_batch['video_id'][:min(len(first_batch['video_id']), 4)]}")
            if args.use_temporal_adapter:
                clip_device = clip.to(device)
                b, t = clip_device.shape[:2]
                flattened = clip_device.view(b * t, clip_device.size(2), clip_device.size(3), clip_device.size(4))
                temporal_meta = {"batch_size": b, "num_frames": t}
                with torch.no_grad():
                    model.encode_image(flattened, args.features_list, DPAM_layer=20, temporal_adapter=temporal_meta)
                print("[Video Debug] temporal adapter forward success")
        return

    # freeze backbone; only train prompt learner + optional temporal adapter
    for param in model.parameters():
        param.requires_grad = False

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    trainable_params = list(prompt_learner.parameters())
    temporal_adapter_trains = False
    if args.use_temporal_adapter and getattr(model.visual, "temporal_adapter_enabled", False) and getattr(model.visual, "temporal_adapter", None) is not None:
        temporal_adapter_trains = True
        model.visual.temporal_adapter.train()
        for param in model.visual.temporal_adapter.parameters():
            param.requires_grad = True
        trainable_params += [param for param in model.visual.temporal_adapter.parameters() if param.requires_grad]

    if len(trainable_params) == 0:
        raise RuntimeError("未找到可训练参数，请检查 prompt learner 与 temporal adapter 配置。")

    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    lam = 4
    
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        if temporal_adapter_trains:
            model.visual.temporal_adapter.train()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            clips = items['clip'].to(device)  # (B, T, 3, H, W)
            masks = items['mask'].to(device)  # (B, T, 1, H, W)
            labels = items['anomaly'].to(device)

            B, T, C, H, W = clips.shape
            frames = clips.view(B * T, C, H, W)
            temporal_meta = {"batch_size": B, "num_frames": T} if args.use_temporal_adapter else None

            if args.use_temporal_adapter:
                image_features, patch_features = model.encode_image(frames, args.features_list, DPAM_layer=20, temporal_adapter=temporal_meta)
            else:
                image_features, patch_features = model.encode_image(frames, args.features_list, DPAM_layer=20)

            image_features = image_features.view(B, T, -1)
            clip_features = image_features.mean(dim=1)
            clip_features = clip_features / (clip_features.norm(dim=-1, keepdim=True) + 1e-7)

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)

            text_logits = clip_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_logits = text_logits[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_logits, labels.long())
            image_loss_list.append(image_loss.item())

            similarity_map_list = []
            gt = masks.view(B * T, 1, H, W)
            gt = (gt > 0.5).float()
            gt_flat = gt.squeeze(1)

            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ (patch_feature.norm(dim = -1, keepdim = True) + 1e-7)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            local_loss = 0.0
            for similarity_map in similarity_map_list:
                local_loss += loss_focal(similarity_map, gt)
                local_loss += loss_dice(similarity_map[:, 1, :, :], gt_flat)
                local_loss += loss_dice(similarity_map[:, 0, :, :], 1 - gt_flat)

            local_loss = lam * local_loss
            total_loss = local_loss + image_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], total_loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            state = {"prompt_learner": prompt_learner.state_dict()}
            if temporal_adapter_trains:
                state["temporal_adapter"] = model.visual.temporal_adapter.state_dict()
            torch.save(state, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--input_type", type=str, default="video", choices=["image", "video"], help="select input pipeline")
    parser.add_argument("--clip_length", type=int, default=16, help="number of frames per video clip")
    parser.add_argument("--clip_stride_train", type=int, default=8, help="temporal stride for training clips")
    parser.add_argument("--clip_stride_test", type=int, default=4, help="temporal stride for validation/test clips")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--debug_video_shapes", action="store_true", help="print video batch shapes and exit")
    parser.add_argument("--video_preload_masks", action="store_true", help="preload mask volumes into memory")
    parser.add_argument("--use_temporal_adapter", action="store_true", help="enable temporal adapter for video inputs")
    parser.add_argument("--temporal_adapter_layers", type=int, default=1, help="number of temporal adapter layers")
    parser.add_argument("--temporal_adapter_heads", type=int, default=8, help="number of attention heads in temporal adapter")
    parser.add_argument("--temporal_adapter_mlp_ratio", type=float, default=2.0, help="MLP ratio inside temporal adapter")
    parser.add_argument("--temporal_adapter_dropout", type=float, default=0.0, help="dropout used in temporal adapter")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
