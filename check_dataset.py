from dataset import VideoAnomalyDataset
from types import SimpleNamespace

with open("check_log.txt", "w", encoding="utf-8") as f:
    try:
        args = SimpleNamespace(image_size=518)
        f.write("creating dataset...\n")
        ds = VideoAnomalyDataset(dataset_name="ucsd", root="data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2", clip_length=16, stride=8, image_size=args.image_size)
        f.write(f"len: {len(ds)}\n")
        sample = ds[0]
        f.write(f"clip shape: {sample['clip'].shape}\n")
        f.write(f"mask shape: {sample['mask'].shape}\n")
    except Exception as exc:
        import traceback
        traceback.print_exc(file=f)
