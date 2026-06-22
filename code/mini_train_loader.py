"""
mini_train_loader.py
baseline / improved 모델 fine-tuning용 mini train DataLoader.
HDF5 구조: {category}/{video_id}: (100, 224, 224, 3)

초기화 시 전체 프레임을 메모리에 캐싱하여 Drive I/O 병목 해소.
split='train'/'val': 비디오 단위 80/20 분리 (진짜 독립적인 val 보장)
라벨: original=0(real), 나머지=1(fake)
"""
import os, h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

FAKE_KEYWORDS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


class MiniTrainDataset(Dataset):
    """
    split='train': 비디오의 앞 80% 사용
    split='val'  : 비디오의 뒤 20% 사용 (train과 완전히 다른 비디오)
    """
    def __init__(self, mini_h5_path, frames_per_video=10, seed=42,
                 split='train'):
        rng = np.random.default_rng(seed)
        self.frames = []
        self.labels = []

        print(f'로딩 중 [{split}]: {os.path.basename(mini_h5_path)} ...',
              end=' ', flush=True)
        with h5py.File(mini_h5_path, 'r') as f:
            for category in f.keys():
                label = 1 if any(k in category for k in FAKE_KEYWORDS) else 0
                vid_ids = sorted(f[category].keys())

                # 비디오 단위 분리
                cut = int(len(vid_ids) * 0.8)
                vid_ids = vid_ids[:cut] if split == 'train' else vid_ids[cut:]

                for vid_id in vid_ids:
                    n = f[category][vid_id].shape[0]
                    idxs = rng.choice(n, size=min(frames_per_video, n),
                                      replace=False)
                    for fi in sorted(idxs):
                        self.frames.append(f[category][vid_id][fi])
                        self.labels.append(label)
        print('완료')

        n_fake = sum(1 for l in self.labels if l == 1)
        n_real = sum(1 for l in self.labels if l == 0)
        print(f"MiniTrainDataset [{split}]: {len(self.labels)} frames  "
              f"fake={n_fake}  real={n_real}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = torch.from_numpy(
            self.frames[idx].astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        return t, self.labels[idx]


def get_mini_train_loader(mini_h5_path, batch_size=32, num_workers=2,
                          frames_per_video=10, seed=42, split='train'):
    ds = MiniTrainDataset(mini_h5_path, frames_per_video=frames_per_video,
                          seed=seed, split=split)
    shuffle = (split == 'train')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
