"""
full_train_loader.py
전체 FF++ c40 데이터셋 fine-tuning용 DataLoader.
초기화 시 전체 프레임을 메모리에 캐싱하여 Drive I/O 병목 해소.
"""
import os, h5py, random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

FAKE_FILES = [
    'FF++_Deepfakes_c40.h5',
    'FF++_Face2Face_c40.h5',
    'FF++_FaceSwap_c40.h5',
    'FF++_NeuralTextures_c40.h5',
]
REAL_FILES = ['FF++_original_c40.h5']


class FullTrainDataset(Dataset):
    """
    초기화 시 전체 프레임을 numpy array로 메모리 캐싱.
    __getitem__은 메모리에서 직접 읽어 Drive I/O 없음.
    """
    def __init__(self, data_dir, frames_per_video=10, seed=42,
                 split='train', augment=False):
        self.augment = augment
        rng = np.random.default_rng(seed)

        self.frames = []   # numpy array (H,W,3) 캐시
        self.labels = []   # int

        all_files = [(f, 1) for f in FAKE_FILES] + [(f, 0) for f in REAL_FILES]

        for fname, label in all_files:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f'[경고] 파일 없음: {fpath}')
                continue

            print(f'로딩 중: {fname} ...', end=' ', flush=True)
            with h5py.File(fpath, 'r') as f:
                vid_ids = sorted(f.keys())
                n = len(vid_ids)
                cut = int(n * 0.8)
                vid_ids = vid_ids[:cut] if split == 'train' else vid_ids[cut:]

                for vid_id in vid_ids:
                    n_frames = f[vid_id].shape[0]
                    idxs = rng.choice(n_frames,
                                      size=min(frames_per_video, n_frames),
                                      replace=False)
                    for fi in sorted(idxs):   # sorted로 순차 읽기 (HDF5 최적화)
                        self.frames.append(f[vid_id][fi])
                        self.labels.append(label)
            print(f'완료 ({len(self.labels)}개 누적)')

        n_fake = sum(1 for l in self.labels if l == 1)
        n_real = sum(1 for l in self.labels if l == 0)
        print(f'FullTrainDataset [{split}]: {len(self.labels)} frames  fake={n_fake}  real={n_real}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]

        t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)

        if self.augment:
            if random.random() > 0.5:
                t = TF.hflip(t)
            t = TF.adjust_brightness(t, random.uniform(0.8, 1.2))
            t = TF.adjust_contrast(t, random.uniform(0.8, 1.2))
            t = TF.adjust_saturation(t, random.uniform(0.9, 1.1))

        return t, label


def get_full_train_loader(data_dir, batch_size=16, num_workers=4,
                          frames_per_video=10, seed=42, split='train',
                          augment=False, balanced=False):
    ds = FullTrainDataset(data_dir, frames_per_video=frames_per_video,
                          seed=seed, split=split, augment=augment)

    if balanced and split == 'train':
        weights = [1.0 / (ds.labels.count(l) or 1) for l in ds.labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)

    return DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                      num_workers=num_workers, pin_memory=True)
