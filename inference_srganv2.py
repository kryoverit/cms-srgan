"""
Inference for SRGAN v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


class ResBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(f, f, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(f, affine=True),
            nn.PReLU(f),
            nn.Conv2d(f, f, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(f, affine=True)
        )
    
    def forward(self, x):
        return x + self.b(x)


class Generator(nn.Module):
    def __init__(self, nc=3, f=96, nb=12):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(nc, f, 9, 1, 4), nn.PReLU(f))
        self.res = nn.Sequential(*[ResBlock(f) for _ in range(nb)])
        self.post_res = nn.Sequential(
            nn.Conv2d(f, f, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(f, affine=True)
        )
        self.upconv = nn.Conv2d(f, f * 4, 3, padding=1)
        self.tail = nn.Sequential(
            nn.Conv2d(f, f//2, 3, 1, 1),
            nn.PReLU(f//2),
            nn.Conv2d(f//2, nc, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.head(x)
        h = h + self.post_res(self.res(h))
        h = F.leaky_relu(F.pixel_shuffle(self.upconv(h), 2), 0.2)
        h = F.interpolate(h, size=(125, 125), mode='bilinear', align_corners=False)
        return self.tail(h)


def load_model(path):
    gen = Generator()
    ckpt = torch.load(path, map_location='cpu')
    if 'gen' in ckpt:
        gen.load_state_dict(ckpt['gen'])
    else:
        gen.load_state_dict(ckpt)
    gen.eval()
    return gen


def preprocess(lr, hr):
    lr = np.clip(lr, 0, None)
    hr = np.clip(hr, 0, None)
    lr = np.log1p(lr)
    hr = np.log1p(hr)
    lr_max = np.percentile(lr.reshape(lr.shape[0], -1), 99.9) + 1e-8
    hr_max = np.percentile(hr.reshape(hr.shape[0], -1), 99.9) + 1e-8
    lr = (lr / lr_max).astype(np.float32)
    hr = (hr / hr_max).astype(np.float32)
    return lr, hr


def visualize(model_path, num_samples=10):
    print(f"Loading model from {model_path}...")
    gen = load_model(model_path)
    
    pf = pq.ParquetFile("QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet")
    
    psnr_vals = []
    
    print(f"Processing {num_samples} samples...")
    for i, batch in enumerate(pf.iter_batches(batch_size=1, columns=['X_jets_LR', 'X_jets'])):
        if i >= num_samples:
            break
        
        df = batch.to_pandas()
        row = df.iloc[0]
        
        lr = np.stack([np.array(ch.tolist(), dtype=np.float32) for ch in row['X_jets_LR']])
        hr = np.stack([np.array(ch.tolist(), dtype=np.float32) for ch in row['X_jets']])
        
        lr, hr = preprocess(lr, hr)
        
        with torch.no_grad():
            sr = gen(torch.from_numpy(lr).unsqueeze(0)).squeeze(0).numpy()
        
        if sr.shape[1] != 125:
            sr = zoom(sr, (1, 125/64, 125/64), order=3)
        
        mse = np.mean((hr - sr) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
        
        psnr_vals.append(psnr)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        lr_img = np.clip(np.transpose(lr, (1, 2, 0)), 0, 1)
        hr_img = np.clip(np.transpose(hr, (1, 2, 0)), 0, 1)
        sr_img = np.clip(np.transpose(sr, (1, 2, 0)), 0, 1)
        
        axes[0].imshow(lr_img)
        axes[0].set_title(f'Low Res (64x64)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(hr_img)
        axes[1].set_title(f'High Res - GT (125x125)', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(sr_img)
        axes[2].set_title(f'Super Resolved (125x125)\nPSNR: {psnr:.2f}dB', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'result_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sample {i+1}: PSNR={psnr:.2f}dB, SR_range=[{sr.min():.4f}, {sr.max():.4f}]")
    
    print(f"\n=== AVERAGE ===")
    print(f"PSNR: {np.mean(psnr_vals):.2f} dB")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'srganv2_final.pth'
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    visualize(model_path, num)
