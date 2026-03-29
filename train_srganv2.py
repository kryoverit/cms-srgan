import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pyarrow.parquet as pq
import numpy as np
import gc
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class JetDataset(Dataset):
    def __init__(self, lr_data, hr_data, augment=True):
        self.lr = lr_data.astype(np.float32)
        self.hr = hr_data.astype(np.float32)
        self.augment = augment
    
    def __len__(self):
        return len(self.lr)
    
    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx]).clone()
        hr = torch.from_numpy(self.hr[idx]).clone()
        
        if self.augment:
            if np.random.random() > 0.5:
                lr = torch.flip(lr, [-1])
                hr = torch.flip(hr, [-1])
            if np.random.random() > 0.5:
                lr = torch.flip(lr, [-2])
                hr = torch.flip(hr, [-2])
        
        return lr, hr


def load_data(parquet_files, max_samples=6000):
    lr_list, hr_list = [], []
    
    for pf_path in parquet_files:
        print(f"Loading {pf_path}...")
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=2000, columns=['X_jets_LR', 'X_jets']):
            df = batch.to_pandas()
            for idx in range(len(df)):
                row = df.iloc[idx]
                lr = np.stack([np.array(ch.tolist(), dtype=np.float32) for ch in row['X_jets_LR']])
                hr = np.stack([np.array(ch.tolist(), dtype=np.float32) for ch in row['X_jets']])
                
                lr = np.clip(lr, 0, None)
                hr = np.clip(hr, 0, None)
                
                lr_list.append(lr)
                hr_list.append(hr)
                
                if len(lr_list) >= max_samples:
                    break
            if len(lr_list) >= max_samples:
                break
        del df
        gc.collect()
    
    lr_all = np.array(lr_list[:max_samples])
    hr_all = np.array(hr_list[:max_samples])
    
    del lr_list, hr_list
    gc.collect()
    
    lr_all = np.log1p(lr_all)
    hr_all = np.log1p(hr_all)
    
    lr_max = np.percentile(lr_all.reshape(lr_all.shape[0], -1), 99.9, axis=1, keepdims=True).reshape(-1, 1, 1, 1) + 1e-8
    hr_max = np.percentile(hr_all.reshape(hr_all.shape[0], -1), 99.9, axis=1, keepdims=True).reshape(-1, 1, 1, 1) + 1e-8
    
    lr_all = (lr_all / lr_max).astype(np.float32)
    hr_all = (hr_all / hr_max).astype(np.float32)
    
    print(f"Loaded: {lr_all.shape}")
    return lr_all, hr_all


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
        
        # Upsample: 64 -> 125 with pixel shuffle
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
        
        # Upsample: 64 -> 256 -> resize to 125
        h = F.leaky_relu(F.pixel_shuffle(self.upconv(h), 2), 0.2)
        h = F.interpolate(h, size=(125, 125), mode='bilinear', align_corners=False)
        
        return self.tail(h)


class Discriminator(nn.Module):
    def __init__(self, nc=3, f=64):
        super().__init__()
        
        def blk(ic, oc, s):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ic, oc, 3, s, 1)),
                nn.LeakyReLU(0.2, True)
            )
        
        self.l1 = blk(nc, f, 1)
        self.l2 = blk(f, f, 2)
        self.l3 = blk(f, f*2, 1)
        self.l4 = blk(f*2, f*2, 2)
        self.l5 = blk(f*2, f*4, 1)
        self.l6 = blk(f*4, f*4, 2)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(f*4, 1))
        )
    
    def forward(self, x):
        feats = []
        x = self.l1(x); feats.append(x)
        x = self.l2(x); feats.append(x)
        x = self.l3(x); feats.append(x)
        x = self.l4(x); feats.append(x)
        x = self.l5(x); feats.append(x)
        x = self.l6(x); feats.append(x)
        return self.head(x), feats


class SparsityWeightedL1(nn.Module):
    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        w = 1.0 + self.alpha * (target > 0).float()
        return (w * (pred - target).abs()).mean()


def gradient_penalty(D, real, fake):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=real.device)
    x = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad = torch.autograd.grad(D(x)[0].sum(), x, create_graph=True)[0]
    return ((grad.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()


def train_epoch(gen, disc, loader, epoch, g_opt, d_opt, sparse_loss, device, use_gan=True, writer=None):
    gen.train()
    disc.train()
    
    g_losses = []
    d_losses = []
    
    for lr, hr in tqdm(loader, desc=f"Epoch {epoch}"):
        lr = lr.to(device)
        hr = hr.to(device)
        
        batch_size = lr.size(0)
        
        if use_gan:
            d_opt.zero_grad()
            
            real_out, real_feats = disc(hr)
            with torch.no_grad():
                sr = gen(lr)
            fake_out, _ = disc(sr)
            
            d_loss = F.relu(1. - real_out).mean() + F.relu(1. + fake_out).mean()
            d_loss = d_loss + 10.0 * gradient_penalty(disc, hr, sr)
            d_loss.backward()
            d_opt.step()
            d_losses.append(d_loss.item())
            
            g_opt.zero_grad()
            sr = gen(lr)
            fake_out, fake_feats = disc(sr)
            _, real_feats = disc(hr)
            
            pix_loss = sparse_loss(sr, hr)
            
            # Edge loss
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=sr.dtype, device=sr.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=sr.dtype, device=sr.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            sr_edges = torch.sqrt(F.conv2d(sr, sobel_x, padding=1, groups=3)**2 + F.conv2d(sr, sobel_y, padding=1, groups=3)**2 + 1e-8)
            hr_edges = torch.sqrt(F.conv2d(hr, sobel_x, padding=1, groups=3)**2 + F.conv2d(hr, sobel_y, padding=1, groups=3)**2 + 1e-8)
            edge_loss = F.l1_loss(sr_edges, hr_edges)
            
            feat_loss = sum(F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats)) / len(fake_feats)
            
            adv_loss = -fake_out.mean()
            
            g_loss = 1.0 * pix_loss + 0.3 * edge_loss + 0.01 * feat_loss + 0.005 * adv_loss
            
            g_loss.backward()
            g_opt.step()
            g_losses.append(g_loss.item())
        else:
            g_opt.zero_grad()
            sr = gen(lr)
            loss = sparse_loss(sr, hr)
            loss.backward()
            g_opt.step()
            g_losses.append(loss.item())
    
    avg_g = np.mean(g_losses)
    avg_d = np.mean(d_losses) if d_losses else 0
    print(f"Epoch {epoch}: G_loss={avg_g:.4f}, D_loss={avg_d:.4f}")

    if writer is not None:
        writer.add_scalar('Loss/G', avg_g, epoch)
        if use_gan:
            writer.add_scalar('Loss/D', avg_d, epoch)

    return avg_g, avg_d


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train SRGANv2 on jet data')
    parser.add_argument('--parquet', nargs='+', default=[
        'QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet',
        'QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_LR.parquet',
        'QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494_LR.parquet'
    ], help='Parquet input files')
    parser.add_argument('--max-samples', type=int, default=6000)
    args = parser.parse_args()

    parquet_files = args.parquet
    print(f"Loading data from: {parquet_files}")
    lr_all, hr_all = load_data(parquet_files, max_samples=args.max_samples)
    
    N = len(lr_all)
    idx = np.random.permutation(N)
    val_n = int(N * 0.05)
    
    train_ds = JetDataset(lr_all[idx[val_n:]], hr_all[idx[val_n:]], augment=True)
    val_ds = JetDataset(lr_all[idx[:val_n]], hr_all[idx[:val_n]], augment=False)
    
    del lr_all, hr_all
    gc.collect()
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Bigger SRGAN v2: 96 features, 12 blocks
    gen = Generator(nc=3, f=96, nb=12).to(device)
    disc = Discriminator(nc=3, f=64).to(device)
    
    sparse_loss = SparsityWeightedL1(alpha=10.0).to(device)
    
    g_opt = torch.optim.Adam(gen.parameters(), lr=5e-5, betas=(0.9, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr=5e-5, betas=(0.9, 0.999))
    
    print(f"Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")
    
    writer = SummaryWriter(log_dir='runs/srganv2_experiment')

    resume_from = None  # Change to resume
    
    if resume_from:
        print(f"Resuming from {resume_from}...")
        ckpt = torch.load(resume_from, map_location=device)
        gen.load_state_dict(ckpt['gen'])
        if 'disc' in ckpt:
            disc.load_state_dict(ckpt['disc'])
    
    print("\n=== Phase 1: Pretrain ===")
    for epoch in range(1, 11):
        train_epoch(gen, disc, train_loader, epoch, g_opt, d_opt, sparse_loss, device, use_gan=False, writer=writer)
        if epoch % 5 == 0:
            torch.save({'gen': gen.state_dict()}, f'srganv2_pretrain_ep{epoch}.pth')
    
    print("\n=== Phase 2: GAN ===")
    for epoch in range(1, 21):
        train_epoch(gen, disc, train_loader, epoch, g_opt, d_opt, sparse_loss, device, use_gan=True, writer=writer)
        if epoch % 5 == 0:
            torch.save({
                'gen': gen.state_dict(),
                'disc': disc.state_dict(),
                'epoch': epoch
            }, f'srganv2_ep{epoch}.pth')
    
    torch.save({
        'gen': gen.state_dict(),
        'disc': disc.state_dict()
    }, 'srganv2_final.pth')

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
