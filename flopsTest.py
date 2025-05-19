from Models.Old_version_1.W2G_VAE import W2G_VAE as W2G_VAE_old
from Models.Old_version_1.T2W_DiT import T2W_DiT as T2W_DiT_old
from Models import W2G_VAE, T2W_DiT
from calflops import calculate_flops
import torch

def flops_W2G_VAE_old():
    # 32 GFLOPS
    model = W2G_VAE_old(
        N_walks=48,
        seg_per_walk=8,
        N_interp=8,
        threshold=0.5
    ).cuda()

    walks = torch.randn(1, 48, 8, 8, 2).cuda()

    calculate_flops(model, args=[walks])

def flops_T2W_DiT_old():
    # 32 GFLOPS
    model = T2W_DiT_old(
        D_in=32,
        N_walks=48,
        seg_per_walk=8,
        L_traj=16,
        d_context=2,
        n_layers=8,
        T=500
    ).cuda()

    noisy_walk_embeds = torch.randn(1, 48, 8, 32).cuda()
    trajs = torch.randn(1, 48, 16, 2).cuda()
    t = torch.randint(0, 500, (1,)).cuda()

    calculate_flops(model, args=[noisy_walk_embeds, trajs, t])


def flops_W2G_VAE():
    # 66 GFLOPS
    model = W2G_VAE(
        walks_shape=[48, 8, 8, 2],
        d_encode=32,
        threshold=0.5
    ).cuda()

    walks = torch.randn(1, 48, 8, 8, 2).cuda()

    calculate_flops(model, args=[walks])


def flops_T2W_DiT():
    # 33 GFLOPS
    model = T2W_DiT(
        vae=W2G_VAE(
            walks_shape=[48, 8, 8, 2],
            d_encode=32,
            threshold=0.5
        ),
        T=500
    ).cuda()

    noisy_walk_embeds = torch.randn(1, 48, 8, 32).cuda()
    trajs = torch.randn(1, 48, 16, 2).cuda()
    t = torch.randint(0, 500, (1,)).cuda()

    calculate_flops(model, args=[noisy_walk_embeds, trajs, t])


if __name__ == "__main__":
    # flops_W2G_VAE_old()
    # flops_T2W_DiT_old()
    # flops_W2G_VAE()
    flops_T2W_DiT()