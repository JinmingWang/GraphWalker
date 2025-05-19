from .Basics import *
from .W2G_VAE import W2G_VAE

class Block(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_head: int,
                 d_v: int,
                 n_heads: int,
                 dropout: float,
                 n_walks: int,
                 d_time: int,
                 expansion: int,
                 ):
        super().__init__()

        self.n_walks = n_walks

        # self.res = SERes1D(d_in * 2, d_in * 2, d_in * 2)
        # self.attn = AttentionBlockOld(d_in * 2, d_head, d_in * 2, d_in * 2, n_heads, dropout, "dist", d_time)

        self.res = Res1D(d_in * 2, d_time, expansion)
        self.attn = NESA(d_in * 2, d_head, d_v, n_heads, dropout, True)
        self.out_proj = nn.Sequential(SiLU(inplace=True), Linear(d_in*2, d_in))

    def forward(self, x, traj_features, t):
        identity = x
        x = torch.cat([x, traj_features], dim=-1)

        # Attention among all edges
        x = self.attn(x)

        # conv res applied to each walk
        # x = rearrange(x, "B (N_walks E_per_W) D -> (B N_walks) D E_per_W", N_walks=self.n_walks)
        x = x.unflatten(1, (self.n_walks, -1)).flatten(0, 1).transpose(-1, -2)
        x = self.res(x, t)
        x = x.transpose(-1, -2).unflatten(0, (-1, self.n_walks)).flatten(1, 2)
        # return rearrange(x, "(B N_walks) D E_per_W -> B (N_walks E_per_W) D", N_walks=self.n_walks)

        return self.out_proj(x) + identity

class TrajectoryEncoder(nn.Sequential):
    def __init__(self, N_walks):
        super().__init__(
            nn.Flatten(0, 1), Transpose(-1, -2),
            Conv1dNormAct(2, 128, 3, 1, 1),
            nn.Conv1d(128, 256, 3, 2, 1),
            Res1D(256, 0, 2),
            *self.__trajsToPieces(N_walks),
            NESA(256, 16, 64, 16, 0.0, True),
            *self.__piecesToTrajs(N_walks),
            Res1D(256, 0, 2),
            *self.__trajsToPieces(N_walks),
            NESA(256, 16, 64, 16, 0.0, True),
            *self.__piecesToTrajs(N_walks),
            Res1D(256, 0, 2),
            *self.__trajsToPieces(N_walks),
            NESA(256, 16, 64, 16, 0.0, True),

            SiLU(inplace=True),
            Linear(256, 256),
        )

    def __trajsToPieces(self, N_walks):
        # (B*N_trajs, D, L) -> (B, N_trajs*L, D)
        return [nn.Unflatten(0, (-1, N_walks)),
                Transpose(2, 3),
                nn.Flatten(1, 2)]

    def __piecesToTrajs(self, N_walks):
        # (B, N_trajs*L, D) -> (B*N_Trajs, D, L)
        return [nn.Unflatten(1, (N_walks, -1)),
                Transpose(2, 3),
                nn.Flatten(0, 1)]


class T2W_DiT(nn.Module):
    def __init__(self, vae: W2G_VAE, T: int):
        super().__init__()
        self.D_in = vae.d_enc
        self.N_walks = vae.N_walks
        self.edges_per_walk = vae.L_walk
        self.T = T

        self.time_embed = nn.Sequential(
            nn.Embedding(T, 256),
            nn.Linear(256, 512),
            SiLU(inplace=True),
            nn.Linear(512, 256),
            SiLU(inplace=True),
            nn.Unflatten(-1, (1, -1))
        )

        # Input: (B, N, L, 2)

        self.walks_proj = nn.Sequential(
            # Rearrange("B N_walks E_per_W D", "B (N_walks E_per_W) D"),
            nn.Flatten(1, 2),
            nn.Linear(self.D_in, 256), SiLU(inplace=True),
            nn.Linear(256, 256), SiLU(inplace=True),
            nn.Linear(256, 256)
        )

        self.trajs_proj = TrajectoryEncoder(self.N_walks)

        self.stages = SequentialWithAdditionalInputs(*[
            Block(256, 16, 64, 16, 0.15, self.N_walks, 256, 2)
            for _ in range(8)
        ])

        # (B, N, L, 128)
        self.head = nn.Sequential(
            # Rearrange("B (N_walks E_per_W) D", "B N_walks E_per_W D", N_walks=N_walks),
            nn.Linear(256, 64), SiLU(inplace=True),
            nn.Linear(64, self.D_in),
            nn.Unflatten(1, (self.N_walks, -1)),
        )

        # Do not freaking add nn.init.zero_, this will mess up the whole training

    def forward(self, noisy_encs, trajs, t):
        t = self.time_embed(t)

        traj_features = self.trajs_proj(trajs)
        x = self.walks_proj(noisy_encs)

        x = self.stages(x, traj_features, t)

        pred_noise = self.head(x)

        return pred_noise

