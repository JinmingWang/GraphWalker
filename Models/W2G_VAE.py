from .Basics import *

class Block(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_head: int,
                 d_v: int,
                 n_heads: int,
                 n_walks: int,
                 expand: int,
                 ):
        super().__init__()

        self.n_walks = n_walks

        self.attn = NESA(d_in, d_head, d_v, n_heads, 0)
        self.res = Res1D(d_in, 0, expand)

    def forward(self, x):
        # Attention among all edges
        x = self.attn(x)
        # conv res applied to each walk
        x = x.unflatten(1, (self.n_walks, -1)).flatten(0, 1).transpose(-1, -2)
        # x = rearrange(x, "B (N_walks E_per_W) D -> (B N_walks) D E_per_W", N_walks=self.n_walks)
        x = self.res(x)
        return x.transpose(-1, -2).unflatten(0, (-1, self.n_walks)).flatten(1, 2)
        # return rearrange(x, "(B N_walks) D E_per_W -> B (N_walks E_per_W) D", N_walks=self.n_walks)


class Deduplicator(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_head: int,
                 n_heads: int,
                 threshold: float):
        super().__init__()
        self.d_in = d_in
        self.n_heads = n_heads
        self.d_head = d_head
        self.threshold = threshold

        self.x_proj = nn.Sequential(
            Linear(d_in, d_head * n_heads),
            SiLU(inplace=True),
            Linear(d_head * n_heads, d_head * n_heads),
            Rearrange("B L (H D)", "(B H) L D", H=n_heads, D=d_head)
        )   # ((Batch, Heads), N_segs, D_seg)

        self.sim_mat_proj = nn.Sequential(
            # Multi-head squeeze
            Rearrange("(B H) R C", "B R C H", H=n_heads),
            Linear(n_heads, n_heads),
            SiLU(inplace=True),
            Linear(n_heads, 1),
            nn.Flatten(2),
            # nn.Sigmoid()
        )

    def forward(self, duplicate_edges: Float[Tensor, "B L D"], x):
        B, L, D = duplicate_edges.shape

        # STEP 1. get pair-wise similarity matrix
        # x: (B*H, N_segs, D_seg)
        x = self.x_proj(x)
        # sim_mat: -inf to inf
        sim_mat = self.sim_mat_proj(- torch.sum(x.unsqueeze(1) - x.unsqueeze(2), dim=-1))  # (B, L, L)

        # STEP 2. keep only the similarity scores of previous elements for each element
        mask = torch.tril(torch.ones(L, L, device=duplicate_edges.device), diagonal=-1).bool()
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        sim_mat = sim_mat.masked_fill(~mask, -torch.inf)

        # STEP 3. get the maximum previous similarity score
        # high score means exists previous similar element
        max_previous_sim, _ = sim_mat.max(dim=2)   # (B, L)

        # STEP 4. construct gating
        uniqueness_score = - max_previous_sim

        return sim_mat, uniqueness_score

    def getGraphs(self, duplicate_edges, uniqueness_score):
        # duplicate_edges: (B, N_edges=384, N_interp=8, 2)
        # uniqueness_score: (B, N_edges=384)
        uniqueness_mask = torch.sigmoid(uniqueness_score) > self.threshold
        batch_edges = torch.unbind(duplicate_edges, dim=0)
        graphs = [edges[uniqueness_mask[b]] for b, edges in enumerate(batch_edges)]
        return graphs



class W2G_VAE(nn.Module):
    def __init__(self, walks_shape: List[int], d_encode: int, threshold: float=0.5):
        super().__init__()
        self.N_walks, self.L_walk, self.N_interp, _ = walks_shape
        self.threshold = threshold
        self.d_enc = d_encode

        # Input (B, N_trajs, L_walk, N_interp, 2)
        self.encoder = nn.Sequential(
            Rearrange("B N_walks L_walk N_interp D", "(B N_walks) (N_interp D) L_walk"),
            nn.Conv1d(2 * self.N_interp, 256, 3, 1, 1),
            Rearrange("(B N_walks) D L_walk", "B (N_walks L_walk) D", N_walks=self.N_walks),
            *[Block(256, 16, 64, 16, self.N_walks, 2) for _ in range(6)],
            Rearrange("B (N_walks L_walk) D", "B N_walks L_walk D", N_walks=self.N_walks),
            SiLU(inplace=True), Linear(256, 256),
            SiLU(inplace=True), Linear(256, d_encode * 2),
        )

        self.decoder_shared = nn.Sequential(
            Rearrange("B N_walks L_walk D", "(B N_walks) D L_walk"),
            nn.Conv1d(d_encode, 512, 3, 1, 1),
            Rearrange("(B N_walks) D L_walk", "B (N_walks L_walk) D", N_walks=self.N_walks),
            *[Block(512, 16, 64, 16, self.N_walks, 2) for _ in range(10)]
        )

        self.segs_head = nn.Sequential(
            *[Block(512, 16, 64, 16, self.N_walks, 2) for _ in range(3)],
            nn.SiLU(inplace=True),
            nn.Linear(512, self.N_interp * 2)
        )

        self.mat_head = nn.Sequential(
            *[Block(512, 16, 64, 16, self.N_walks, 2) for _ in range(3)],
            nn.SiLU(inplace=True),
            nn.Linear(512, 256)
        )

        self.deduplicator = Deduplicator(256, 4, 32, threshold)


    def encode(self, walks):
        return torch.split(self.encoder(walks), self.d_enc, dim=-1)

    def decode(self, z):
        # z: (B, N_walks, L_walk, d_enc)
        x = self.decoder_shared(z)  # (B, N_edges, 384)
        duplicate_edges = self.segs_head(x)  # (B, N_edges, N_interp * 2)
        sim_mat, uniqueness_score = self.deduplicator(duplicate_edges.detach(), self.mat_head(x))
        return duplicate_edges.unflatten(-1, (-1, 2)), sim_mat, uniqueness_score

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        epsilon = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        return z

    def forward(self, walks):
        z_mean, z_logvar = self.encode(walks)
        z = self.reparameterize(z_mean, z_logvar)
        duplicate_edges, sim_mat, uniqueness_score = self.decode(z)
        return z_mean, z_logvar, duplicate_edges, sim_mat, uniqueness_score


    def getGraphs(self, duplicate_edges, uniqueness_score):
        return self.deduplicator.getGraphs(duplicate_edges, uniqueness_score)