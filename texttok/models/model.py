# TexTok: Text-Conditioned Image Tokenization (PyTorch)
# pip install torch torchvision transformers lpips  (lpips optional)
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5TokenizerFast

# ----------------------------
# Basic building blocks
# ----------------------------

class PatchEmbed(nn.Module):
    """Image -> patch tokens (B, hw, D)."""
    def __init__(self, in_ch=3, patch=8, dim=768):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: (B,3,H,W) -> (B,D,H/p,W/p) -> (B, hw, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim=768, heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        h = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + h
        return x

class ViT1D(nn.Module):
    """Transformer over a single token sequence."""
    def __init__(self, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# ----------------------------
# TexTok model (encoder + detokenizer)
# ----------------------------

class TexTok(nn.Module):
    """
    In-context text conditioning (concat text tokens into self-attn stream).
    - Encoder input: [image_patches, learnable_image_tokens, text_tokens]
    - Decoder input: [learnable_patch_tokens, image_tokens(projected), text_tokens]
    Produces N continuous tokens of size d.
    """
    def __init__(
        self,
        img_res: int = 256,
        patch: int = 8,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        num_image_tokens: int = 128,   # N
        out_token_dim: int = 8,        # d
        t5_model: str = "t5-3b",
        freeze_t5: bool = True,
        dropout: float = 0.0,
        max_txt_len: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.N = num_image_tokens
        self.d = out_token_dim
        self.patch = patch
        self.max_txt_len = max_txt_len

        # Text encoder (frozen)
        self.tok = T5TokenizerFast.from_pretrained(t5_model)
        self.t5 = T5EncoderModel.from_pretrained(t5_model)
        if freeze_t5:
            for p in self.t5.parameters():
                p.requires_grad = False
        t5_w = self.t5.config.d_model

        # Project text -> model dim (separate projections for enc/dec)
        self.txt_proj_enc = nn.Linear(t5_w, dim)
        self.txt_proj_dec = nn.Linear(t5_w, dim)

        # Patch embed / unpatch
        self.embed = PatchEmbed(3, patch, dim)
        hw = (img_res // patch) * (img_res // patch)
        self.hw = hw

        # Learnable tokens
        self.learnable_image_tokens = nn.Parameter(torch.randn(1, self.N, dim) * 0.02)
        self.learnable_patch_tokens = nn.Parameter(torch.randn(1, hw, dim) * 0.02)

        # Encoder/decoder cores
        self.encoder = ViT1D(dim, depth, heads, mlp_dim, dropout)
        self.decoder = ViT1D(dim, depth, heads, mlp_dim, dropout)

        # Projections to/from latent (N,d) <-> (N,dim)
        self.to_latent = nn.Linear(dim, self.d)
        self.from_latent = nn.Linear(self.d, dim)

        # Predict patches back to pixels
        self.patch_head = nn.Linear(dim, patch * patch * 3)

    # ---- text helpers ----
    def encode_text(self, captions: List[str], device) -> torch.Tensor:
        toks = self.tok(
            captions, padding=True, truncation=True,
            max_length=self.max_txt_len, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            txt = self.t5(**toks).last_hidden_state  # (B, Nt, t5_w)
        return txt

    # ---- forward paths ----
    @torch.no_grad()
    def encode_only(self, images: torch.Tensor, captions: List[str]) -> torch.Tensor:
        """Return latent tokens Z (B,N,d) without reconstructing."""
        return self.tokenize(images, captions)

    def tokenize(self, images: torch.Tensor, captions: List[str]) -> torch.Tensor:
        B, _, H, W = images.shape
        device = images.device

        P = self.embed(images)  # (B, hw, dim)
        L = self.learnable_image_tokens.expand(B, -1, -1)  # (B, N, dim)

        T = self.encode_text(captions, device)             # (B, Nt, t5_w)
        T = self.txt_proj_enc(T)                           # (B, Nt, dim)

        seq = torch.cat([P, L, T], dim=1)                  # (B, hw + N + Nt, dim)
        seq = self.encoder(seq)

        # Slice back the image-token positions and map to latent d
        img_slice = seq[:, P.size(1): P.size(1) + L.size(1), :]  # (B, N, dim)
        Z = self.to_latent(img_slice)                             # (B, N, d)
        return Z

    def detokenize(self, Z: torch.Tensor, captions: List[str], H: int, W: int) -> torch.Tensor:
        """Z (B,N,d) -> image (B,3,H,W)."""
        B = Z.size(0)
        device = Z.device
        h, w = H // self.patch, W // self.patch
        assert h * w == self.hw, "H,W must match init img_res."

        Zp = self.from_latent(Z)                                   # (B, N, dim)
        Pp = self.learnable_patch_tokens.expand(B, -1, -1)         # (B, hw, dim)

        T = self.encode_text(captions, device)                     # (B, Nt, t5_w)
        T = self.txt_proj_dec(T)                                   # (B, Nt, dim)

        seq = torch.cat([Pp, Zp, T], dim=1)                        # (B, hw + N + Nt, dim)
        seq = self.decoder(seq)

        # Take first hw tokens -> per-patch pixel prediction
        patch_tokens = seq[:, :self.hw, :]                         # (B, hw, dim)
        px = self.patch_head(patch_tokens)                         # (B, hw, p*p*3)
        px = px.view(B, h, w, self.patch, self.patch, 3).permute(0,5,1,3,2,4)
        img = px.reshape(B, 3, H, W).contiguous()
        # img = img.clamp(0, 1)
        img = img.clamp(-1, 1)

        return img

    def forward(self, images: torch.Tensor, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Z, recon)."""
        B, _, H, W = images.shape
        Z = self.tokenize(images, captions)                  # (B,N,d)
        recon = self.detokenize(Z, captions, H, W)           # (B,3,H,W)
        return recon

# ----------------------------
# Optional: a tiny PatchGAN-style discriminator (very light)
# ----------------------------

class TinyPatchGAN(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        # Downsample to a patch map of logits
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.net = nn.Sequential(
            block(in_ch, base),
            block(base, base*2),
            block(base*2, base*4),
            nn.Conv2d(base*4, 1, 3, 1, 1)
        )
    def forward(self, x):
        return self.net(x)  # (B,1,h',w')


# def training_step(
#     model: TexTok,
#     batch: Dict[str, torch.Tensor],
#     captions: List[str],
#     optimizer: torch.optim.Optimizer,
#     disc: Optional[nn.Module] = None,
#     disc_optim: Optional[torch.optim.Optimizer] = None,
#     loss_fn: Optional[TexTokLoss] = None,
# ) -> Dict[str, float]:
#     """
#     batch["images"]: (B,3,H,W) in [0,1]
#     captions: list[str] of length B
#     """
#     if loss_fn is None:
#         loss_fn = TexTokLoss()

#     images = batch["images"]
#     # --- G step ---
#     optimizer.zero_grad(set_to_none=True)
#     Z, recon = model(images, captions)
#     g_loss, g_logs = loss_fn(recon, images, disc=disc)
#     g_loss.backward()
#     optimizer.step()


#     print(g_loss)

#     logs = {"g_loss": g_loss.item(), **g_logs}

#     # --- D step (optional) ---
#     if disc is not None and disc_optim is not None:
#         disc_optim.zero_grad(set_to_none=True)
#         d_loss, r1 = loss_fn.disc_loss(disc, images, recon.detach())
#         d_loss.backward()
#         disc_optim.step()
#         logs.update({"d_loss": d_loss.item(), "r1": r1.item()})

#     return logs


# # ----------------------------
# # Quick smoke test
# # ----------------------------
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     B, H, W = 2, 256, 256

#     model = TexTok(
#         img_res=H, patch=8, dim=768, depth=12, heads=12, mlp_dim=3072,
#         num_image_tokens=128, out_token_dim=8, t5_model="t5-large", freeze_t5=True
#     ).to(device)

#     opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.0, 0.99))

#     # Optional discriminator
#     disc = TinyPatchGAN().to(device)
#     disc_opt = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.0, 0.99))

#     # Fake data
#     images = torch.rand(B, 3, H, W, device=device)
#     captions = [
#         "A playful Pembroke Welsh Corgi trots through a sunlit field.",
#         "A vibrant scarlet macaw perched on a branch amidst green foliage."
#     ]

#     logs = training_step(
#         model,
#         {"images": images},
#         captions,
#         optimizer=opt,
#         disc=disc,
#         disc_optim=disc_opt,
#         loss_fn=TexTokLoss(lambda_perc=0.1, lambda_adv=0.1, r1_gamma=10.0, use_lpips=False),
#     )
#     print({k: round(v, 4) for k, v in logs.items()})
