import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
import math
from einops import rearrange, repeat


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TexTokTokenizer(nn.Module):
    """TexTok Tokenizer (Encoder) with text conditioning"""
    def __init__(self, 
                 img_size=256, 
                 patch_size=8, 
                 in_chans=3, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 num_tokens=128,
                 token_dim=8,
                 text_dim=768,
                 max_text_len=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Learnable image tokens
        self.image_tokens = nn.Parameter(torch.randn(1, num_tokens, embed_dim))
        
        # Text projection
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # Positional embeddings (not used for global tokens but kept for compatibility)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + num_tokens + max_text_len, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection for image tokens
        self.token_proj = nn.Linear(embed_dim, token_dim)
        
    def forward(self, x, text_embeds):
        B = x.shape[0]
        
        # Patch embedding
        patch_tokens = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Expand learnable image tokens
        image_tokens = self.image_tokens.expand(B, -1, -1)  # (B, num_tokens, embed_dim)
        
        # Project text embeddings
        text_tokens = self.text_proj(text_embeds)  # (B, text_len, embed_dim)
        
        # Concatenate all tokens: [patch_tokens, image_tokens, text_tokens]
        tokens = torch.cat([patch_tokens, image_tokens, text_tokens], dim=1)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        
        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        tokens = self.norm(tokens)
        
        # Extract only the image tokens
        start_idx = patch_tokens.shape[1]
        end_idx = start_idx + self.num_tokens

        image_tokens_out = tokens[:, start_idx:end_idx]
        
        # Project to final token dimension
        image_tokens_out = self.token_proj(image_tokens_out)
        
        return image_tokens_out


class TexTokDetokenizer(nn.Module):
    """TexTok Detokenizer (Decoder) with text conditioning"""
    def __init__(self, 
                 img_size=256, 
                 patch_size=8, 
                 out_chans=3, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 num_tokens=128,
                 token_dim=8,
                 text_dim=768,
                 max_text_len=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # Learnable patch tokens
        self.patch_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Input projections
        self.token_proj = nn.Linear(token_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + num_tokens + max_text_len, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection and unpatchify
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size * out_chans)
        
    def unpatchify(self, x):
        """Convert patch tokens back to image"""
        p = self.patch_size
        h = w = self.grid_size
    
        imgs = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=h, w=w, p1=p, p2=p, c=3)
        
        return imgs
        
    def forward(self, image_tokens, text_embeds):
        B = image_tokens.shape[0]
        
        # Expand learnable patch tokens
        patch_tokens = self.patch_tokens.expand(B, -1, -1)  # (B, num_patches, embed_dim)
        
        # Project input tokens
        image_tokens_proj = self.token_proj(image_tokens)  # (B, num_tokens, embed_dim)
        text_tokens = self.text_proj(text_embeds)  # (B, text_len, embed_dim)

        # Concatenate all tokens: [patch_tokens, image_tokens, text_tokens]
        tokens = torch.cat([patch_tokens, image_tokens_proj, text_tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        
        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        tokens = self.norm(tokens)

        # Extract only the patch tokens
        patch_tokens_out = tokens[:, :self.num_patches, :]

        # Project to pixel space
        patch_tokens_out = self.output_proj(patch_tokens_out)
        
        # Unpatchify to get final image
        images = self.unpatchify(patch_tokens_out)

        images = torch.tanh(images)
        
        return images


class TexTokVAE(nn.Module):
    """Complete TexTok AutoEncoder"""
    def __init__(self, 
                 img_size=256,
                 patch_size=8,
                 num_tokens=128,
                 token_dim=8,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 text_encoder_name="t5-base",
                 max_text_len=128):
        super().__init__()
        
        # Text encoder (frozen)
        self.text_tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        text_dim = self.text_encoder.config.d_model
        
        # Tokenizer and Detokenizer
        self.tokenizer = TexTokTokenizer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_tokens=num_tokens,
            token_dim=token_dim,
            text_dim=text_dim,
            max_text_len=max_text_len
        )
        
        self.detokenizer = TexTokDetokenizer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_tokens=num_tokens,
            token_dim=token_dim,
            text_dim=text_dim,
            max_text_len=max_text_len
        )
        
    def encode_text(self, captions):
        """Encode text captions using T5"""
        if isinstance(captions[0], str):
            # Tokenize text
            inputs = self.text_tokenizer(
                captions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            )
            input_ids = inputs.input_ids.to(next(self.parameters()).device)
            attention_mask = inputs.attention_mask.to(next(self.parameters()).device)
        else:
            input_ids = captions
            attention_mask = None
            
        # # Get text embeddings
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = outputs.last_hidden_state
            
        return text_embeds
        
    def encode(self, images, captions):
        """Encode images to tokens"""
        text_embeds = self.encode_text(captions)
        tokens = self.tokenizer(images, text_embeds)
        return tokens
        
    def decode(self, tokens, captions):
        """Decode tokens to images"""
        text_embeds = self.encode_text(captions)
        images = self.detokenizer(tokens, text_embeds)
        return images
        
    def forward(self, images, captions):
        """Full forward pass"""
        tokens = self.encode(images, captions)
        reconstructed = self.decode(tokens, captions)
        return reconstructed, tokens


# Discriminator for GAN training (StyleGAN-style)
class StyleGANDiscriminator(nn.Module):
    """StyleGAN-style discriminator for TexTok training"""
    def __init__(self, img_size=256, img_channels=3, base_channels=128, channel_multipliers=[1, 2, 4, 4, 4, 4]):
        super().__init__()
        
        self.img_size = img_size
        layers = []
        in_channels = img_channels
        
        # Initial layer
        layers.append(nn.Conv2d(in_channels, base_channels, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2))
        
        current_size = img_size
        in_channels = base_channels
        
        # Downsampling layers
        for multiplier in channel_multipliers:
            out_channels = base_channels * multiplier
            
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 2, 1))
            layers.append(nn.LeakyReLU(0.2))
            
            in_channels = out_channels
            current_size //= 2
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Final layers
        self.final_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.final_linear = nn.Linear(in_channels, 1)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        return x


# Example usage and training utilities
def create_textok_model(config):
    """Create TexTok model with given configuration"""
    model = TexTokVAE(
        img_size=config.get('img_size', 256),
        patch_size=config.get('patch_size', 8),
        num_tokens=config.get('num_tokens', 128),
        token_dim=config.get('token_dim', 8),
        embed_dim=config.get('embed_dim', 768),
        depth=config.get('depth', 12),
        num_heads=config.get('num_heads', 12),
        text_encoder_name=config.get('text_encoder_name', 't5-base'),
        max_text_len=config.get('max_text_len', 128)
    )
    return model


# Loss functions
class TexTokLoss(nn.Module):
    """Combined loss for TexTok training"""
    def __init__(self, 
                 recon_weight=1.0,
                 perceptual_weight=0.1,
                 gan_weight=0.1,
                 lecam_weight=0.0001):
        super().__init__()
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        self.lecam_weight = lecam_weight
        
        # Perceptual loss (VGG features)
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        self.perceptual_net = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.perceptual_net.parameters():
            param.requires_grad = False
            
    def perceptual_loss(self, input, target):
        input_features = self.perceptual_net(input)
        target_features = self.perceptual_net(target)
        return F.mse_loss(input_features, target_features)
        
    def forward(self, input_images, reconstructed, discriminator_real, discriminator_fake):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, input_images)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(reconstructed, input_images)
        
        # GAN loss (generator)
        gan_loss = F.softplus(-discriminator_fake).mean()
        
        # Total loss
        total_loss = (self.recon_weight * recon_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.gan_weight * gan_loss)
        
        return total_loss, {
            'recon_loss': recon_loss,
            'perceptual_loss': perceptual_loss,
            'gan_loss': gan_loss
        }


if __name__ == "__main__":
    # Example configuration
    config = {
        'img_size': 256,
        'patch_size': 8,
        'num_tokens': 128,
        'token_dim': 8,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'text_encoder_name': 't5-large',
        'max_text_len': 128
    }
    
    # Create model
    model = create_textok_model(config)
    discriminator = StyleGANDiscriminator(img_size=256)
    
    # Example forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)
    captions = [
        "A red car driving on a highway",
        "A cat sitting on a table", 
        "A mountain landscape with snow",
        "A person walking in a park"
    ]
    
    # Forward pass
    reconstructed, tokens = model(images, captions)
    print(f"Input shape: {images.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Discriminator outputs
    disc_real = discriminator(images)
    disc_fake = discriminator(reconstructed.detach())
    print(f"Discriminator real: {disc_real.shape}")
    print(f"Discriminator fake: {disc_fake.shape}")