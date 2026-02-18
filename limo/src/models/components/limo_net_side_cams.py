import torch
from torch import nn
from torchvision import transforms
from typing import Tuple


class LimoNet(nn.Module):
    def __init__(
        self,
        goal_dim: int = 3,
        path_length: int = 50,
        se2_dim: int = 3,
        backbone_name: str = "dinov2_vits14",
        pretrained: bool = True,
        image_size: Tuple[int, int] = (308, 476),
        patch_size: int = 14,
        attn_heads: int = 6,
        decoder_layers: int = 4,
        ff_dim_factor: int = 4,
    ):
        super().__init__()
        self.backbone = None
        self._initialized = False

        self.goal_dim = goal_dim
        self.path_length = path_length
        self.se2_dim = se2_dim
        self.decoder_layers = decoder_layers
        self.ff_dim_factor = ff_dim_factor

        self.backbone_name = backbone_name
        self.pretrained = pretrained

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_h = image_size[0] // patch_size
        self.grid_w = image_size[1] // patch_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

        self.attn_heads = attn_heads

        self.setup()

    def setup(self):
        if self._initialized:
            return

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=self.pretrained,
        )

        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

        self.embed_dim = self.backbone.embed_dim

        self.row_embed = nn.Embedding(self.grid_h, self.embed_dim)
        self.col_embed = nn.Embedding(self.grid_w, self.embed_dim)
        self.view_embed = nn.Embedding(3, self.embed_dim)

        self.goal_proj = nn.Linear(self.goal_dim, self.embed_dim)
        self.time_embed = nn.Embedding(self.path_length, self.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.attn_heads,
            dim_feedforward=self.embed_dim * self.ff_dim_factor,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.decoder_layers
        )

        self.out_proj = nn.Linear(self.embed_dim, self.se2_dim)

        self._initialized = True

    def _encode_single_view(
        self, image: torch.Tensor, pos: torch.Tensor, view_id: int
    ) -> torch.Tensor:
        tokens = self.backbone.forward_features(image)["x_norm_patchtokens"]
        B, N, D = tokens.shape
        tokens = tokens + pos

        vid = torch.full((B, 1), view_id, device=image.device, dtype=torch.long)
        vemb = self.view_embed(vid).expand(B, N, D)
        tokens = tokens + vemb
        return tokens

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.setup()

        image_front = batch["image_front"]
        image_left = batch["image_left"]
        image_right = batch["image_right"]
        goal = batch["goal"]

        B, device = image_front.size(0), image_front.device

        N = self.grid_h * self.grid_w
        row_ids = torch.arange(self.grid_h, device=device)
        col_ids = torch.arange(self.grid_w, device=device)
        pos = self.row_embed(row_ids).unsqueeze(1) + self.col_embed(col_ids).unsqueeze(
            0
        )
        pos = pos.view(1, N, self.embed_dim)

        tokens_front = self._encode_single_view(image_front, pos, view_id=0)
        tokens_left = self._encode_single_view(image_left, pos, view_id=1)
        tokens_right = self._encode_single_view(image_right, pos, view_id=2)

        patch_tokens = torch.cat([tokens_front, tokens_left, tokens_right], dim=1)

        goal_emb = self.goal_proj(goal)

        time_ids = torch.arange(self.path_length, device=device)
        t_emb = self.time_embed(time_ids)
        t_emb = t_emb.unsqueeze(0).expand(B, -1, -1)
        queries = t_emb + goal_emb.unsqueeze(1)

        decoder_out = self.decoder(tgt=queries, memory=patch_tokens)
        path = self.out_proj(decoder_out)
        return path


if __name__ == "__main__":
    model = LimoNet()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,}")

    batch = {
        "image_front": torch.randn(2, 3, 308, 476),
        "image_left": torch.randn(2, 3, 308, 476),
        "image_right": torch.randn(2, 3, 308, 476),
        "goal": torch.randn(2, 3),
    }
    out = model(batch)
    print("Output shape:", out.shape)
    print("Output shape:", out.shape)  # expected (2, 50, 3)
