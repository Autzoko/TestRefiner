"""SAM Transformer Decoder — standalone PyTorch, no mmdet/mmcv/mmengine."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class SAMAttention(nn.Module):
    """Multi-head attention with optional downsampling of the internal dim."""

    def __init__(self, embed_dim: int, num_heads: int = 8, downsample_rate: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.internal_dim = embed_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, (
            f"internal_dim ({self.internal_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.head_dim = self.internal_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, self.internal_dim)
        self.k_proj = nn.Linear(embed_dim, self.internal_dim)
        self.v_proj = nn.Linear(embed_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embed_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Reshape (B, N, C) -> (B, num_heads, N, head_dim)."""
        B, N, C = x.shape
        x = x.reshape(B, N, num_heads, C // num_heads)
        return x.transpose(1, 2)  # (B, num_heads, N, head_dim)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, num_heads, N, head_dim) -> (B, N, C)."""
        B, n_heads, N, head_dim = x.shape
        x = x.transpose(1, 2)  # (B, N, num_heads, head_dim)
        return x.reshape(B, N, n_heads * head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Standard multi-head attention with NO residual connection.

        Args:
            query: (B, N_q, embed_dim)
            key:   (B, N_k, embed_dim) — defaults to query
            value: (B, N_k, embed_dim) — defaults to query
        Returns:
            (B, N_q, embed_dim)
        """
        if key is None:
            key = query
        if value is None:
            value = query

        q = self._separate_heads(self.q_proj(query), self.num_heads)
        k = self._separate_heads(self.k_proj(key), self.num_heads)
        v = self._separate_heads(self.v_proj(value), self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N_q, head_dim)
        out = self._recombine_heads(out)  # (B, N_q, internal_dim)
        return self.out_proj(out)


class SAMTransformerLayer(nn.Module):
    """One layer of the SAM two-way transformer."""

    def __init__(
        self,
        num_heads: int = 8,
        embedding_dim: int = 256,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        mlp_dim: int = 2048,
    ):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe

        self.self_attn = SAMAttention(embedding_dim, num_heads)
        self.cross_attn_token_to_image = SAMAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        )
        self.cross_attn_image_to_token = SAMAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embedding_dim),
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries:  (B, N_q, C) — token embeddings (sparse prompt tokens)
            keys:     (B, N_k, C) — image embeddings
            query_pe: (B, N_q, C) — positional encoding for queries
            key_pe:   (B, N_k, C) — positional encoding for keys
        Returns:
            (queries, keys) with same shapes
        """
        # Step 1: Self attention on queries
        if self.skip_first_layer_pe:
            attn_out = self.self_attn(query=queries, key=queries, value=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(query=q, key=q, value=queries)
        queries = self.norm1(queries + attn_out)

        # Step 2: Cross attention — token to image
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(query=q, key=k, value=keys)
        queries = self.norm2(queries + attn_out)

        # Step 3: MLP on queries
        mlp_out = self.mlp(queries)
        queries = self.norm3(queries + mlp_out)

        # Step 4: Cross attention — image to token
        q = keys + key_pe
        k = queries + query_pe
        attn_out = self.cross_attn_image_to_token(query=q, key=k, value=queries)
        keys = self.norm4(keys + attn_out)

        return queries, keys


class SAMTransformerDecoder(nn.Module):
    """SAM two-way transformer decoder.

    Takes image embeddings, image positional encodings, and query (prompt)
    positional encodings, and produces refined query and key embeddings.
    """

    def __init__(
        self,
        num_layers: int = 2,
        num_heads: int = 8,
        embedding_dim: int = 256,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SAMTransformerLayer(
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    mlp_dim=mlp_dim,
                )
            )

        self.final_attn_token_to_image = SAMAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        )
        self.post_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pos: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embedding: (B, C, H, W) — image features from the encoder
            image_pos:       (B, C, H, W) or (C, H, W) — positional encoding
            query_pos:       (B, N_q, C) — prompt query positional encodings
        Returns:
            queries: (B, N_q, C)
            keys:    (B, HW, C) — refined image embeddings
        """
        B, C, H, W = image_embedding.shape

        # Flatten spatial dims: (B, C, H, W) -> (B, HW, C)
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Handle image_pos that may be (C, H, W) — broadcast to batch
        if image_pos.dim() == 3:
            image_pos = image_pos.unsqueeze(0).expand(B, -1, -1, -1)
        image_pos = image_pos.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Initialize queries from the positional encoding
        queries = query_pos
        keys = image_embedding

        # Run through transformer layers
        for layer in self.layers:
            queries, keys = layer(queries, keys, query_pe=query_pos, key_pe=image_pos)

        # Final token-to-image attention
        q = queries + query_pos
        k = keys + image_pos
        attn_out = self.final_attn_token_to_image(query=q, key=k, value=keys)
        queries = queries + attn_out
        queries = self.post_norm(queries)

        return queries, keys
