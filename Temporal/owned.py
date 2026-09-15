"""Owned EdgeTAM v1 graphs. See CONTRACT.md before changing tensor semantics.

Attention/rotary formulas and backbone construction are adapted from Meta's
Apache-2.0 EdgeTAM, pinned in UPSTREAM_REVISION. Attribution: Temporal/NOTICE.
The original model remains unmodified for independent reference comparisons.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_REVISION = "7711e012a30a2402c4eaab637bdb00a521302c91"
CHECKPOINT_SHA256 = "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"
CONTRACT = "bytewave.edgetam-temporal-owned.v2"
PREVIOUS_GRAPH_REVISION = "dense-initializer-points.v1"
GRAPH_REVISION = "dense-points-encoder-fanout.v1"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_OUTPUTS = ("raw_vision_features", "initial_vision_features", "high_res_feature_0", "high_res_feature_1")
MASK_OUTPUTS = ("low_res_mask", "high_res_mask", "best_iou", "object_pointer", "object_score")
MEMORY_OUTPUTS = ("memory_features", "memory_positions")
BANK_INPUTS = ("spatial_bank", "spatial_positions", "pointer_bank", "valid_slots")


class OfflineTimmBackbone(nn.Module):
    """Same module/state-dict layout; skip irrelevant ImageNet download.

    Strict full checkpoint loading below supplies every learned weight.
    """
    def __init__(self, name, features):
        super().__init__()
        from timm.models import create_model
        self.body = create_model(name, pretrained=False, in_chans=3,
                                 features_only=True,
                                 out_indices=tuple(int(f[len("layer"):]) for f in features))
        self.channel_list = self.body.feature_info.channels()[::-1]

    def forward(self, image):
        return list(self.body(image))


def load_reference(upstream: Path):
    upstream = upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, found {revision}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Upstream tracked files have local changes; use an unmodified checkout.")
    checkpoint = upstream / "checkpoints/edgetam.pt"
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Unexpected upstream checkpoint bytes (or an unexpanded LFS pointer).")
    sys.path.insert(0, str(upstream))
    import sam2
    if Path(sam2.__file__).resolve().parent != upstream / "sam2":
        raise ValueError("A different sam2 package is already imported.")
    from sam2.build_sam import build_sam2
    model = build_sam2("configs/edgetam.yaml", str(checkpoint), device="cpu",
                      apply_postprocessing=False, hydra_overrides_extra=[
                          "model.image_encoder.trunk._target_=owned.OfflineTimmBackbone",
                          "++model.binarize_mask_from_pts_for_mem_enc=true",
                      ])
    # Fail when assumptions change, rather than silently exporting another contract.
    expected = {"num_maskmem": 7, "image_size": 1024, "hidden_dim": 256,
                "mem_dim": 64, "max_obj_ptrs_in_encoder": 16,
                "memory_temporal_stride_for_eval": 1, "add_tpos_enc_to_obj_ptrs": False,
                "directly_add_no_mem_embed": True, "use_high_res_features_in_sam": True,
                "use_obj_ptrs_in_encoder": True, "multimask_output_for_tracking": True}
    for name, value in expected.items():
        if getattr(model, name) != value:
            raise ValueError(f"Unsupported model setting {name}")
    if model.spatial_perceiver.num_latents != 256 or model.spatial_perceiver.num_latents_2d != 256:
        raise ValueError("Expected 256 unrotated and 256 spatial memory tokens.")
    return model.eval()


def normalize(image):
    mean = image.new_tensor(MEAN).reshape(1, 3, 1, 1)
    std = image.new_tensor(STD).reshape(1, 3, 1, 1)
    return (image - mean) / std


def axial_tables(size, dimension, theta=10000.0):
    # Real equivalent of compute_axial_cis; evaluate once, outside tracing.
    indices = torch.arange(size * size, dtype=torch.float32)
    inverse = 1.0 / (theta ** (torch.arange(0, dimension, 4, dtype=torch.float32) / dimension))
    phase = torch.cat(((indices % size)[:, None] * inverse,
                       torch.floor(indices / size)[:, None] * inverse), dim=-1)
    return torch.cos(phase), torch.sin(phase)


def rotate(value, cosine, sine):
    # Interleaved real/imaginary pairs; no complex tensors in the exported graph.
    pairs = value.reshape(1, 1, cosine.shape[0], 128, 2)
    real, imaginary = pairs[..., 0], pairs[..., 1]
    return torch.stack((real * cosine - imaginary * sine,
                        real * sine + imaginary * cosine), dim=-1).reshape(1, 1, cosine.shape[0], 256)


class FixedAttention(nn.Module):
    def __init__(self, source, cross):
        super().__init__()
        if source.num_heads != 1 or source.internal_dim != 256:
            raise ValueError("Owned v1 attention requires one 256-channel head.")
        self.q_proj, self.k_proj = source.q_proj, source.k_proj
        self.v_proj, self.out_proj = source.v_proj, source.out_proj
        qcos, qsin = axial_tables(64, 256)
        if cross:
            spatial_cos, spatial_sin = axial_tables(16, 256)
            # Per memory: 256 learned 1D tokens bypass RoPE, then 256 2D tokens.
            kcos = torch.cat((torch.ones_like(spatial_cos), spatial_cos)).repeat(7, 1)
            ksin = torch.cat((torch.zeros_like(spatial_sin), spatial_sin)).repeat(7, 1)
            kcos = torch.cat((kcos, torch.ones(64, 128)))
            ksin = torch.cat((ksin, torch.zeros(64, 128)))
        else:
            # TorchScript requires separately registered buffers to be distinct
            # tensor objects, even when self-attention uses identical tables.
            kcos, ksin = qcos.clone(), qsin.clone()
        self.register_buffer("qcos", qcos)
        self.register_buffer("qsin", qsin)
        self.register_buffer("kcos", kcos)
        self.register_buffer("ksin", ksin)

    def forward(self, q, k, v, bias=None):
        q = rotate(self.q_proj(q).unsqueeze(1), self.qcos, self.qsin)
        k = rotate(self.k_proj(k).unsqueeze(1), self.kcos, self.ksin)
        v = self.v_proj(v).unsqueeze(1)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        return self.out_proj(output.squeeze(1))


class FixedMemoryLayer(nn.Module):
    def __init__(self, source):
        super().__init__()
        if source.pos_enc_at_attn or source.pos_enc_at_cross_attn_queries or not source.pos_enc_at_cross_attn_keys:
            raise ValueError("Unexpected positional-encoding policy.")
        if source.activation_str != "relu":
            raise ValueError("Unexpected memory feed-forward activation.")
        self.norm1, self.norm2, self.norm3 = source.norm1, source.norm2, source.norm3
        self.linear1, self.linear2 = source.linear1, source.linear2
        self.self_attn = FixedAttention(source.self_attn, cross=False)
        self.cross_attn = FixedAttention(source.cross_attn_image, cross=True)

    def forward(self, value, memory, positions, bias):
        normalized = self.norm1(value)
        value = value + self.self_attn(normalized, normalized, normalized)
        value = value + self.cross_attn(self.norm2(value), memory + positions, memory, bias)
        return value + self.linear2(F.relu(self.linear1(self.norm3(value))))


class FixedMemoryAttention(nn.Module):
    def __init__(self, model):
        super().__init__()
        if not model.memory_attention.pos_enc_at_input or len(model.memory_attention.layers) != 2:
            raise ValueError("Unexpected memory attention configuration.")
        self.layers = nn.ModuleList(FixedMemoryLayer(layer) for layer in model.memory_attention.layers)
        self.norm = model.memory_attention.norm
        current_pos = model.image_encoder.neck.position_encoding(torch.zeros(1, 256, 64, 64))
        self.register_buffer("current_pos", current_pos.flatten(2).transpose(1, 2))
        # slot 0 = condition (row 6), slots 1..6 = lag 6..1 (rows 5..0).
        self.register_buffer("temporal", model.maskmem_tpos_enc.detach().reshape(7, 64).flip(0).reshape(1, 7, 1, 64))

    def forward(self, raw, spatial, positions, pointers, valid):
        memory = torch.cat((spatial.reshape(1, 3584, 64), pointers.reshape(1, 64, 64)), dim=1)
        positions = torch.cat(((positions + self.temporal).reshape(1, 3584, 64),
                               torch.zeros_like(pointers.reshape(1, 64, 64))), dim=1)
        spatial_valid = valid[:, :7].unsqueeze(-1).expand(1, 7, 512).reshape(1, 3584)
        pointer_valid = valid[:, 7:].unsqueeze(-1).expand(1, 16, 4).reshape(1, 64)
        token_valid = torch.cat((spatial_valid, pointer_valid), dim=-1)
        bias = ((1.0 - token_valid) * -10000.0).reshape(1, 1, 1, 3648)
        value = raw.flatten(2).transpose(1, 2) + 0.1 * self.current_pos
        for layer in self.layers:
            value = layer(value, memory, positions, bias)
        return self.norm(value).transpose(1, 2).reshape(1, 256, 64, 64)


def selected_masks(model, raw, high0, high1, points=None):
    outputs = model._forward_sam_heads(raw, point_inputs=points,
                                       high_res_features=[high0, high1], multimask_output=True)
    return outputs[3], outputs[4], outputs[2].max(dim=-1).values, outputs[5], outputs[6]


class ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        features = self.model.forward_image(normalize(image))["backbone_fpn"]
        raw = features[2]
        initial = raw + self.model.no_mem_embed.reshape(1, 256, 1, 1)
        return raw, initial, features[0], features[1]


class DensePointPromptEncoder(nn.Module):
    """Same point arithmetic/weights as upstream, using broadcast selects.

    Supports the owned point-only/no-mask prompt contract.
    It also removes the upstream concatenation with a [B,0,C] empty tensor.
    """
    def __init__(self, source):
        super().__init__()
        self.source = source

    def get_dense_pe(self):
        return self.source.get_dense_pe()

    def forward(self, points, boxes=None, masks=None):
        if points is None or boxes is not None or masks is not None:
            raise ValueError('Owned prompt encoder supports points only')
        coords, labels = points
        shifted = coords + 0.5
        coords = torch.cat((shifted, torch.zeros_like(coords[:, :1, :])), dim=1)
        labels = torch.cat((labels, -torch.ones_like(labels[:, :1])), dim=1)
        value = self.source.pe_layer.forward_with_coords(coords, self.source.input_image_size)
        missing = (labels == -1).unsqueeze(-1)
        value = torch.where(missing, torch.zeros_like(value), value)
        value = torch.where(missing, value + self.source.not_a_point_embed.weight, value)
        for index in range(4):
            value = torch.where((labels == index).unsqueeze(-1),
                                value + self.source.point_embeddings[index].weight, value)
        dense = self.source.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            coords.shape[0], -1, *self.source.image_embedding_size)
        return value, dense


class Initializer(nn.Module):
    def __init__(self, model, *, dense_points=True):
        super().__init__()
        if dense_points:
            # Copy the module registry before replacing a child. Learned modules
            # are shared read-only; the independent upstream reference is untouched.
            self.model = copy.copy(model)
            self.model._modules = model._modules.copy()
            self.model.sam_prompt_encoder = DensePointPromptEncoder(model.sam_prompt_encoder).eval()
        else:
            # Retain the original graph for historical controlled diagnostics.
            self.model = model

    def forward(self, initial, high0, high1, coords, labels):
        return selected_masks(self.model, initial, high0, high1,
                              {"point_coords": coords, "point_labels": labels})


class InitialMemoryEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, raw, mask, score):
        features, positions = self.model._encode_new_memory(
            [raw.flatten(2).permute(2, 0, 1)], [(64, 64)], mask, score, is_mask_from_pts=True)
        return features, positions[0]


class Propagator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention = FixedMemoryAttention(model)

    def forward(self, raw, high0, high1, spatial, positions, pointers, valid):
        conditioned = self.attention(raw, spatial, positions, pointers, valid)
        masks = selected_masks(self.model, conditioned, high0, high1)
        features, positions_out = self.model._encode_new_memory(
            [raw.flatten(2).permute(2, 0, 1)], [(64, 64)], masks[1], masks[4], is_mask_from_pts=False)
        return (*masks, features, positions_out[0])


def components(model):
    return {"ImageEncoder": ImageEncoder(model).eval(), "Initializer": Initializer(model).eval(),
            "InitialMemoryEncoder": InitialMemoryEncoder(model).eval(), "Propagator": Propagator(model).eval()}
