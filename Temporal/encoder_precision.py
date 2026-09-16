"""Encoder policy selected from the passing Mac [99,132) control.

Use module paths from the pinned graph, not positional ranges in future exports.
The original checkpoint and floating tensor interfaces remain unchanged.
"""

EXPECTED_ENCODER_CONVOLUTIONS = 132
ENCODER_FP32_CONV_SCOPES = frozenset((
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '12', 'token_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '12', 'se', 'fc1'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '12', 'se', 'fc2'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '12', 'channel_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '12', 'channel_mixer', 'conv2', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '13', 'token_mixer', 'conv', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '13', 'token_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '13', 'channel_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_2', 'blocks', '13', 'channel_mixer', 'conv2', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'pre_block', 'token_mixer', 'conv', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'pre_block', 'token_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'pre_block', 'channel_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'pre_block', 'channel_mixer', 'conv2', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'spatial_downsample', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'channel_downsample', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'ffn', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'downsample', 'ffn', 'conv2', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '0', 'token_mixer', 'conv', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '0', 'token_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '0', 'se', 'fc1'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '0', 'se', 'fc2'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '0', 'channel_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '0', 'channel_mixer', 'conv2', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '1', 'token_mixer', 'conv', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '1', 'token_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '1', 'channel_mixer', 'conv1', 'bn'),
    ('image_encoder', 'trunk', 'body', 'stages_3', 'blocks', '1', 'channel_mixer', 'conv2', 'bn'),
    ('image_encoder', 'neck', '0', 'conv'),
    ('image_encoder', 'neck', '1', 'conv'),
    ('image_encoder', 'neck', '2', 'conv'),
    ('image_encoder', 'neck', '3', 'conv'),
    ('conv_s0',),
    ('conv_s1',),
))
