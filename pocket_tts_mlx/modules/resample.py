"""Streaming resampling layers using strided convolutions."""

import mlx.nn as nn

from pocket_tts_mlx.modules.conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(nn.Module):
    """Downsample with strided Conv1d using streaming padding."""
    def __init__(self, stride: int, dimension: int, out_dimension: int | None = None):
        super().__init__()
        if out_dimension is None:
            out_dimension = dimension
        self.conv = StreamingConv1d(
            dimension,
            out_dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=1,
            bias=False,
            pad_mode="replicate",
        )

    def __call__(self, x, model_state: dict | None):
        """Apply downsampling convolution."""
        return self.conv(x, model_state)


class ConvTrUpsample1d(nn.Module):
    """Upsample with grouped ConvTranspose1d in streaming mode."""
    def __init__(self, stride: int, dimension: int, in_dimension: int | None = None):
        super().__init__()
        if in_dimension is None:
            in_dimension = dimension
        self.convtr = StreamingConvTranspose1d(
            in_dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=dimension,
            bias=False,
        )

    def __call__(self, x, model_state: dict | None):
        """Apply upsampling transposed convolution."""
        return self.convtr(x, model_state)
