import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.left_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.left_padding, 0),
        )

        return self.conv(x)


class WaveNetBlock(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()

        self.dilated_conv = CausalConv1d(
            input_channels=residual_channels,
            output_channels=2 * residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.projection = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=skip_channels + residual_channels,
            kernel_size=1,
        )

        self.skip_channels = skip_channels

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, 2 * residual_channels, T]
        dilated = self.dilated_conv(x)

        # Each:
        # [B, residual_channels, T]
        conv_filter, conv_gate = dilated.chunk(
            2,
            dim=1,
        )

        gated = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)

        # [B, skip_channels + residual_channels, T]
        projected = self.projection(gated)

        skip, residual = torch.split(
            projected,
            [
                self.skip_channels,
                projected.size(1) - self.skip_channels,
            ],
            dim=1,
        )

        x = x + residual

        return x, skip


class WaveNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        dilations: list[int],
        filter_widths: list[int],
        skip_channels: int,
        residual_channels: int,
    ) -> None:
        super().__init__()

        if len(dilations) != len(filter_widths):
            raise ValueError("dilations and filter_widths must have the same length")

        self.input_projection = nn.Linear(
            input_size,
            residual_channels,
        )

        self.blocks = nn.ModuleList(
            [
                WaveNetBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=filter_width,
                    dilation=dilation,
                )
                for dilation, filter_width in zip(dilations, filter_widths)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # [B, T, input_size]
        #       ↓
        # [B, T, residual_channels]
        x = torch.tanh(self.input_projection(x))

        # [B, T, channels] -> [B, channels, T]
        x = x.transpose(1, 2)

        skip_outputs = []

        for block in self.blocks:
            x, skip = block(x)

            skip_outputs.append(skip)

        # [B, skip_channels, T]
        #       ↓
        # [B, num_layers * skip_channels, T]
        x = torch.cat(
            skip_outputs,
            dim=1,
        )

        x = F.relu(x)

        # [B, T, num_layers * skip_channels]
        return x.transpose(1, 2)
