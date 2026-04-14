from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.convolution_lstm import ConvLSTM


class TemporalAttention(nn.Module):
    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        if sequence_features.ndim != 5:
            raise ValueError(
                "Temporal attention expects [batch, steps, channels, height, width] tensors."
            )

        flattened = sequence_features.flatten(start_dim=2)
        reference = flattened[:, -1:].contiguous()
        scale = math.sqrt(float(flattened.shape[-1]))
        weights = torch.softmax((flattened * reference).sum(dim=-1) / scale, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return (sequence_features * weights).sum(dim=1)


class CnnEncoder(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1), nn.SELU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.SELU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.SELU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.SELU())

    @staticmethod
    def _reshape_to_sequence(feature_map: torch.Tensor, batch_size: int, steps: int) -> torch.Tensor:
        channels, height, width = feature_map.shape[1:]
        return feature_map.reshape(batch_size, steps, channels, height, width)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, steps, channels, height, width = inputs.shape
        flattened = inputs.reshape(batch_size * steps, channels, height, width)

        conv1_out = self.conv1(flattened)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        return (
            self._reshape_to_sequence(conv1_out, batch_size, steps),
            self._reshape_to_sequence(conv2_out, batch_size, steps),
            self._reshape_to_sequence(conv3_out, batch_size, steps),
            self._reshape_to_sequence(conv4_out, batch_size, steps),
        )


class ConvLSTMEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_lstm = ConvLSTM(input_channels=32, hidden_channels=32, kernel_size=3)
        self.conv2_lstm = ConvLSTM(input_channels=64, hidden_channels=64, kernel_size=3)
        self.conv3_lstm = ConvLSTM(input_channels=128, hidden_channels=128, kernel_size=3)
        self.conv4_lstm = ConvLSTM(input_channels=256, hidden_channels=256, kernel_size=3)
        self.attention = TemporalAttention()

    def forward(
        self,
        conv1_out: torch.Tensor,
        conv2_out: torch.Tensor,
        conv3_out: torch.Tensor,
        conv4_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conv1_lstm_out, _ = self.conv1_lstm(conv1_out)
        conv2_lstm_out, _ = self.conv2_lstm(conv2_out)
        conv3_lstm_out, _ = self.conv3_lstm(conv3_out)
        conv4_lstm_out, _ = self.conv4_lstm(conv4_out)

        return (
            self.attention(conv1_lstm_out),
            self.attention(conv2_lstm_out),
            self.attention(conv3_lstm_out),
            self.attention(conv4_lstm_out),
        )


class CnnDecoder(nn.Module):
    def __init__(self, output_channels: int) -> None:
        super().__init__()
        self.up4 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.SELU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2), nn.SELU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2), nn.SELU())
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    @staticmethod
    def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        conv1_lstm_out: torch.Tensor,
        conv2_lstm_out: torch.Tensor,
        conv3_lstm_out: torch.Tensor,
        conv4_lstm_out: torch.Tensor,
    ) -> torch.Tensor:
        up4 = self._resize_like(self.up4(conv4_lstm_out), conv3_lstm_out)
        up4 = torch.cat((up4, conv3_lstm_out), dim=1)

        up3 = self._resize_like(self.up3(up4), conv2_lstm_out)
        up3 = torch.cat((up3, conv2_lstm_out), dim=1)

        up2 = self._resize_like(self.up2(up3), conv1_lstm_out)
        up2 = torch.cat((up2, conv1_lstm_out), dim=1)
        return self.output_layer(up2)


class MSCRED(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.encoder = CnnEncoder(input_channels=input_channels)
        self.temporal_encoder = ConvLSTMEncoder()
        self.decoder = CnnDecoder(output_channels=input_channels)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("MSCRED expects inputs shaped as [batch, steps, scales, height, width].")

        conv1_out, conv2_out, conv3_out, conv4_out = self.encoder(inputs)
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out = self.temporal_encoder(
            conv1_out,
            conv2_out,
            conv3_out,
            conv4_out,
        )
        return self.decoder(conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out)
