from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int | tuple[int, int]) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state, cell_state = state
        gates = self.gates(torch.cat([inputs, hidden_state], dim=1))
        input_gate, forget_gate, output_gate, candidate = torch.chunk(gates, chunks=4, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)

        cell_state = forget_gate * cell_state + input_gate * candidate
        hidden_state = output_gate * torch.tanh(cell_state)
        return hidden_state, cell_state

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        shape = (batch_size, self.hidden_channels, height, width)
        hidden_state = torch.zeros(shape, device=device)
        cell_state = torch.zeros(shape, device=device)
        return hidden_state, cell_state


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int | list[int] | tuple[int, ...],
        kernel_size: int | tuple[int, int] = 3,
    ) -> None:
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        channels = [input_channels] + list(hidden_channels)
        self.cells = nn.ModuleList(
            ConvLSTMCell(channels[index], channels[index + 1], kernel_size)
            for index in range(len(hidden_channels))
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if inputs.ndim != 5:
            raise ValueError(
                "ConvLSTM expects input shaped as [batch, steps, channels, height, width]."
            )

        current = inputs
        last_states: list[tuple[torch.Tensor, torch.Tensor]] = []

        for cell in self.cells:
            batch_size, steps, _, height, width = current.shape
            hidden_state, cell_state = cell.init_hidden(
                batch_size=batch_size,
                spatial_size=(height, width),
                device=current.device,
            )

            outputs = []
            for step_index in range(steps):
                hidden_state, cell_state = cell(current[:, step_index], (hidden_state, cell_state))
                outputs.append(hidden_state)

            current = torch.stack(outputs, dim=1)
            last_states.append((hidden_state, cell_state))

        return current, last_states
