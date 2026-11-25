import math
import torch
from torch import nn


class RNNCell(nn.Module):
    """A simple RNN cell written directly from its equation."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x_t, h_previous):
        return torch.tanh(x_t @ self.W.T + h_previous @ self.U.T + self.b)


class LSTMCell(nn.Module):
    """An LSTM cell with four manually calculated gates."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x_t, h_previous, c_previous):
        gates = x_t @ self.W.T + h_previous @ self.U.T + self.b
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)

        c_new = forget_gate * c_previous + input_gate * candidate
        h_new = output_gate * torch.tanh(c_new)
        return h_new, c_new


class LSTM(nn.Module):
    """Run the custom LSTM cell over every time step."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        # x has shape (batch, time, input_size)
        batch_size, time_steps, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)
        outputs = []
        for time_index in range(time_steps):
            h, c = self.cell(x[:, time_index, :], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class BiLSTM(nn.Module):
    """Two separate custom LSTMs: one forward and one backward."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm = LSTM(input_size, hidden_size)
        self.backward_lstm = LSTM(input_size, hidden_size)

    def forward(self, x):
        forward_output = self.forward_lstm(x)
        reversed_input = torch.flip(x, dims=[1])
        backward_reversed_output = self.backward_lstm(reversed_input)
        backward_output = torch.flip(backward_reversed_output, dims=[1])
        return torch.cat([forward_output, backward_output], dim=2)


class Attention(nn.Module):
    """Learn one attention score for each time step and combine the states."""
    def __init__(self, input_size, attention_size):
        super().__init__()
        self.Wa = nn.Parameter(torch.empty(attention_size, input_size))
        self.ba = nn.Parameter(torch.empty(attention_size))
        self.v = nn.Parameter(torch.empty(attention_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(64)
        nn.init.uniform_(self.Wa, -bound, bound)
        nn.init.uniform_(self.ba, -bound, bound)
        nn.init.uniform_(self.v, -bound, bound)

    def forward(self, sequence):
        # sequence: (batch, time, features)
        energy = torch.tanh(sequence @ self.Wa.T + self.ba)
        scores = energy @ self.v
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(sequence * alpha.unsqueeze(-1), dim=1)
        return context, alpha
