
import math
import torch
from torch import nn

# Import layers from the same src package.
try:
    from .layers import BiLSTM, Attention
except ImportError:
    from layers import BiLSTM, Attention


class HARNet(nn.Module):
    """Bidirectional LSTM, temporal attention, and six-class output layer."""

    def __init__(self, input_size=9, hidden_size=64, num_classes=6):
        super().__init__()

        self.bilstm = BiLSTM(input_size, hidden_size)

        self.attention = Attention(
            input_size=2 * hidden_size,
            attention_size=hidden_size
        )

        self.classifier = nn.Linear(
            2 * hidden_size,
            num_classes
        )

        bound = 1.0 / math.sqrt(hidden_size)

        nn.init.uniform_(
            self.classifier.weight,
            -bound,
            bound
        )

        nn.init.uniform_(
            self.classifier.bias,
            -bound,
            bound
        )

    def forward(self, x):
        sequence = self.bilstm(x)

        context, alpha = self.attention(sequence)

        logits = self.classifier(context)

        return logits, alpha


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":

    torch.manual_seed(0)

    model = HARNet()

    sample = torch.zeros(4, 128, 9)

    logits, alpha = model(sample)

    print(
        "LSTM output shape:",
        model.bilstm.forward_lstm(sample).shape
    )

    print(
        "BiLSTM output shape:",
        model.bilstm(sample).shape
    )

    print("Logits shape:", logits.shape)

    print("Attention shape:", alpha.shape)

    print("Attention sums:", alpha.sum(dim=1))

    print(
        "Trainable parameters:",
        count_parameters(model)
    )
