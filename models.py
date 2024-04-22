import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dim
        self.num_classes = output_dim
        self.fc = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.fc(x)
    
class linearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class CIFARCNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(SimpleNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.fc_1(x))
        return self.fc_2(out)


class NextCharacterRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.encoder.weight.requires_grad = False

        self.rnn =\
            nn.RNN(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output

