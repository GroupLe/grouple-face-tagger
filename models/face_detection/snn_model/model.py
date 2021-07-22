import torch.nn as nn
import torch


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.conv_block = nn.Sequential(self.make_conv_block(3, 32),
                                        self.make_conv_block(32, 64),
                                        self.make_conv_block(64, 128, 5),
                                        self.make_conv_block(128, 256),
                                        self.make_conv_block(256, 256))

        self.fc = nn.Sequential(nn.Linear(256, 128),
                                nn.Tanh(),
                                nn.Linear(128, 64)
                                )

    @staticmethod
    def make_conv_block(in_channels, out_channels, kernel_size=3) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size),
                             nn.BatchNorm2d(out_channels),
                             nn.Tanh(),
                             nn.MaxPool2d(2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_block(x)

        output = output.squeeze()
        output = self.fc(output)
        return output

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_net(x)
