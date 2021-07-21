import torch.nn as nn
import torch


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3),
                                         nn.ReLU(),

                                         nn.Conv2d(in_channels=9, out_channels=15, kernel_size=3),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2))

        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels=15, out_channels=18, kernel_size=5),
                                         nn.ReLU(),

                                         nn.Conv2d(in_channels=18, out_channels=27, kernel_size=5),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2),

                                         nn.Conv2d(in_channels=27, out_channels=33, kernel_size=17),
                                         nn.ReLU(),
                                         nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(825, 768),
                                nn.ReLU(),
                                nn.Linear(768, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256)
                                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_block1(x)
        output = self.conv_block2(output)

        output = output.view(output.size()[0], -1)
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
