import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from grouple.models.face_detection.snn_model.datasets import TripletPathDataset
from grouple.models.face_detection.snn_model.model import EmbeddingNet, TripletNet
from grouple.models.face_detection.snn_model.functions import accuracy
from grouple.models.face_detection.snn_model.transformations import EmptyTransformation


if __name__ == '__main__':
    print(os.getcwd())
    root = Path('../../data/face_detection/processed/')

    composed = torchvision.transforms.Compose([T.ToTensor(),
                                               T.Resize((128, 128)),
                                               T.RandomChoice((T.ColorJitter(0.1, 0.1, 0.1),
                                                               T.RandomRotation(degrees=(0, 60)),
                                                               EmptyTransformation()))
                                               ])

    siamse_dataset = TripletPathDataset(Path(root), transform=composed)

    siamse_dataset_train, siamse_dataset_test = torch.utils.data.random_split(siamse_dataset,
                                                                              (round(len(siamse_dataset) * 0.8),
                                                                               int(len(siamse_dataset) * 0.2)))

    batch_size = 100
    train_dl = DataLoader(siamse_dataset_train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(siamse_dataset_test, batch_size=batch_size, shuffle=True)

    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    criterion = nn.TripletMarginLoss(margin=2.0, p=2)
    optimizer = torch.optim.Adagrad(model.parameters())

    for epoch in range(10):
        print('Epoch', epoch)

        print('train')
        for i, (anchor, positive, negative) in (enumerate(train_dl)):
            optimizer.zero_grad()

            pred = model.forward(anchor, positive, negative)
            pred_anchor, pred_positive, pred_negative = pred
            loss = criterion(pred_anchor, pred_positive, pred_negative)

            loss.backward()  # count gradients
            optimizer.step()  # update weights

            cur_accuracy = accuracy(pred_anchor, pred_positive, pred_negative)
            print(f'  [{i}/{len(train_dl)}] acc {cur_accuracy} loss {loss}')

        print('test')
        with torch.no_grad():
            for i, (anchor, positive, negative) in enumerate(test_dl):
                pred = model.forward(anchor, positive, negative)
                pred_anchor, pred_positive, pred_negative = pred
                loss = criterion(pred_anchor, pred_positive, pred_negative)
                cur_accuracy = accuracy(pred_anchor, pred_positive, pred_negative)
                print(f'  [{i}/{len(train_dl)}] acc {cur_accuracy} loss {loss}')



