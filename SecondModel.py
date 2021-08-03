from torch import nn


class SecondModel(nn.Module):
    def __init__(self):
        super(SecondModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(16, 32, (3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Dropout2d(0.05),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, input_):
        output_ = self.layer_stack(input_)
        return output_
