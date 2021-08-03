from torch import nn
from Inception import Inception


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            Inception(16, 20, 10, 20, 15, 30, 10),
            nn.Conv2d(80, 100, (3, 3), padding='same'),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(100, 130, (3, 3), padding='same'),
            nn.BatchNorm2d(130),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(130, 150, (3, 3), padding='same'),
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            Inception(150, 75, 50, 75, 60, 50, 50),
            nn.Conv2d(250, 300, (3, 3), padding='same'),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(300, 512),
            nn.Dropout2d(0.4),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, input_):
        output_ = self.layer_stack(input_)
        return output_
