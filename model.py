from torch.nn.functional import pad
import torch.nn as nn
from torchaudio.transforms import Spectrogram
from PIL import Image

class OneDimConv(nn.Module):
    def __init__(self):
        super(OneDimConv, self).__init__()
        # self.padding_layer = nn.ZeroPad2d(23) # Change tensor to 256x256
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8)
            )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
            )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)
            )
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
            )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU()
        # )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        # out = self.padding_layer(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc3(out)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.padding_layer = nn.ZeroPad2d(23) # Change tensor to 256x256
        self.specgram = nn.Sequential(
            Spectrogram(win_length=128, hop_length=64, n_fft=128),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 10), stride=(2, 2), padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 10), stride=(2, 2), padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Sequential(
            nn.Linear(2304, 128),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        # out = self.padding_layer(x)
        out = self.specgram(x)
        # print(out[0].size())
        # for out_t in out:
        #     print(out_t.size())
        #     # print(out_t.numpy())
        #     im = Image.fromarray(out_t.numpy()[0] * 1000)
        #     im.show()
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc2(out)
        return out
