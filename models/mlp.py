
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.dropout = nn.Dropout(p=0.3)
        self.num_classes = num_classes

        self.lin1 = nn.Linear(768, 128)
        self.hid1 = nn.Linear(128, 128)
        self.hid2 = nn.Linear(128, self.num_classes)

    def forward(self, data):
        x = self.lin1(data.x_text_feat)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.hid1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.hid2(x)

        return F.log_softmax(x, dim=1)


