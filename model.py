import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
  def __init__(self, num_classes=2):
    super().__init__()
    self.Conv1 = nn.Conv2d(3, 6, 3, 1)
    self.Conv2 = nn.Conv2d(6, 12, 3, 1)
    self.Conv3 = nn.Conv2d(12, 24, 3, 1)
    self.gap = nn.AdaptiveAvgPool2d(1)   # shrinks 26x26x24 -> 1x1x24
    self.fc1 = nn.Linear(24, 64)
    self.fc2 = nn.Linear(64, num_classes)

  def forward(self, X):
    X = F.relu(self.Conv1(X))
    X = F.max_pool2d(X, 2, 2)
    X = F.relu(self.Conv2(X))
    X = F.max_pool2d(X, 2, 2)
    X = F.relu(self.Conv3(X))
    X = F.max_pool2d(X, 2, 2)

    X = self.gap(X)
    X = X.view(X.size(0), -1)   # flatten to (batch, 24)

    X = F.relu(self.fc1(X))
    X = self.fc2(X)
    return F.log_softmax(X, dim=1)


def load_model(checkpoint_path: str, device: torch.device):
   
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # this assumes you saved like:
    # {"model_state_dict": model.state_dict(), "class_names": train_data.classes}
    class_names = checkpoint["class_names"]

    model = ConvolutionalNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names
