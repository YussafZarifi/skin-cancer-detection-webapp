import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 6, 3, 1)
        self.Conv2 = nn.Conv2d(6, 12, 3, 1)
        self.Conv3 = nn.Conv2d(12, 24, 3, 1)

        # Fully Connected Layer
        self.fc1 = nn.Linear(26 * 26 * 24, 10000)
        self.fc2 = nn.Linear(10000, 7000)
        self.fc3 = nn.Linear(7000, 4000)
        self.fc4 = nn.Linear(4000, 1000)
        self.fc5 = nn.Linear(1000, 100)
        self.fc6 = nn.Linear(100, 10)

    def forward(self, X):
        # First Pass
        X = F.relu(self.Conv1(X))
        X = F.max_pool2d(X, 2, 2)

        # Second Pass
        X = F.relu(self.Conv2(X))
        X = F.max_pool2d(X, 2, 2)

        # Third Pass
        X = F.relu(self.Conv3(X))
        X = F.max_pool2d(X, 2, 2)

        # Flatten
        X = X.view(-1, 24 * 26 * 26)

        # Fully connected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        X = F.relu(self.fc5(X))
        X = self.fc6(X)

        # You trained with log_softmax at the end
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
