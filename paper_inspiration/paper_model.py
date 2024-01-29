import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class DeepQLearningNetwork(nn.Module):
    def __init__(self, lr, action_space, momentum):
        super(DeepQLearningNetwork, self).__init__()

        # Assuming the state is a 4x4 grid, with a single 'channel'
        input_channels = 1

        # Layer 1: Two parallel convolutional layers
        self.conv1a = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(input_channels, 128, kernel_size=5, stride=1, padding=2)

        # Layer 2: Four parallel convolutional layers
        # Adjust kernel sizes and paddings to ensure the output dimensions are the same
        self.conv2a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv2c = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3)
        self.conv2d = nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4)
        self.conv2e = nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5)

        # Layer 3: Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Layer 4: Linear layer for output
        # Update the input features to the linear layer based on the output size
        self.fc = nn.Linear(128 * 5 * 4 * 4 *2, action_space)
        #self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.loss = nn.HuberLoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.float().unsqueeze(1)  # Add a channel dimension

        # Apply convolutional layers
        x1 = F.relu(self.conv1a(x))
        x2 = F.relu(self.conv1b(x))

        x1a = F.relu(self.conv2a(x1))
        x1b = F.relu(self.conv2b(x1))
        x1c = F.relu(self.conv2c(x1))
        x1d = F.relu(self.conv2d(x1))
        x1e = F.relu(self.conv2e(x1))

        x2a = F.relu(self.conv2a(x2))
        x2b = F.relu(self.conv2b(x2))
        x2c = F.relu(self.conv2c(x2))
        x2d = F.relu(self.conv2d(x2))
        x2e = F.relu(self.conv2e(x2))

        # Concatenate outputs
        # Make sure the dimensions match for concatenation

        x = T.cat((x1a, x1b, x1c, x1d, x1e, x2a, x2b, x2c, x2d, x2e), 1)

        # Apply dropout
        x = self.dropout(x)

        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)

        # Apply the linear layer
        x = self.fc(x)

        return x
