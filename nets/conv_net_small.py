import torch.nn as nn


class ConvNetSmall(nn.Module):
    def __init__(self, num_classes=2):
        """
        Smaller Convolutional Neural Network, given the nature of this project - studying loss of plasticity -
        the size and consequently peak performance of a model is not of the largest concern, this change was
        made so that this experiment could be run without prohibitively expensive equipment.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride = 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 32 * self.last_filter_output
        self.fc1 = nn.Linear(self.num_conv_outputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.pool = nn.MaxPool2d(2, 2)

        # architecture
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.append(nn.ReLU())
        self.layers.append(self.conv2)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc1)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc2)

        self.act_type = 'relu'

    def predict(self, x):
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x2 = x2.view(-1, self.num_conv_outputs)
        x3 = self.layers[5](self.layers[4](x2))
        x4 = self.layers[6](x3)
        return x4, [x1, x2, x3]
