import torch
import torch.nn as nn
import torch.functional as f

class ConvNet(nn.Module):
    def __init__(self, target, input_shape, lr=0.2, batch_size=16, epochs=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.BatchNorm3d(input_shape),
                nn.Conv3d(16, 32, kernel_size=(input_shape[2],5,5), stride=(1,2,2)),
                nn.Relu(),
                nn.MaxPool3d(


