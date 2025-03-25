import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetLN(nn.Module):
    def __init__(self):
        super(ResNetLN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # Input: 3x32x32, Output: 6x28x28
        self.ln1 = nn.LayerNorm([6, 28, 28])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # Output: 16x10x10
        self.ln2 = nn.LayerNorm([16, 10, 10])
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.ln3 = nn.LayerNorm(120)
        self.fc2 = nn.Linear(120, 84)
        self.ln4 = nn.LayerNorm(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ln_data = []  # To store (input, normalized_output) for each LN layer

        # Conv1 + LN1
        x_conv1 = self.conv1(x)
        mean1 = x_conv1.mean(dim=[1, 2, 3], keepdim=True)
        var1 = x_conv1.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        normalized1 = (x_conv1 - mean1) / torch.sqrt(var1 + self.ln1.eps)
        ln_data.append((x_conv1, normalized1))
        x = self.ln1(x_conv1)
        x = self.pool(F.relu(x))

        # Conv2 + LN2
        x_conv2 = self.conv2(x)
        mean2 = x_conv2.mean(dim=[1, 2, 3], keepdim=True)
        var2 = x_conv2.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        normalized2 = (x_conv2 - mean2) / torch.sqrt(var2 + self.ln2.eps)
        ln_data.append((x_conv2, normalized2))
        x = self.ln2(x_conv2)
        x = self.pool(F.relu(x))

        # Flatten
        x = x.view(-1, 16 * 5 * 5)

        # FC1 + LN3
        x_fc1 = self.fc1(x)
        mean3 = x_fc1.mean(dim=-1, keepdim=True)
        var3 = x_fc1.var(dim=-1, keepdim=True, unbiased=False)
        normalized3 = (x_fc1 - mean3) / torch.sqrt(var3 + self.ln3.eps)
        ln_data.append((x_fc1, normalized3))
        x = self.ln3(x_fc1)
        x = F.relu(x)

        # FC2 + LN4
        x_fc2 = self.fc2(x)
        mean4 = x_fc2.mean(dim=-1, keepdim=True)
        var4 = x_fc2.var(dim=-1, keepdim=True, unbiased=False)
        normalized4 = (x_fc2 - mean4) / torch.sqrt(var4 + self.ln4.eps)
        ln_data.append((x_fc2, normalized4))
        x = self.ln4(x_fc2)
        x = F.relu(x)

        # Output layer
        x = self.fc3(x)
        return x, ln_data

class DyT(nn.Module):
    def __init__(self, C, init_α=0.5):
        super(DyT, self).__init__()
        self.α = nn.Parameter(torch.ones(1) * init_α)  # Scalar learnable parameter
        self.γ = nn.Parameter(torch.ones(C))  # Per-channel scaling
        self.β = nn.Parameter(torch.zeros(C))  # Per-channel shift

    def forward(self, x):
        # Apply tanh transformation
        tanh_output = torch.tanh(self.α * x)
        # Adjust γ and β shapes based on input dimensions
        if x.dim() == 4:  # For convolutional layers: [batch_size, C, H, W]
            gamma = self.γ.view(1, -1, 1, 1)
            beta = self.β.view(1, -1, 1, 1)
        elif x.dim() == 2:  # For fully connected layers: [batch_size, C]
            gamma = self.γ
            beta = self.β
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        # Final output with affine transformation
        final_output = gamma * tanh_output + beta
        return final_output, tanh_output  # Return both for visualization


class ResNetDyT(nn.Module):
    def __init__(self):
        super(ResNetDyT, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # Input: 3x32x32, Output: 6x28x28
        self.dyt1 = DyT(C=6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # Output: 16x10x10
        self.dyt2 = DyT(C=16)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dyt3 = DyT(C=120)
        self.fc2 = nn.Linear(120, 84)
        self.dyt4 = DyT(C=84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        dyt_data = []  # To store (input, tanh_output) for each DyT layer

        # Conv1 + DyT1
        x = self.conv1(x)
        x_dyt1 = x
        x, tanh_dyt1 = self.dyt1(x)
        dyt_data.append((x_dyt1, tanh_dyt1))
        x = self.pool(F.relu(x))

        # Conv2 + DyT2
        x = self.conv2(x)
        x_dyt2 = x
        x, tanh_dyt2 = self.dyt2(x)
        dyt_data.append((x_dyt2, tanh_dyt2))
        x = self.pool(F.relu(x))

        # Flatten
        x = x.view(-1, 16 * 5 * 5)

        # FC1 + DyT3
        x = self.fc1(x)
        x_dyt3 = x
        x, tanh_dyt3 = self.dyt3(x)
        dyt_data.append((x_dyt3, tanh_dyt3))
        x = F.relu(x)

        # FC2 + DyT4
        x = self.fc2(x)
        x_dyt4 = x
        x, tanh_dyt4 = self.dyt4(x)
        dyt_data.append((x_dyt4, tanh_dyt4))
        x = F.relu(x)

        # Output layer
        x = self.fc3(x)
        return x, dyt_data



# Custom LayerNorm with tracking
def track_layer_norm(name):
    class TrackedLayerNorm(nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.norm = nn.LayerNorm(normalized_shape)

        def forward(self, x):
            input_mean, input_std = x.mean().item(), x.std().item()
            x = self.norm(x)
            output_mean, output_std = x.mean().item(), x.std().item()
            print(f"{name} - Input Mean: {input_mean:.4f}, Std: {input_std:.4f}")
            print(f"{name} - Output Mean: {output_mean:.4f}, Std: {output_std:.4f}")
            return x, (input_mean, input_std, output_mean, output_std)
    return TrackedLayerNorm

# DyT Module
"""
class DyT(nn.Module):
    def __init__(self, C, init_α=0.5):
        super(DyT, self).__init__()
        self.α = nn.Parameter(torch.ones(1) * init_α)  # Scalar learnable parameter
        self.γ = nn.Parameter(torch.ones(C))  # Per-channel scaling
        self.β = nn.Parameter(torch.zeros(C))  # Per-channel shift

    def forward(self, x):
        tanh_output = torch.tanh(self.α * x)
        if x.dim() == 4:
            gamma = self.γ.view(1, -1, 1, 1)
            beta = self.β.view(1, -1, 1, 1)
        elif x.dim() == 2:
            gamma = self.γ
            beta = self.β
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        final_output = gamma * tanh_output + beta
        return final_output
"""
# Define two CNNs, one with LayerNorm and one with DyT
def cnn_layernorm():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        track_layer_norm("LN1")(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        track_layer_norm("LN2")(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        track_layer_norm("LN3")(128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def cnn_dyt():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        DyT(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        DyT(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        DyT(128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )