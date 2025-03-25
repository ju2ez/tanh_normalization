import torch
from torch import nn
import numpy as np
import random



def set_seed(seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_model_weights(models, init_type='kaiming'):
    # Given a list of torch models,
    # initialize the models with the same weights
    with torch.no_grad():
        for m0 in models[0].named_parameters():
            if "ln" in m0[0] or "DyT" in m0[0]:
                # if it is a layernorm or DyT layer, skip
                continue
            # check for pooling layers
            if "pool" in m0[0]:
                continue
            try:
                for m0 in models[0].named_parameters():  # Iterate through model 0 parameters
                    for m1 in models[1].named_parameters():  # Iterate through model 1 parameters
                        if m0[0] == m1[0]:  # If the layer names match
                            try:
                                if init_type == 'kaiming':
                                    # Apply Kaiming initialization
                                    nn.init.kaiming_normal_(m0[1])
                                    m1[1].copy_(m0[1].clone())
                                elif init_type == 'xavier':
                                    # Apply Xavier initialization
                                    nn.init.xavier_normal_(m0[1])
                                    m1[1].copy_(m0[1].clone())
                            except Exception:
                                # print(f"Initialization failed for {m0[0]}. Error: {f}")
                                # print("Copying weights instead")
                                # If initialization fails, just copy the weights
                                m1[1].copy_(m0[1].clone())
            except Exception as e:
                print(f"Error: {e}")
                print('Skipping weights synchronization')


def compare_weights(model1, model2):
    # check if the models have the same weights
    # up to tanh vs layernorm differences they should be the same
    layer = 0
    # get layer names for each model
    layers1 = [name for name, _ in model1.named_parameters()]
    layers2 = [name for name, _ in model2.named_parameters()]
    # check if the layers are the same
    for m1, m2 in zip(model1.parameters(), model2.parameters()):
        if layers1[layer] != layers2[layer]:
            if "ln" in layers1[layer] or "ln" in layers2[layer]:
                # layernorm can and should differ because it is
                # exchanged with the dynamic tanh
                continue
            print('Layer', layer, 'names are not equal')
            print(layers1[layer], layers2[layer])
            continue
        if not torch.equal(m1, m2):
            print('Layer', layer, 'weights are not equal')
            print(layers1[layer], layers2[layer])
            return False
        layer += 1
    return True