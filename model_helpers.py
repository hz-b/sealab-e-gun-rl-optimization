import torch
from torch import nn
import math

class ScaledSigmoid(nn.Module):
    def __init__(self, scale_min=0.1, scale_max=0.9):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x):
        x = self.scale_min + (self.scale_max - self.scale_min) * x
        x = torch.sigmoid(x)
        x = (x - self.scale_min) / (self.scale_max - self.scale_min)
        return x

def create_sequential(input_length, output_length, layer_size, blow: int | float = 0, shrink_factor="log", activation_function=nn.ReLU(), last_activation=None, batch_norm=False, layer_norm=False):
    layers = [input_length]
    blow_disabled = blow == 1 or blow == 0
    if not blow_disabled:
        layers.append(input_length*blow)

    if shrink_factor == "log":
        add_layers = torch.logspace(math.log(layers[-1], 10), math.log(output_length,10), steps=layer_size+2-len(layers), base=10).long()
        # make sure the first and last element is correct, even though rounding
        if blow_disabled:
            add_layers[0] = input_length
        add_layers[-1] = output_length
    elif shrink_factor == "lin":
        add_layers = torch.linspace(layers[-1], output_length, steps=layer_size+2-len(layers)).long()
    else:
        shrink_factor = float(shrink_factor)
        new_length = layer_size+1-len(layers)
        add_layers = (torch.ones(new_length)*layers[-1] * ((torch.ones(new_length) * shrink_factor) ** torch.arange(new_length))).long()
        layers = torch.cat((torch.tensor([input_length]), add_layers))
        layers = torch.cat((layers, torch.tensor([output_length])))

    if not blow_disabled:
        layers = torch.tensor([layers[0]])
        layers = torch.cat((layers, add_layers))
    else:
       layers = add_layers

    nn_layers = []
    for i in range(len(layers)-1):
        nn_layers.append(nn.Linear(int(layers[i].item()), int(layers[i+1].item())))
        if not i == len(layers)-2:
            if layer_norm:
                nn_layers.append(nn.LayerNorm(int(layers[i+1].item())))
            nn_layers.append(activation_function)
            if batch_norm:
                nn_layers.append(nn.BatchNorm1d(int(layers[i+1].item())))
        if i == len(layers)-2 and last_activation is not None:
            nn_layers.append(last_activation)
    return nn.Sequential(*nn_layers)
