import torch

class Normalizer:
    def __init__(self, x, y, method="minmax"):
        assert method in ["minmax", "zscore"]
        self.method = method

        # Store normalization stats
        if method == "minmax":
            self.x_min = x.min(dim=0).values
            self.x_max = x.max(dim=0).values
            self.x_range = self.x_max - self.x_min

            self.y_min = y.min(dim=0).values
            self.y_max = y.max(dim=0).values
            self.y_range = self.y_max - self.y_min
        elif method == "zscore":
            self.x_mean = x.mean(dim=0)
            self.x_std = x.std(dim=0)
            self.y_mean = y.mean(dim=0)
            self.y_std = y.std(dim=0)

    def to(self, device):
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self

    def score_x(self, x):
        if self.method == "minmax":
            return (x - self.x_min) / self.x_range
        else:
            return (x - self.x_mean) / self.x_std

    def unscore_x(self, x):
        if self.method == "minmax":
            return x * self.x_range + self.x_min
        else:
            return x * self.x_std + self.x_mean

    def score_y(self, y):
        if self.method == "minmax":
            return (y - self.y_min) / self.y_range
        else:
            return (y - self.y_mean) / self.y_std

    def unscore_y(self, y):
        if self.method == "minmax":
            return y * self.y_range + self.y_min
        else:
            return y * self.y_std + self.y_mean
