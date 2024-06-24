import torch
from torch.utils.data import Dataset


class TestModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__()

        self.conv = torch.nn.Conv2d(1, 1, 3, 1, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestLoss:
    def __init__(self):
        pass

    def __call__(self, pred, gts):
        mse = test_loss(pred, gts).mean()
        sad = torch.abs(pred - gts).mean()
        # return torch.mean(res)
        return {
            'mse': mse,
            'sad': sad,
        }

def test_loss(pred, gts):
    return (pred - gts) ** 2

#
# def abs(pred, gts):
#     return (pred - gts) ** 2


class TestMetrics:
    def __init__(self):
        pass

    def __call__(self, pred, true):
        return test_metrics(pred, true)


def test_metrics(pred, gts):
    return torch.sum(torch.abs(pred - gts) > 0.2) / pred.numel()


class TestDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self._len = 10
        self.idx = 0

    def __len__(self):
        return self._len

    def __next__(self):
        self.idx = (self.idx + 1) % self._len
        return self[self.idx]

    def __getitem__(self, item):
        return (torch.randn(1, 20, 20), torch.randn(1, 20, 20))

    def initialize_detection_model(self, obj):
        pass

    def prediction_collate(self, pred):
        pass

