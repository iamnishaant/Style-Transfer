import torch

class StyleEngine:
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device

    def stylize(self, tensor):
        with torch.no_grad():
            return self.model(tensor)
