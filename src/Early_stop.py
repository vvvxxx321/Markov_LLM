class EarlyStopper:
    """
    Minimal early-stopping helper (pure PyTorch).
    mode='min' for metrics you want to minimize (MAE/MSE/CRPS/CE).
    """
    def __init__(self, patience=8, min_delta=0.0, mode="min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.count = 0

    def step(self, value: float):
        improved = False
        if self.best is None:
            self.best = value
            self.count = 0
            improved = True
        else:
            if self.mode == "min":
                if value < self.best - self.min_delta:
                    self.best = value
                    self.count = 0
                    improved = True
                else:
                    self.count += 1
            else:
                if value > self.best + self.min_delta:
                    self.best = value
                    self.count = 0
                    improved = True
                else:
                    self.count += 1
        stop = self.count > self.patience
        return improved, stop