class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: int = 0):
        self.min_metric: float = 0.0
        self.count: int = 0
        self.patience = patience
        self.min_delta = min_delta
        self.stop = False

    def check_early_stop(self, metric: float):
        if metric < self.min_metric:
            self.min_metric = metric
            self.count = 0
        elif metric > self.min_metric + self.min_delta:
            self.count += 1
        self.stop = self.count >= self.patience
