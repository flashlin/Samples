from pytorch_forecasting.metrics import MultiHorizonMetric

# Implement a new metric
class MAE(MultiHorizonMetric):
    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss