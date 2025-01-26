import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torchvision

class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 50) # Linear layers
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss() # Loss function we want to use!
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        # self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        # accuracy = self.my_accuracy(scores, y)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
          {
            'train_loss': loss,
            'train_accuraccy': accuracy,
            'train_f1_score': f1_score
          },
          on_step=False,
          on_epoch=True,
          prog_bar=True
        )

        if batch_idx % 100 == 0:
          x = x[:8]
          grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
          self.logger.experiment.add_image("mnist_images", grid, self.global_step)
        return {'loss': loss, 'scores': scores, 'y': y}


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x) # Calculate scores for input values
        loss = self.loss_fn(scores, y) # Find loss after calculating scores
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    # Return optimizer
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)