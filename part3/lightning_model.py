import torch
import torch.nn.functional as F
import lightning as L


class LightningModelWrapper(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        return optimizer

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def supervised_forward(self, batch):
        y_hat = self(batch.x, batch.edge_index)
        y_hat = y_hat[:batch.batch_size]
        y = batch.y[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y.view(-1))
        accuracy = torch.mean(torch.argmax(y_hat, dim=-1).eq(y).float())
        return loss, accuracy

    def training_step(self, batch, _):
        loss, accuracy = self.supervised_forward(batch)
        self.log('train_loss', loss, batch_size=batch.batch_size, on_epoch=True)
        self.log('train_accuracy', accuracy, batch_size=batch.batch_size, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, accuracy = self.supervised_forward(batch)
        self.log('val_loss', loss, batch_size=batch.batch_size, on_epoch=True)
        self.log('val_accuracy', accuracy, batch_size=batch.batch_size, on_epoch=True)
        return loss