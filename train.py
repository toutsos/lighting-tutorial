import pytorch_lightning as pl
import torch
from model import NN
from dataset import MnistDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler



if __name__ == '__main__':

  logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")

  profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule = torch.profiler.schedule(
          skip_first=10,
          wait=1,
          warmup=1,
          active=20,
          repeat=1
        ),
        record_shapes = True,
        profile_memory = True,
    )


  model = NN(
    input_size=config.INPUT_SIZE,
    learning_rate=config.LEARNING_RATE,
    num_classes=config.NUM_CLASSES
  )

  # Data Module
  dm = MnistDataModule(
    data_dir=config.DATA_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS
  )

  # trainer = pl.Trainer(accelerator="gpu", devices=[1], min_epochs=1, max_epochs=3, precision=16, num_nodes=1)
  trainer = pl.Trainer(
    accelerator=config.ACCELERATOR,
    profiler=profiler,
    logger=logger,
    min_epochs=1,
    max_epochs=config.NUM_EPOCHS,
    callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")]
  )
  # trainer.tune()

  trainer.fit(model, dm) # Runs validation on each Epoch
  trainer.validate(model, dm)
  trainer.test(model, dm)


  # tensorboard --logdir = tb_logs --bind_all