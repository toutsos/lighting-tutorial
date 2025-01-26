from pytorch_lightning.callbacks import EarlyStopping, Callback

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_training_start(self, trainer, pl_module):
        print("Training is starting!")

    def on_training_end(self, trainer, pl_module):
        print("Training is done!")




