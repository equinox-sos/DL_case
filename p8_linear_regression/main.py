from torch.utils.data import DataLoader
from dataset import Dataset_My, Test_My
from model import PL_Model
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd

log = TensorBoardLogger(save_dir=r"./log")

train_data = Dataset_My(flag="train")
val_data = Dataset_My(flag="val")
test_data = Dataset_My(flag="test") 
pre_data = Test_My()
train_dataloader = DataLoader(dataset=train_data, batch_size=16)
val_dataloader = DataLoader(dataset=val_data, batch_size=16)
test_dataloader = DataLoader(dataset=test_data, batch_size=16)

in_size, out_size = train_data.getsize()

checkpoint = ModelCheckpoint(
    monitor="val_loss", 
    save_top_k=1, 
    mode="min", 
    save_last=True,
)

model = PL_Model(insize=in_size, outsize=out_size)
trainer = Trainer(
    max_epochs=10000, 
    logger=log, 
    log_every_n_steps=5, 
    callbacks=[checkpoint]
    )

trainer.fit(
    model=model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader, 
    ckpt_path=r'./log/lightning_logs/version_29/checkpoints/last.ckpt'
    )

trainer.test(model, dataloaders=test_dataloader)

bestmodel = PL_Model.load_from_checkpoint(checkpoint_path=checkpoint.best_model_path, insize=in_size, outsize=out_size)
y = []
index = []
for i in range(len(pre_data)):
    y.append(bestmodel.forecast(pre_data[i]).item())
    index.append("id_" + str(i + 1))

data_pd = pd.DataFrame()
data_pd['ID'] = index
data_pd['value'] = y
path = r'../dataset/boston_housing_data/predition.csv'
data_pd.to_csv(path, index=False)