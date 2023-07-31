from torch.utils.data import DataLoader
from dataset import Dataset_My, Test_My
from model import PL_Model
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
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

model = PL_Model(insize=in_size, outsize=out_size)
trainer = Trainer(
    max_epochs=10000, 
    logger=log, 
    log_every_n_steps=5, 
    
    )
trainer.fit(
    model=model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader, 
    ckpt_path=r'./log/lightning_logs/version_11/checkpoints/epoch=9999-step=200000.ckpt'
    )

trainer.test(model, dataloaders=test_dataloader)

y = []
index = []
for i in range(len(pre_data)):
    y.append(model.forecast(pre_data[i]).item())
    index.append("id_" + str(i + 1))

data_pd = pd.DataFrame()
data_pd['ID'] = index
data_pd['value'] = y
path = r'../dataset/boston_housing_data/predition.csv'
data_pd.to_csv(path, index=False)