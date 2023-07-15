from torch.utils.data import DataLoader
from dataset import Dataset_My
from model import PL_Model
from pytorch_lightning.trainer import Trainer


train_data = Dataset_My(flag="train")
val_data = Dataset_My(flag="val")
test_data = Dataset_My(flag="test")
train_dataloader = DataLoader(dataset=train_data, batch_size=32)
val_dataloader = DataLoader(dataset=val_data, batch_size=32)
test_dataloader = DataLoader(dataset=test_data, batch_size=32)

in_size, out_size = train_data.getsize()

model = PL_Model(insize=in_size, outsize=out_size)
trainer = Trainer(max_epochs=50)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
