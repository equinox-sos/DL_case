from torch.utils.data import DataLoader
from dataset import Dataset_My
from model import PL_Model
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

log = TensorBoardLogger(save_dir=r"./log")

train_data = Dataset_My(flag="train")
val_data = Dataset_My(flag="val")
test_data = Dataset_My(flag="test")
train_dataloader = DataLoader(dataset=train_data, batch_size=16)
val_dataloader = DataLoader(dataset=val_data, batch_size=16)
test_dataloader = DataLoader(dataset=test_data, batch_size=16)

in_size, out_size = train_data.getsize()

model = PL_Model(insize=in_size, outsize=out_size)
trainer = Trainer(max_epochs=50, logger=log, log_every_n_steps=5)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, dataloaders=test_dataloader)

yt = []
yp = []
for i in range(len(test_data)):
    x, y = test_data[i]
    yt.append(y.item())
    yp.append(model.forecast(x).item())

# 创建 x 轴坐标
x = range(len(yt))

# 绘制 yt 和 yp 的折线图
plt.plot(x, yt, label='yt')
plt.plot(x, yp, label='yp')

# 添加图例和标签
plt.legend()
plt.xlabel('x')
plt.ylabel('Value')

# 显示图像
plt.show()
