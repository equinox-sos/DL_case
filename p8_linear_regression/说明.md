### 关于数据集：

总感觉下载的数据集有问题，有好几行的价格都是缺失的

为了删掉最后一列是缺失值的行，需要使用dropna()函数，参数为subset=[data.columns[-1]]

如果缺失值没有出现在最后一列，需要用fillna(0)来将其填补为0

### 关于模型

直接一层linear，框架是用的pl框架，loss用了mse

### Tensorboard

又忘了tensorboard怎么用了，复习一下：

导入from pytorch_lightning.logger import TensorBoardLogger

实例化logger = TensorBoradLogger(path=路径)

传给trainer，在模型中可以直接self.log(名称，值)

查看的方法是命令行输入tensorboard log_dir="上面的路径"

然后浏览器localhost:6006


### dataset

两个接口：

\_\_getitem\_\_(self, index):

返回一个x, y，一般是tensor

\_\_len\_\_(self):

返回一个长度

### dataloader

一次返回batch个dataset

batch=32，还是x，y，但是会增加一维

32个getitem，聚合

x.shape=[12]------>x.shape[32, 12]


### trainer

fit，一个model，一个训练集，一个验证集

取出一个batch，丢给model的training_step

取出一个batch，丢给validation_step

返回loss，指定optimizor
