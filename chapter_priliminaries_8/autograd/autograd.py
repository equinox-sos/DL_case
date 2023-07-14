import torch    #评价为知道pytorch能自动求导就行了
x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
print(x.grad)
y = 2 * torch.dot(x, x)
print(y)
y.backward()    #每次输出梯度之前需要进行一次反向传播是因为求梯度代价高，不会默认求
print(x.grad)
print(x.grad == 4 * x)
x.grad.zero_()  #如果把这行注释掉则下面的运算会累积之前的梯度
y = x.sum() #y是向量(x1, x2, x3, x4)
y.backward()
print(x.grad)
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)
x.grad.zero_()  #以下操作可以将对u的计算从已经建立的计算图(关于y和x的)中移除
y = x * x   #y是向量(x1*x1, x2*x2, x3*x3, x4*x4)
u = y.detach()  #u去除了y对x的依赖，现在是一个向量(0, 1, 4, 9)，这一步可以用来固定网络参数
z = u * x
z.sum().backward()  #x是向量，这里z.sum()是标量(u1x1+u2x2+u3x3+u4x4)
z.sum().backward()
print(x.grad)