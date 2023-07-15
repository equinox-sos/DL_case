class A():
    def __init__(self, x):
        self.x = x

    def __getitem__(self, index):
        return index * 3 + 1
    
    def __len__(self):
        return 3
    
    def g(self):
        return self.x

a = A(1)
b = A(2)

print(A.g(b))