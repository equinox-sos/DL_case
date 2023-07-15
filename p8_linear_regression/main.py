from dataset import Dataset_My


train_data = Dataset_My(flag="train")
val_data = Dataset_My(flag="val")
test_data = Dataset_My(flag="test")

print(len(train_data), len(val_data), len(test_data))