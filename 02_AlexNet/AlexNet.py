import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def build_fashionmnist_dataloader(batch_size, num_workers=0, root=f'{base_path}/Datasets/FashionMNIST', resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=trans)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net, device='cpu'):
    predicted_correctness = []
    for X, y in data_iter:
        X = X.to(device)    # (bsz, 1, 28, 28)
        y = y.to(device)    # (bsz, )
        y_hat = net(X)      # (bsz, 10)
        predicted_correctness.extend((y_hat.argmax(dim=1) == y).tolist())
    return np.mean(predicted_correctness)

def train(net, train_iter, test_iter, num_epochs, lr, device='cpu'):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    # 参数初始化，转移到指定设备
    net.apply(init_weights)
    print('training on', device)
    net.to(device)

    # 定义优化器 & 损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # 开始训练
    logs = {'train_loss': [], 'train_acc':[], 'test_acc':[]}
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_correctness = []
        
        # 训练一个 epoch
        net.train()
        with tqdm(total=len(train_iter), desc=f'Epoch {epoch}') as pbar:
            for X, y in train_iter:
                X = X.to(device)    # (bsz, 1, 28, 28)
                y = y.to(device)    # (bsz, )
                y_hat = net(X)      # (bsz, 10)
                l = loss(y_hat, y)
                
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                
                epoch_losses.append(l.item())
                epoch_correctness.extend((y_hat.argmax(dim=1) == y).tolist())

                info = {
                    'train_loss': f'{np.mean(epoch_losses):.3f}',
                    'train_acc': f'{np.mean(epoch_correctness):.3f}',
                }
                pbar.set_postfix(info)
                pbar.update()

        # 统计该 epoch 的训练损失和平均精度
        train_loss = np.mean(epoch_losses)
        train_acc = np.mean(epoch_correctness)

        # 评估测试集上的准确率
        with torch.inference_mode():
            test_acc = evaluate_accuracy(test_iter, net, device)

        print(f'epoch {epoch}, loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')        
        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['test_acc'].append(test_acc)
        
    return logs

def show_log(logs:Dict, save_path):
    # 绘制所有数据
    epochs = range(len(list(logs.values())[0]))
    plt.figure(figsize=(10, 6))
    for label, data in logs.items():
        plt.plot(epochs, data, label=label, marker='o')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics')

    plt.legend()            # 添加图例
    plt.grid(True)          # 显示网格
    plt.savefig(save_path)  # 保存到指定路径

if __name__ == "__main__":
    # 参数准备
    lr = 0.01
    num_epochs = 10
    batch_size = 128
    resize = 224

    # 构造 LeNet 模型
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),   # torch.Size([1, 96, 54, 54])
        nn.MaxPool2d(kernel_size=3, stride=2),                              # torch.Size([1, 96, 26, 26])
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),            # torch.Size([1, 256, 26, 26])
        nn.MaxPool2d(kernel_size=3, stride=2),                              # torch.Size([1, 256, 12, 12])
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),           # torch.Size([1, 384, 12, 12])
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),           # torch.Size([1, 384, 12, 12])
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),           # torch.Size([1, 384, 12, 12])
        nn.MaxPool2d(kernel_size=3, stride=2),                              # torch.Size([1, 256, 5, 5])
        nn.Flatten(),                                                       # torch.Size([1, 6400])
        nn.Linear(6400, 4096), nn.ReLU(),                                   # torch.Size([1, 4096])
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),                                   # torch.Size([1, 4096])
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10)                                                 # torch.Size([1, 10])
    )

    # 构造 DataLoader，这里把 Fashion-MNIST 图像分辨率从 28x28 提升到 224x224，以适配 AlexNet 的输入尺寸
    train_iter, test_iter = build_fashionmnist_dataloader(batch_size=batch_size, resize=resize)
    
    # 开始训练
    device='cuda' if torch.cuda.is_available() else 'cpu'
    logs = train(net, train_iter, test_iter, num_epochs, lr, device=device)

    # 可视化并保存
    show_log(logs, save_path=f'{base_path}/02_AlexNet/train_log.png')

'''
training on cuda
Epoch 0: 100%|██████████| 469/469 [01:15<00:00,  6.19it/s, train_loss=1.314, train_acc=0.515]
epoch 0, loss 1.314, train acc 0.515, test acc 0.676
Epoch 1: 100%|██████████| 469/469 [01:13<00:00,  6.39it/s, train_loss=0.642, train_acc=0.759]
epoch 1, loss 0.642, train acc 0.759, test acc 0.784
Epoch 2: 100%|██████████| 469/469 [01:14<00:00,  6.29it/s, train_loss=0.530, train_acc=0.802]
epoch 2, loss 0.530, train acc 0.802, test acc 0.808
Epoch 3: 100%|██████████| 469/469 [01:15<00:00,  6.25it/s, train_loss=0.467, train_acc=0.826]
epoch 3, loss 0.467, train acc 0.826, test acc 0.828
Epoch 4: 100%|██████████| 469/469 [01:14<00:00,  6.30it/s, train_loss=0.428, train_acc=0.842]
epoch 4, loss 0.428, train acc 0.842, test acc 0.841
Epoch 5: 100%|██████████| 469/469 [01:14<00:00,  6.33it/s, train_loss=0.398, train_acc=0.855]
epoch 5, loss 0.398, train acc 0.855, test acc 0.841
Epoch 6: 100%|██████████| 469/469 [01:14<00:00,  6.32it/s, train_loss=0.378, train_acc=0.862]
epoch 6, loss 0.378, train acc 0.862, test acc 0.855
Epoch 7: 100%|██████████| 469/469 [01:14<00:00,  6.34it/s, train_loss=0.359, train_acc=0.868]
epoch 7, loss 0.359, train acc 0.868, test acc 0.858
Epoch 8: 100%|██████████| 469/469 [01:16<00:00,  6.13it/s, train_loss=0.346, train_acc=0.874]
epoch 8, loss 0.346, train acc 0.874, test acc 0.859
Epoch 9: 100%|██████████| 469/469 [01:16<00:00,  6.13it/s, train_loss=0.330, train_acc=0.880]
epoch 9, loss 0.330, train acc 0.880, test acc 0.866
'''