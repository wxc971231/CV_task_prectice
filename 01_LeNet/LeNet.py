import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import torchvision
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def build_fashionmnist_dataloader(batch_size, num_workers=0, root=f'{base_path}/Datasets/FashionMNIST'):
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net, device='cuda'):
    predicted_correctness = []
    for X, y in data_iter:
        X = X.to(device)    # (bsz, 1, 28, 28)
        y = y.to(device)    # (bsz, )
        y_hat = net(X)      # (bsz, 10)
        predicted_correctness.extend((y_hat.argmax(dim=1) == y).tolist())
    return np.mean(predicted_correctness)

def train(net, train_iter, test_iter, num_epochs, lr, device='cuda'):
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
    lr = 0.9
    num_epochs = 10
    batch_size = 256

    # 构造 LeNet 模型
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),    # output_size [bsz, 6, 28, 28]
        nn.AvgPool2d(kernel_size=2, stride=2),                      # output_size [bsz, 6, 14, 14]
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),              # output_size [bsz, 16, 10, 10]
        nn.AvgPool2d(kernel_size=2, stride=2),                      # output_size [bsz, 16, 5, 5]
        nn.Flatten(),                                               # output_size [bsz, 16 * 5 * 5]
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),                   # output_size [bsz, 120]
        nn.Linear(120, 84), nn.Sigmoid(),                           # output_size [bsz, 84]
        nn.Linear(84, 10)                                           # output_size [bsz, 10]
    )

    # 构造 DataLoader
    train_iter, test_iter = build_fashionmnist_dataloader(batch_size=batch_size)
    
    # 开始训练
    logs = train(net, train_iter, test_iter, num_epochs, lr, device='cuda')

    # 可视化并保存
    show_log(logs, save_path=f'{base_path}/01_LeNet/train_log.png')

