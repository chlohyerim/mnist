# PyTorch 불러오기
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim

from torchvision import datasets
from torchvision.transforms import ToTensor

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# training & test data 불러오기
train_data = datasets.MNIST(root='./data',
                            train=True,
                            download=True,
                            transform=ToTensor())
test_data = datasets.MNIST(root='./data',
                           train=False,
                           download=True,
                           transform=ToTensor())

# hyper-param
_batch_size = 64
learning_rate = 0.001
n_epoch = 10

# batch size = 64인 mini batch 구성
# test data는 섞지 않는다.
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=_batch_size,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=_batch_size,
                             shuffle=False)

train_features, train_labels = next(iter(train_dataloader))

# 모델 구성
class Model(nn.Module):
    # layer 정의
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
                               
        self.fc1 = nn.Linear(3136, 256)  # 28 * 28 / 2 / 2 * 64 = 3136
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout2d(0.5)
    
    # layer 쌓기
    def forward(self, x):
        # input data dimension: 28 * 28

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # max pooling -> / (2 * 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # max pooling -> / (2 * 2)

        x = torch.flatten(x, start_dim=1)  # conv2의 output channel = 64 -> * 64

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
model_state_dict_filepath = './model_state_dict.pt'

if os.path.exists(model_state_dict_filepath):
    checkpoint = torch.load(model_state_dict_filepath)

    model.load_state_dict(checkpoint['model_state_dict'])
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # 트레이닝 모드로 전환
    model.train()

    step = 1
    loss_min = None
    model_state_dict_tosave = None

    # train
    for epoch in range(n_epoch):
        for data, target in train_dataloader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()  # 누적되는 gradient 값 초기화

            model_output = model(data)
            loss = loss_fn(model_output, target)

            loss.backward()  # backward propagation
            optimizer.step()

            is_updating = False

            # loss_min의 값이 없거나 현재 iteration의 loss 값이 loss_min보다 작으면 loss_min 값 및 저장할 모델을 업데이트
            if loss_min == None or loss_min > loss.item():
                is_updating = True
                loss_min = loss.item()
                model_state_dict_tosave = model.state_dict()

            if step % 1000 == 0:
                print("Step: {}\tLoss: {}\tUpdating model to save: {}".format(step, loss.item(), is_updating))

            step += 1

    torch.save({'model_state_dict': model_state_dict_tosave}, model_state_dict_filepath)

# 평가 모드로 전환
model.eval()

probas = []
preds = []
targets = []

for data, target in test_dataloader:
    data = data.to(device)
    target = target.to(device)

    output = model(data)  # test data를 모델에 feed
    proba = (nn.functional.softmax(output, dim=1)).data.cpu().numpy()

    probas.extend(proba)

    pred = (torch.max(torch.exp(output), dim=1)[1]).data.cpu().numpy()  # arg-max

    preds.extend(pred)

    target = target.data.cpu().numpy()
    
    targets.extend(target)

cf_matrix = confusion_matrix(targets, preds)  # sklearn.metrics의 confusion matrix 라이브러리 기준 targets가 세로, preds가 가로
df_cf_matrix = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1))

df_cf_matrix.index = ['True ' + str(i) for i in range(10)]
df_cf_matrix.columns = ['Predicted ' + str(i) for i in range(10)]

accuracy = accuracy_score(targets, preds)

print('Accuracy: {}'.format(accuracy))

fig1 = plt.figure(figsize=(12, 9))
fig1 = sns.heatmap(df_cf_matrix, annot=True)
fig1 = plt.title('Confusion Matrix')

infer_index = 42
infer_image, infer_label = test_data[infer_index]

fig2 = plt.figure(figsize=(12, 6))
fig2 = plt.subplot(1, 3, 1)
fig2 = plt.axis('off')
fig2 = plt.imshow(infer_image.squeeze().numpy(), cmap='gray')
fig2 = plt.subplot(1, 3, 2)
fig2 = plt.table([['Predicted', preds[infer_index]], ['True', infer_label]], loc='center')
fig2 = plt.axis('off')
fig2 = plt.title('Inference Test Data ' + str(infer_index))
fig2 = plt.subplot(1, 3, 3)
fig2 = plt.table(np.swapaxes([['Proba ' + str(i) for i in range(10)], probas[infer_index]], 0, 1), loc='center')
fig2 = plt.axis('off')

plt.show()