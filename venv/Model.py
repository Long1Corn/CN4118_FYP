import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from Net import net

writer = SummaryWriter('E:/PyProjects/runs/mymodel')

torch.manual_seed(11)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = numpy.genfromtxt('E:\PyProjects\data\Classification\Dataset.csv', delimiter=',', dtype=float)
dataset = torch.from_numpy(dataset)
trainset = torch.cat((dataset[:550,:], dataset[1125:,:]), 0)
testset = dataset[875:1125,:]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=0)

classes = ('R', 'L')

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=350, gamma=0.7)
net.eval()
for epoch in range(12):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        # labels = torch.zeros(5,2).to(device=device, dtype=torch.long)
        # for m in range(5):
        #     labels[m, data[m,0].to(dtype=torch.long)]= 1
        labels = data[:,0].to(device=device, dtype=torch.long)
        inputs = data[:, 1:].view(5, 1, 90, -1).to(device=device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()

        net.eval()
        if i % 25 == 24:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #       (epoch + 1, i + 1, running_loss / 25))
            writer.add_scalar('training loss',
                            running_loss / 25,
                            epoch * len(trainloader) + i)

            running_loss = 0.0

            class_correct = list(0. for i in range(2))
            class_total = list(0. for i in range(2))

            test_loss = 0.0
            with torch.no_grad():
                for data in testloader:
                    labels = data[:, 0].to(device=device, dtype=torch.long)
                    images = data[:, 1:].view(5, 1, 90, -1).to(device=device, dtype=torch.float)
                    outputs = net(images)
                    test_loss = test_loss + criterion(outputs, labels)

            print(test_loss/50.0)
            writer.add_scalar('test loss',
                            test_loss / 50.0,
                            epoch * len(trainloader) + i)



print('Finished Training')

PATH = 'E:\PyProjects\data\Classification\mymodel.pth'
torch.save(net.state_dict(), PATH)

# print(data.shape)
# print(data[0,1:].shape)
#
# print(data[0,0])
# img= data[0][1:]
# img= numpy.reshape(img,(-1,90))
#
# imgplot = plt.imshow(img,cmap='gray')
# plt.show()
