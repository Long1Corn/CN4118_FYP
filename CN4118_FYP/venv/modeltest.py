import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Net import net

torch.manual_seed(11)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = numpy.genfromtxt('E:\PyProjects\data\Classification\Dataset.csv', delimiter=',', dtype=float)
validationset= dataset[550:875,:]

validationloader = torch.utils.data.DataLoader(validationset, batch_size=5, shuffle=False, num_workers=0)

classes = ('R', 'L')

net.load_state_dict(torch.load('E:\PyProjects\data\Classification\mymodel.pth'))
net.to(device)
criterion = nn.CrossEntropyLoss()

val_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in validationloader:
        labels = data[:, 0].to(device=device, dtype=torch.long)
        images = data[:, 1:].view(5, 1, 90, -1).to(device=device, dtype=torch.float)
        outputs = net(images)

        val_loss = val_loss + criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(5):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(2):
    print('Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

print(val_loss/115.0)



