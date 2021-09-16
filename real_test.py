import torch
import torch.nn as nn

class ResNetCAM(nn.Module): 
 
    def __init__(self): 
        super(ResNetCAM, self).__init__() 
 
        self.fc1 = torch.nn.Conv2d(2048, 128, 1, stride=1, padding=0, dilation = 0, bias=True)
        self.fc2 = torch.nn.Conv2d(128, 3, 1, stride=1, padding=0, dilation = 0, bias=True)
        self.fc3 = nn.Linear(2048, 128)
        self.fc4 = nn.Linear(128, 3)
 
    def forward(self, x): 
        result = self.fc1(x)
        result = self.fc2(result)

        return result



# model = ResNetCAM()
# print(model.state_dict().keys())
# for i in model.state_dict().keys():
#     print(model.state_dict()[i].shape)

# for i in model.children():
#     print(i)
a = torch.tensor([[1,2,3], [2,3,4]])
b = a.unsqueeze(-1).unsqueeze(-1)
print(b.shape)
print(b)