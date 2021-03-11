import torchvision
import torch.nn as nn


class Two_Stage_ResNet152(nn.Module):
    def __init__(self, model):
        super(Two_Stage_ResNet152, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        self.pool_layer = nn.MaxPool2d(32)
        self.Linear_layer_1 = nn.Linear(2048, 3)
        self.Linear_layer_2 = nn.Linear(2048, 3)
        self.Linear_layer_3 = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.transion_layer(x)
        x = self.pool_layer(x)
        x = x.view(x.size(0), -1)
        output_1 = self.Linear_layer_1(x)
        output_2 = self.Linear_layer_2(x)
        output_3 = self.Linear_layer_3(x)
        return output_1, output_2, output_3


if __name__ == "__main__":
    resnet = torchvision.models.resnet152(pretrained=True)
    model = Two_Stage_ResNet152(resnet)








