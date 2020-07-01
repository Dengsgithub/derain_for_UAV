import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNet(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64, resunit_num=16):
        super(ResNet, self).__init__()
        self.input_channel_num = input_channel_num
        self.feature_dim = feature_dim
        self.resunit_num = resunit_num

        self.rain_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel_num, self.feature_dim, kernel_size=(3, 3), padding=1),
            nn.PReLU(self.feature_dim)
        )

        self.rain_rrb_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.PReLU(feature_dim),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(feature_dim),
                SEBlock(feature_dim, 6)
            )
            for _ in range(self.resunit_num)]
        )

        self.rain_conv_2 = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.feature_dim)
        )

        self.rain_conv_3 = nn.Conv2d(self.feature_dim, self.input_channel_num, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.rain_conv_1(x)
        x0 = x
        for resblock in self.rain_rrb_1:
            x_in = x
            x = resblock(x)
            x = x + x_in
        x = self.rain_conv_2(x)
        x = x + x0
        x = self.rain_conv_3(x)
        return x


if __name__ == '__main__':
    model = DRDNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.ones((1, 3, 64, 64))
    inputs = inputs.to(device)
    model = model.to(device)
    print(model(inputs))
