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


class split_block(nn.Module):
    def __init__(self, input_dim, number):
        super(split_block, self).__init__()
        self.conv1x1 = nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1))
        self.conv = nn.Conv2d(input_dim, input_dim, kernel_size=(number, number), padding=number//2)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv(x)
        return x


class attention(nn.Module):
    def __init__(self, input_dim):
        super(attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_share = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16, input_dim),
            nn.Softmax(dim=1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(16, input_dim),
            nn.Softmax(dim=1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(16, input_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, in_x):
        x = in_x[0]+in_x[1]+in_x[2]
        b, c, _, _ = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc_share(x)
        x1 = self.fc1(x).view(b, c, 1, 1)
        x2 = self.fc2(x).view(b, c, 1, 1)
        x3 = self.fc3(x).view(b, c, 1, 1)
        x1 = x1*in_x[0]
        x2 = x2*in_x[1]
        x3 = x3*in_x[2]
        return x1+x2+x3

class cardinal_block(nn.Module):
    def __init__(self, input_dim):
        super(cardinal_block, self).__init__()
        self.s1 = split_block(input_dim, 3)
        self.s2 = split_block(input_dim, 5)
        self.s3 = split_block(input_dim, 7)
        self.sp_a = attention(input_dim)

    def forward(self, x):
        x1 = self.s1(x)
        x2 = self.s2(x)
        x3 = self.s3(x)
        return self.sp_a([x1, x2, x3])


class REsNeSt_block(nn.Module):
    def __init__(self, input_dim):
        super(REsNeSt_block, self).__init__()
        self.cardinal1 = cardinal_block(input_dim)
        self.cardinal2 = cardinal_block(input_dim)
        self.conv1x1 = nn.Conv2d(input_dim*2, input_dim, kernel_size=(1, 1))

    def forward(self, in_x):
        x1 = self.cardinal1(in_x)
        x2 = self.cardinal2(in_x)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1x1(x)
        x = x+in_x
        return x

class detail_net(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64, resunit_num=16):
        super(detail_net, self).__init__()
        self.input_channel_num = input_channel_num
        self.feature_dim = feature_dim
        self.resunit_num = resunit_num

        self.detail_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel_num, self.feature_dim, kernel_size=(3, 3), padding=1),
            nn.PReLU(self.feature_dim)
        )

        self.detail_rrb = nn.ModuleList([
            REsNeSt_block(self.feature_dim)
            for _ in range(self.resunit_num)]
        )

        self.detail_conv_2 = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.feature_dim)
        )

        self.detail_conv_3 = nn.Conv2d(self.feature_dim, self.input_channel_num, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.detail_conv_1(x)
        x0 = x
        for resblock in self.detail_rrb:
            x = resblock(x)
        x = self.detail_conv_2(x)
        x = x + x0
        x = self.detail_conv_3(x)
        return x

if __name__ == '__main__':
    print(7//2)
    # model = drd_net()
    # for name, param in model.named_parameters():
    #     print(name, '      ', param.size())
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = torch.ones((1, 3, 64, 64))
    # inputs = inputs.to(device)
    # model = model.to(device)
    # print(model(inputs))
