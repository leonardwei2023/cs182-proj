import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SharedMLP(nn.Sequential):
    def __init__(self, layer_sizes):
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Conv1d(layer_sizes[i], layer_sizes[i+1], 1))
        layers.append(nn.BatchNorm2d(layer_sizes[-1]))
        layers.append(nn.ReLU())
        super(SharedMLP, self).__init__(*layers)


class T_net(nn.Module):
    def __init__(self, size, dropout=0.7, bn_momentum=None):
        super(T_net, self).__init__()
        self.shared_mlp = SharedMLP([size, 64, 128, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        '''
            Input: B x size x N
        '''
        out = self.shared_mlp(x)
        out = F.relu(self.bn(self.fc1(out)))
        out = self.fc2(out)
        out = out.view(-1, 3, 3) + torch.eye(3).reshape((1,3,3)).repeat(x.size(0))
        return out

class InputTransform(nn.Module):
    def __init__(self):
        super(InputTransform, self).__init__()
        self.T_net = T_net(3)

    def forward(self, x):
        out = self.T_net(x)
        return torch.bmm(torch.transpose(input, 1, 2), out).transpose(1, 2)


class FeatureTransform(nn.Module):
    def __init__(self, size=64, reg=0.001):
        super(FeatureTransform, self).__init__()
        self.T_net = T_net(size)
        self.feat_size = size

    def forward(self, x):
        out = self.T_net(x)
        # TODO: Check the dim of regulariation loss
        loss = self.regularize(out)
        return loss
    
    def regularize(self, x):
        I = torch.eye(self.feat_size)[None, :, :].to(device)
        out = torch.bmm(x, x.transpose(2,1)) - I
        loss = torch.mean(torch.norm(out, dim=(1,2)))
        return loss




class ClassficationNN(nn.Sequential):
    def __init__(self, num_classes):
        super(ClassficationNN, self).__init__()
        self.input_transform = InputTransform()
        self.mlp_1 = SharedMLP([3, 64, 64])
        self.feature_transform = FeatureTransform()
        self.mlp_2 = SharedMLP([64, 64, 128, 1024])
        self.mlp_3 = SharedMLP([1024, 512, 256, num_classes])

    def forward(self, x):
        out = self.input_transform(x)
        out = self.mlp_1(out)
        out = self.feature_transform(out)
        out = self.mlp_2(out)
        out = F.max_pool2d(out, x.size(2))
        out = self.mlp_3(out)
        return out

class SegmentationNN(nn.Sequential):
    def __init__(self, num_features: int):
        super(SegmentationNN, self).__init__()
        self.input_transform = InputTransform()
        self.feature_transform = FeatureTransform()
        self.mlp_1 = SharedMLP([3, 64, 64])
        self.mlp_2 = SharedMLP([64, 64, 128, 1024])
        self.mlp_3 = SharedMLP([1088, 512, 256, 128])
        self.mlp_4 = SharedMLP([128, 128, num_features])
    def forward(self, x):
        out = self.input_transform(x)
        out = self.mlp_1(out)
        out = self.feature_transform(out)
        global_feature = self.mlp_2(out)
        global_feature = F.max_pool2d(global_feature, x.size(2))
        global_feature = global_feature.expand(-1, -1, x.size(-1))
        out = torch.cat([out, global_feature], 1)
        out = self.mlp_3(out)
        out = self.mlp_4(out)
        return out




if __name__ == '__main__':
    input_transform = T_net(3)
    print('Input Transform:\n', input_transform)

    feature_transform = T_net(64)
    print('Feature Transform:\n', feature_transform)

    
