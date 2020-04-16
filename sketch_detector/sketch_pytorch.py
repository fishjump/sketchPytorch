import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class sketch_pytorch(nn.Module):
    def __init__(self, weight_file):
        super(sketch_pytorch, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_1 = self.__batch_normalization(2, 'batch_normalization_1', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_2 = self.__batch_normalization(2, 'batch_normalization_2', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_3 = self.__conv(2, name='conv2d_3', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_3 = self.__batch_normalization(2, 'batch_normalization_3', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_4 = self.__conv(2, name='conv2d_4', in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_4 = self.__batch_normalization(2, 'batch_normalization_4', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_5 = self.__conv(2, name='conv2d_5', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_5 = self.__batch_normalization(2, 'batch_normalization_5', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_6 = self.__conv(2, name='conv2d_6', in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_6 = self.__batch_normalization(2, 'batch_normalization_6', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_7 = self.__conv(2, name='conv2d_7', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_7 = self.__batch_normalization(2, 'batch_normalization_7', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_8 = self.__conv(2, name='conv2d_8', in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_8 = self.__batch_normalization(2, 'batch_normalization_8', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_9 = self.__conv(2, name='conv2d_9', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_9 = self.__batch_normalization(2, 'batch_normalization_9', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_10 = self.__conv(2, name='conv2d_10', in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_10 = self.__batch_normalization(2, 'batch_normalization_10', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_11 = self.__conv(2, name='conv2d_11', in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_11 = self.__batch_normalization(2, 'batch_normalization_11', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_12 = self.__conv(2, name='conv2d_12', in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_12 = self.__batch_normalization(2, 'batch_normalization_12', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_13 = self.__conv(2, name='conv2d_13', in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_13 = self.__batch_normalization(2, 'batch_normalization_13', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_14 = self.__conv(2, name='conv2d_14', in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_14 = self.__batch_normalization(2, 'batch_normalization_14', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_15 = self.__conv(2, name='conv2d_15', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_15 = self.__batch_normalization(2, 'batch_normalization_15', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_16 = self.__conv(2, name='conv2d_16', in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_16 = self.__batch_normalization(2, 'batch_normalization_16', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_17 = self.__conv(2, name='conv2d_17', in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_17 = self.__batch_normalization(2, 'batch_normalization_17', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_18 = self.__conv(2, name='conv2d_18', in_channels=64, out_channels=1, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        conv2d_1_pad    = F.pad(x, (1, 1, 1, 1))
        conv2d_1        = self.conv2d_1(conv2d_1_pad)
        batch_normalization_1 = self.batch_normalization_1(conv2d_1)
        activation_1    = F.relu(batch_normalization_1)
        conv2d_2_pad    = F.pad(activation_1, (1, 1, 1, 1))
        conv2d_2        = self.conv2d_2(conv2d_2_pad)
        batch_normalization_2 = self.batch_normalization_2(conv2d_2)
        activation_2    = F.relu(batch_normalization_2)
        conv2d_3_pad    = F.pad(activation_2, (1, 1, 1, 1))
        conv2d_3        = self.conv2d_3(conv2d_3_pad)
        batch_normalization_3 = self.batch_normalization_3(conv2d_3)
        activation_3    = F.relu(batch_normalization_3)
        conv2d_4_pad    = F.pad(activation_3, (1, 1, 1, 1))
        conv2d_4        = self.conv2d_4(conv2d_4_pad)
        batch_normalization_4 = self.batch_normalization_4(conv2d_4)
        activation_4    = F.relu(batch_normalization_4)
        conv2d_5_pad    = F.pad(activation_4, (1, 1, 1, 1))
        conv2d_5        = self.conv2d_5(conv2d_5_pad)
        batch_normalization_5 = self.batch_normalization_5(conv2d_5)
        activation_5    = F.relu(batch_normalization_5)
        conv2d_6_pad    = F.pad(activation_5, (1, 1, 1, 1))
        conv2d_6        = self.conv2d_6(conv2d_6_pad)
        batch_normalization_6 = self.batch_normalization_6(conv2d_6)
        activation_6    = F.relu(batch_normalization_6)
        conv2d_7_pad    = F.pad(activation_6, (1, 1, 1, 1))
        conv2d_7        = self.conv2d_7(conv2d_7_pad)
        batch_normalization_7 = self.batch_normalization_7(conv2d_7)
        activation_7    = F.relu(batch_normalization_7)
        conv2d_8_pad    = F.pad(activation_7, (1, 1, 1, 1))
        conv2d_8        = self.conv2d_8(conv2d_8_pad)
        batch_normalization_8 = self.batch_normalization_8(conv2d_8)
        activation_8    = F.relu(batch_normalization_8)
        conv2d_9_pad    = F.pad(activation_8, (1, 1, 1, 1))
        conv2d_9        = self.conv2d_9(conv2d_9_pad)
        batch_normalization_9 = self.batch_normalization_9(conv2d_9)
        activation_9    = F.relu(batch_normalization_9)
        concatenate_1   = torch.cat((activation_8, activation_9), 1)
        conv2d_10_pad   = F.pad(nn.Upsample(scale_factor=2)(concatenate_1), (1, 2, 1, 2))
        conv2d_10       = self.conv2d_10(conv2d_10_pad)
        batch_normalization_10 = self.batch_normalization_10(conv2d_10)
        activation_10   = F.relu(batch_normalization_10)
        conv2d_11_pad   = F.pad(activation_10, (1, 1, 1, 1))
        conv2d_11       = self.conv2d_11(conv2d_11_pad)
        batch_normalization_11 = self.batch_normalization_11(conv2d_11)
        activation_11   = F.relu(batch_normalization_11)
        concatenate_2   = torch.cat((activation_7, activation_11), 1)
        conv2d_12_pad   = F.pad(nn.Upsample(scale_factor=2)(concatenate_2), (1, 2, 1, 2))
        conv2d_12       = self.conv2d_12(conv2d_12_pad)
        batch_normalization_12 = self.batch_normalization_12(conv2d_12)
        activation_12   = F.relu(batch_normalization_12)
        conv2d_13_pad   = F.pad(activation_12, (1, 1, 1, 1))
        conv2d_13       = self.conv2d_13(conv2d_13_pad)
        batch_normalization_13 = self.batch_normalization_13(conv2d_13)
        activation_13   = F.relu(batch_normalization_13)
        concatenate_3   = torch.cat((activation_5, activation_13), 1)
        conv2d_14_pad   = F.pad(nn.Upsample(scale_factor=2)(concatenate_3), (1, 2, 1, 2))
        conv2d_14       = self.conv2d_14(conv2d_14_pad)
        batch_normalization_14 = self.batch_normalization_14(conv2d_14)
        activation_14   = F.relu(batch_normalization_14)
        conv2d_15_pad   = F.pad(activation_14, (1, 1, 1, 1))
        conv2d_15       = self.conv2d_15(conv2d_15_pad)
        batch_normalization_15 = self.batch_normalization_15(conv2d_15)
        activation_15   = F.relu(batch_normalization_15)
        concatenate_4   = torch.cat((activation_3, activation_15), 1)
        conv2d_16_pad   = F.pad(nn.Upsample(scale_factor=2)(concatenate_4), (1, 2, 1, 2))
        conv2d_16       = self.conv2d_16(conv2d_16_pad)
        batch_normalization_16 = self.batch_normalization_16(conv2d_16)
        activation_16   = F.relu(batch_normalization_16)
        conv2d_17_pad   = F.pad(activation_16, (1, 1, 1, 1))
        conv2d_17       = self.conv2d_17(conv2d_17_pad)
        batch_normalization_17 = self.batch_normalization_17(conv2d_17)
        activation_17   = F.relu(batch_normalization_17)
        concatenate_5   = torch.cat((activation_1, activation_17), 1)
        conv2d_18_pad   = F.pad(concatenate_5, (1, 1, 1, 1))
        conv2d_18       = self.conv2d_18(conv2d_18_pad)
        return conv2d_18


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']).permute((3,2,0,1)))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer