from torch import nn
import torch


class Inception(nn.Module):
    def __init__(self, prev_layer_channels, n_1x1, n_3x3_reduce, n_3x3, n_5x5_reduce, n_5x5, n_pool_proj):
        super(Inception, self).__init__()
        self.first_conv_unit = nn.Conv2d(prev_layer_channels, n_1x1, (1, 1), padding='same')
        self.sec_conv_unit_begin = nn.Conv2d(prev_layer_channels, n_3x3_reduce, (1, 1), padding='same')
        self.sec_conv_unit_final = nn.Conv2d(n_3x3_reduce, n_3x3, (3, 3), padding='same')

        self.third_conv_unit_begin = nn.Conv2d(prev_layer_channels, n_5x5_reduce, (1, 1), padding='same')
        self.third_conv_unit_final = nn.Conv2d(n_5x5_reduce, n_5x5, (5, 5), padding='same')

        self.pool_proj_begin = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool_proj_final = nn.Conv2d(prev_layer_channels, n_pool_proj, (1, 1), padding='same')

    def forward(self, input_):
        first_conv_output = self.first_conv_unit(input_)

        sec_conv_output = self.sec_conv_unit_begin(input_)
        sec_conv_output = self.sec_conv_unit_final(sec_conv_output)

        third_conv_output = self.third_conv_unit_begin(input_)
        third_conv_output = self.third_conv_unit_final(third_conv_output)

        pool_proj_output = self.pool_proj_begin(input_)
        pool_proj_output = self.pool_proj_final(pool_proj_output)

        final_output = torch.cat([first_conv_output, sec_conv_output, third_conv_output, pool_proj_output], 1)
        return final_output
