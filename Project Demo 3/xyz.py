import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class PositionAttentionModule(nn.Module):
    """ Position attention module as described in the Dual Attention Network for scene segmentation """
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return out + x

class ChannelAttentionModule(nn.Module):
    """ Channel attention module """
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        self.position_attention = PositionAttentionModule(in_channels)
        self.channel_attention = ChannelAttentionModule(in_channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.position_attention(x)
        x = self.channel_attention(x)
        x = self.conv_block(x)
        return x



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class EnhancedSDI(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(EnhancedSDI, self).__init__()
        total_in_channels = sum(in_channels_list)
        self.conv = nn.Conv2d(total_in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, xs, anchor):
        target_size = anchor.shape[-1]
        concatenated = torch.cat([F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True) for x in xs], dim=1)
        output = self.conv(concatenated)
        return output


class EdgeAwareAttention(nn.Module):
    def __init__(self, channels):
        super(EdgeAwareAttention, self).__init__()
        self.channels = channels
        # Sobel operator kernels for edge detection
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view((1, 1, 3, 3)), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view((1, 1, 3, 3)), requires_grad=False)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-6
    def forward(self, x):
        # Apply Sobel filters to input
        edge_x = F.conv2d(x, self.sobel_x.repeat(self.channels, 1, 1, 1), padding=1, groups=self.channels)
        edge_y = F.conv2d(x, self.sobel_y.repeat(self.channels, 1, 1, 1), padding=1, groups=self.channels)
        # Combine gradients
        edges = torch.sqrt(torch.pow(edge_x, 2) + torch.pow(edge_y, 2) + self.eps)
        # Apply convolution and sigmoid activation
        attention = self.sigmoid(self.conv(edges))
        # Apply attention
        return x * attention

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(PixelShuffleUpsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.relu(x)
        return x
         
class ChannelAttentionBridge(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(ChannelAttentionBridge, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        total_in_channels = sum(in_channels_list)
        self.fc = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels // 8, total_in_channels, 1, bias=False)  # Note the use of total_in_channels
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        # Determine the target size based on the largest feature map
        target_size = inputs[0].size()[2:]

        # Resize all feature maps to the target size
        resized_inputs = [F.interpolate(input, size=target_size, mode='bilinear', align_corners=False) for input in inputs]

        # Concatenate the resized feature maps along the channel dimension
        x = torch.cat(resized_inputs, dim=1)

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return attention * x  # Ensure the dimensions match here




class SpatialAttentionBridge(nn.Module):
    def __init__(self, in_channels_list):
        super(SpatialAttentionBridge, self).__init__()
        self.conv1 = nn.Conv2d(len(in_channels_list), 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        # Determine the target size based on the largest feature map
        target_size = inputs[0].size()[2:]

        # Resize all feature maps to the target size
        resized_inputs = [F.interpolate(input, size=target_size, mode='bilinear', align_corners=False) for input in inputs]

        # Compute the mean along the channel dimension for each resized input
        x = torch.cat([torch.mean(input, dim=1, keepdim=True) for input in resized_inputs], dim=1)

        attention = self.conv1(x)
        attention = self.sigmoid(attention)
        return x * attention


class VMUNetV2(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 mid_channel=48,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 deep_supervision=True,
                 output_size=(256, 256)):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.output_size = output_size

        # SDI
        self.ca_1 = ChannelAttention(2 * mid_channel)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(4 * mid_channel)
        self.sa_2 = SpatialAttention()
        # TODO 320 or mid_channel * 8?
        self.ca_3 = ChannelAttention(8 * mid_channel)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(16 * mid_channel)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(2 * mid_channel, mid_channel, 1)
        self.Translayer_2 = BasicConv2d(4 * mid_channel, mid_channel, 1)
        self.Translayer_3 = BasicConv2d(8 * mid_channel, mid_channel, 1)
        self.Translayer_4 = BasicConv2d(16 * mid_channel, mid_channel, 1)

        # Update SDI with dynamic input channels
        self.sdi_1 = EnhancedSDI([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel], mid_channel)
        self.sdi_2 = EnhancedSDI([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel], mid_channel)
        self.sdi_3 = EnhancedSDI([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel], mid_channel)
        self.sdi_4 = EnhancedSDI([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel], mid_channel)

        self.pixel_shuffle_upsample1 = PixelShuffleUpsample(mid_channel, mid_channel, 2)
        self.pixel_shuffle_upsample2 = PixelShuffleUpsample(mid_channel, mid_channel, 2)
        self.pixel_shuffle_upsample3 = PixelShuffleUpsample(mid_channel, mid_channel, 2)
        self.pixel_shuffle_upsample4 = PixelShuffleUpsample(mid_channel, mid_channel, 2)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(mid_channel, num_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv6 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)

        # Edge Aware Attention
        self.eaa_1 = EdgeAwareAttention(2 * mid_channel)
        self.eaa_2 = EdgeAwareAttention(4 * mid_channel)
        self.eaa_3 = EdgeAwareAttention(8 * mid_channel)
        self.eaa_4 = EdgeAwareAttention(16 * mid_channel)
        self.da_1 = DANetHead(2 * mid_channel, 2 * mid_channel)
        self.da_2 = DANetHead(4 * mid_channel, 4 * mid_channel)
        self.da_3 = DANetHead(8 * mid_channel, 8 * mid_channel)
        self.da_4 = DANetHead(16 * mid_channel, 16 * mid_channel)

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate)

        # Using the correct in_channels_list for the bridges
        self.channel_attention_bridge = ChannelAttentionBridge([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel], sum([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel]))
        self.spatial_attention_bridge = SpatialAttentionBridge([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel])

        # Convolution layer to unify the number of channels after combining bridges
        self.bridge_conv = nn.Conv2d(sum([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel]) + len([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel]), sum([2 * mid_channel, 4 * mid_channel, 8 * mid_channel, 16 * mid_channel]), kernel_size=1)

    def forward(self, x):
        seg_outs = []
        if x.size()[1] == 1:  # If it's a grayscale image, convert 1 channel to 3 channels
            x = x.repeat(1, 3, 1, 1)
        f1, f2, f3, f4 = self.vmunet(x)  #  f1 [2, 64, 64, 96]  f3  [2, 8, 8, 768]  [b h w c]
        # b h w c --> b c h w
        f1 = f1.permute(0, 3, 1, 2)  # f1 [2, 96, 64, 64]
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)

        f1 = self.eaa_1(f1)
        f2 = self.eaa_2(f2)
        f3 = self.eaa_3(f3)
        f4 = self.eaa_4(f4)

        f1 = self.da_1(f1)
        f2 = self.da_2(f2)
        f3 = self.da_3(f3)
        f4 = self.da_4(f4)

        # Use channel attention bridge
        ca_bridge_out = self.channel_attention_bridge(f1, f2, f3, f4)
        sa_bridge_out = self.spatial_attention_bridge(f1, f2, f3, f4)

        # Concatenate the bridge outputs and apply a convolution to unify the number of channels
        combined_bridge_out = torch.cat([ca_bridge_out, sa_bridge_out], dim=1)
        f_bridge_combined = self.bridge_conv(combined_bridge_out)

        # Upsampling and combining features
        f41 = self.sdi_4([f1, f2, f3, f4], f_bridge_combined)
        f41_up = self.pixel_shuffle_upsample4(f41)
        f31 = self.sdi_3([f1, f2, f3, f4], f_bridge_combined)
        f31_up = self.pixel_shuffle_upsample3(f31)
        f21 = self.sdi_2([f1, f2, f3, f4], f_bridge_combined)
        f21_up = self.pixel_shuffle_upsample2(f21)
        f11 = self.sdi_1([f1, f2, f3, f4], f_bridge_combined)
        f11_up = self.pixel_shuffle_upsample1(f11)

        # Ensure matching spatial dimensions before addition
        y = self.deconv2(f41_up)
        f31_up_resized = F.interpolate(f31_up, size=y.shape[2:], mode='bilinear', align_corners=False)
        y = y + f31_up_resized

        seg_outs.append(self.seg_outs[0](y))

        y = self.deconv3(y)
        f21_up_resized = F.interpolate(f21_up, size=y.shape[2:], mode='bilinear', align_corners=False)
        y = y + f21_up_resized

        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv4(y)
        f11_up_resized = F.interpolate(f11_up, size=y.shape[2:], mode='bilinear', align_corners=False)
        y = y + f11_up_resized

        seg_outs.append(self.seg_outs[2](y))

        # Final output handling
        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, size=self.output_size, mode='bilinear', align_corners=False)

        if self.deep_supervision:
            temp = seg_outs[::-1]
            out_0 = temp[0]
            out_1 = temp[1]
            out_1 = self.deconv6(out_1)
            out_1 = F.interpolate(out_1, size=self.output_size, mode='bilinear', align_corners=False)
            return torch.sigmoid(out_0 + out_1)
        else:
            return torch.sigmoid(seg_outs[-1]) if self.num_classes == 1 else seg_outs[-1]

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")


pretrained_path = '/content/drive/MyDrive/465 Project/pre_trained_weights_mamba/vmamba_small_e238_ema.pth'
model = VMUNetV2(
            num_classes=1,
            input_channels=3,
            depths= [2,2,9,2],
            depths_decoder=[2,2,2,1],
            drop_path_rate=0.2,
            load_ckpt_path=pretrained_path,
            deep_supervision = True,
            output_size=(256, 256)  # Adjust this to (256, 256) or (512, 512) as needed
        ).cuda()

model.load_from()
model.eval()
with torch.no_grad():
  x = torch.randn(1, 3, 256, 256).cuda()
  predict = model(x)
  print(predict.shape)
