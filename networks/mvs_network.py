import torch
import torch.nn as nn
import layers

class FeatExtractorNet(nn.Module):
    def __init__(self, base_channels, num_stages=3):
        super(FeatExtractorNet, self).__init__()

        self.base_channels = base_channels
        self.num_stages = num_stages

        self.conv0 = nn.Sequential(
            layers.Conv2dLayer(3, base_channels, 3, 1, padding=1),
            layers.Conv2dLayer(base_channels, base_channels, 3, 1, padding=1)
        )

        self.conv1 = nn.Sequential(
            layers.Conv2dLayer(base_channels, base_channels * 2, 5, stride=2, padding=2),
            layers.Conv2dLayer(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            layers.Conv2dLayer(base_channels * 2, base_channels * 2, 3, 1, padding=1)
        )

        self.conv2 = nn.Sequential(
            layers.Conv2dLayer(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            layers.Conv2dLayer(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            layers.Conv2dLayer(base_channels * 4, base_channels * 4, 3, 1, padding=1)
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if num_stages == 2:
            self.deconv1 = layers.Deconv2dLayerPlus(base_channels * 4, base_channels * 2, 3)

            self.out2_pre = layers.Conv2dLayer(base_channels * 4, base_channels * 2, 3, padding=1)
            self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
            self.out_channels.append(2 * base_channels)

        elif num_stages == 3:
            self.deconv1 = layers.Deconv2dLayerPlus(base_channels * 4, base_channels * 2, 3)
            self.out2_pre = layers.Conv2dLayer(base_channels * 4, base_channels * 2, 3, padding=1)
            self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)

            self.deconv2 = layers.Deconv2dLayerPlus(base_channels * 2, base_channels, 3)
            self.out3_pre = layers.Conv2dLayer(base_channels *2, base_channels, 3, padding=1)
            self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
            self.out_channels.append(2 * base_channels)
            self.out_channels.append(base_channels)

    def forward(self, x):
        outputs = {}

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        out_bottleneck = conv2

        out = self.out1(out_bottleneck)
        outputs['stage1'] = out

        if self.num_stages == 2:
            deconv1 = self.deconv1(conv1, out_bottleneck)
            out2_pre = self.out2_pre(deconv1)
            out = self.out2(out2_pre)
            outputs['stage2'] = out

        elif self.num_stages == 3:
            deconv1 = self.deconv1(conv1, out_bottleneck)
            out2_pre = self.out2_pre(deconv1)
            out = self.out2(out2_pre)
            outputs['stage2'] = out

            deconv2 = self.deconv2(conv0, out2_pre)
            out3_pre = self.out3_pre(deconv2)
            out = self.out3(out3_pre)
            outputs['stage3'] = out

        return outputs


class CostRegularizerNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegularizerNet, self).__init__()

        self.conv0 = layers.Conv3dLayer(in_channels, base_channels, padding=1)

        self.conv1 = layers.Conv3dLayer(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = layers.Conv3dLayer(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = layers.Conv3dLayer(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = layers.Conv3dLayer(base_channels * 4, base_channels * 4, padding =1)

        self.conv5 = layers.Conv3dLayer(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = layers.Conv3dLayer(base_channels * 8, base_channels * 8, padding=1)

        self.deconv7 = layers.Deconv3dLayer(base_channels * 8, base_channels * 4, stride=2, padding=1,
                                            output_padding=1)

        self.deconv8 = layers.Deconv3dLayer(base_channels * 4, base_channels * 2, stride=2, padding=1,
                                            output_padding=1)

        self.deconv8 = layers.Deconv3dLayer(base_channels * 4, base_channels * 2, stride=2, padding=1,
                                            output_padding=1)

        self.deconv9 = layers.Deconv3dLayer(base_channels * 2, base_channels * 1, stride=2, padding=1,
                                            output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)

        return x


if __name__ == '__main__':
    torch.manual_seed(0)

    @torch.no_grad()
    # For testing purposes
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            m.weight.fill_(1.0)

    import numpy as np
    import sys
    sys.path.append('/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz')
    sys.path.append('/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/UCSNet-master/networks')
    import Dataset.preprocessing as prep
    import submodules
    some_input = np.array([[[[14, 33, 82, 17],
                              [10, 36, 52, 76],
                              [89, 20, 11, 3]],
                             [[12, 31, 45, 63],
                              [96, 22, 14, 7],
                              [10, 30, 21, 13]],
                             [[4, 15, 36, 27],
                               [11, 32, 50, 16],
                               [90, 3, 10, 13]]],

                           [[[4, 3, 34, 43],
                             [43, 23, 52, 76],
                             [89, 20, 234, 234]],
                            [[12, 21, 23, 63],
                             [16, 22, 14, 7],
                             [10, 30, 21, 13]],
                            [[42, 15, 36, 27],
                             [11, 32, 50, 16],
                             [90, 3, 10, 13]]]
                           ], dtype=np.float32)/96.
    some_input = torch.from_numpy(some_input)
    image1 = prep.read_exr_image("/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_dataset/03325088/ee9f4810517891d6c36fb70296e45483/view_4/right_reciprocal/r_reciprocal0001.exr")
    image2 = prep.read_exr_image("/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_dataset/03325088/ee9f4810517891d6c36fb70296e45483/view_2/right_reciprocal/r_reciprocal0001.exr")
    realimage = np.stack((image1, image2))
    image = torch.from_numpy(realimage.transpose([0, 3, 1, 2]))

    someExtractor2 = submodules.FeatExtNet(8, 3)
    someExtractor2.apply(init_weights)
    output = someExtractor2(image)
    print('Size of the stages of orinal featExt:',output['stage1'].size(), output['stage2'].size(), output['stage3'].size())
    volume_input = output['stage1'].unsqueeze(2).repeat(1, 1, 8, 1, 1)
    print("Original: volume input shape: ", volume_input.size())
    costRegularizer2 = submodules.CostRegNet(32, 8)
    volume = costRegularizer2(volume_input)
    print("Original: volume output shape: ", volume.shape)

    someExtractor = FeatExtractorNet(8, 3)
    someExtractor.apply(init_weights)
    output2 = someExtractor(image)
    volume_input2 = output2['stage1'].unsqueeze(2).repeat(1, 1, 8, 1, 1)
    print(output2['stage1'].size(), output2['stage2'].size(), output2['stage3'].size())
    print('Custom: volume input shape: ', volume_input.size())
    print('Comparing the results of original featExt and custom FeatExt:',  torch.equal(output['stage1'], output2['stage1']), torch.equal(output['stage2'], output2['stage2']),
          torch.equal(output['stage3'], output2['stage3']))
    costRegularizer = CostRegularizerNet(32, 8)
    volume2 = costRegularizer2(volume_input2)
    print('Custom: volume output shape: ', volume2.shape)
    print('both original and custom are equal' if torch.equal(volume, volume2)
          else 'They are not equal')