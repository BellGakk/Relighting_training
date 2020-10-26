import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

def get_shading(Normal, Lighting):

    #c1 for all the channel-elements in the batch
    c1 = 0.8862269254527579
    # c2 for all the channel-elements in the batch
    c2 = 1.0233267079464883
    c3 = 0.24770795610037571
    c4 = 0.8580855308097834
    c5 = 0.4290427654048917

    # classic representation for three channels in the batches
    nx = Normal[:, 0, :, :]
    ny = Normal[:, 1, :, :]
    nz = Normal[:, 2, :, :]

    b, c, h, w = Normal.shape

    Y1 = c1 * torch.ones(b, h, w)
    Y2 = c2 * nz
    Y3 = c2 * nx
    Y4 = c2 * ny
    Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
    Y6 = c4 * nx * nz
    Y7 = c4 * ny * nz
    Y8 = c5 * (nx * nx - ny * ny)
    Y9 = c4 * nx * ny

    # split the L at dim 1, each block with 9 small pieces.
    Lighting = Lighting.type(torch.float)
    spheric_harmonic = torch.split(Lighting, 9, dim=1)

    assert (c == len(spheric_harmonic))
    shading = torch.zeros(b, c, h, w)

    #if torch.cuda.is_available():
    #    Y1 = Y1.cuda()
    #    shading = shading.cuda()

    for j in range(c):
        l = spheric_harmonic[j]
        # scale to 'h * w' dim
        l = l.repeat(1, h * w).view(b, h, w, 9)
        # convert l into 'batch_size', 'index_sh', 'h', 'w'
        l = l.permute([0, 3, 1, 2])
        shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                              Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                              Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

    return shading

class sfsNetShading(nn.Module):
    def __init__(self):
        super(sfsNetShading, self).__init__()

    def forward(self, Normal, Lighting):
        #following values are computed from equation
        #from SFSNet

        c1 = 0.8862269254527579
        # c2 for all the channel-elements in the batch
        c2 = 1.0233267079464883
        c3 = 0.24770795610037571
        c4 = 0.8580855308097834
        c5 = 0.4290427654048917

        # classic representation for three channels in the batches
        nx = Normal[:, 0, :, :]
        ny = Normal[:, 1, :, :]
        nz = Normal[:, 2, :, :]

        b, c, h, w = Normal.shape

        Y1 = c1 * torch.ones(b, h, w)
        Y2 = c2 * nz
        Y3 = c2 * nx
        Y4 = c2 * ny
        Y5 = c3 * (2 * nz * nz - nx * nx - ny * ny)
        Y6 = c4 * nx * nz
        Y7 = c4 * ny * nz
        Y8 = c5 * (nx * nx - ny * ny)
        Y9 = c4 * nx * ny

        # split the L at dim 1, each block with 9 small pieces.
        Lighting = Lighting.type(torch.float)
        spheric_harmonic = torch.split(Lighting, 9, dim=1)

        assert (c == len(spheric_harmonic))
        shading = torch.zeros(b, c, h, w)

        if torch.cuda.is_available():
            Y1 = Y1.cuda()
            shading = shading.cuda()

        for j in range(c):
            l = spheric_harmonic[j]
            # scale to 'h * w' dim
            l = l.repeat(1, h * w).view(b, h, w, 9)
            # convert l into 'batch_size', 'index_sh', 'h', 'w'
            l = l.permute([0, 3, 1, 2])
            shading[:, j, :, :] = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
                                  Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
                                  Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]

        return shading

def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class baseFeatureExtraction(nn.Module):
    '''
    Base Feature Extraction
    '''

    def __init__(self):
        super(baseFeatureExtraction, self).__init__()
        self.conv1 = get_conv(3, 64, kernel_size=7, padding=3)
        self.conv2 = get_conv(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.res = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, 3, stride),
            nn.BatchNorm2d(in_planes),
            nn.ReLu(inplace=True),
            nn.Conv2d(in_planes, out_planes, 3, stride)
        )

    def forward(self, x):
        residual = x
        out = self.res(x)
        out += residual

#help for normal of the object
class NormalResidualBlock(nn.Module):
    '''
    Net to general normal from features
    '''

    def __init__(self):
        super(NormalResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = nn.ReLU(self.bn(out))
        return out

#help for albedo for the object
class AlbedoResidualBlock(nn.Module):
    '''
    Net to general albedo from features
    '''

    def __init__(self):
        super(AlbedoResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = nn.ReLU(self.bn(out))
        return out

class NormalGenerationNet(nn.Module):
    '''
    Generating normals from images, first we need to unsample the image, then
    pass the processed image to different post convolutional layers
    '''
    def __init__(self):
        super(NormalGenerationNet, self).__init__()
        self.unsample = nn.Unsample(scale_factor=2, mode='bilinear')
        self.conv1 = get_conv(128, 128, kernel_size=1)
        self.conv2 = get_conv(128, 64, kernel_size=3, padding=1)
        self.conv3 = get_conv(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.unsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class AlbedoGenerationNet(nn.Module):

    def __init__(self):
        super(AlbedoGenerationNet, self).__init__()
        self.unsample = nn.Unsample(scale_factor=2, mode='bilinear')
        self.conv1 = get_conv(128, 128, kernel_size=1)
        self.conv2 = get_conv(128, 64, kernel_size=3, padding=1)
        self.conv3 = get_conv(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.unsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class LightEstimator(nn.Module):
    '''
    Estimate lighting from normal, albedo and conv features
    '''

    def __init__(self):
        super(LightEstimator, self).__init__()
        self.conv1 = get_conv(384, 128, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(64, stride=1, padding=0)
        self.fc = nn.Linear(128, 27)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = out.view(-1, 128)
        out = self.fc(out)
        return out

def reconstruct_image(shading, albedo):
    return shading * albedo

class SfsNetPipeline(nn.Module):
    '''
    SfsNet Pipeline
    '''

    def __init__(self):
        super(SfsNetPipeline, self).__init__()

        self.conv_model = baseFeatureExtraction()
        self.normal_residual_model = NormalResidualBlock()
        self.normal_gen_model = NormalGenerationNet()
        self.albedo_residual_model = AlbedoResidualBlock()
        self.Albedo_gen_model = AlbedoGenerationNet()
        self.light_estimator_model = LightEstimator()

    def get_face(self, spheric_harmonic, normal, albedo):
        shading = get_shading(normal, spheric_harmonic)
        recon = reconstruct_image(shading, albedo)
        return recon

    def forward(self, face):
        out_features = self.conv_model(face)

        #2. Pass Conv features through Normal Residual
        out_normal_features = self.normal_residual_model(out_features)
        #3. Pass Conv features through Albedo Residual
        out_albedo_features = self.albedo_residual_model(out_features)

        #3 a.Generate Normal
        predicted_normal = self.normal_gen_model(out_normal_features)
        #3 b.Generate ALbedo
        predicted_albedo = self.albedo_gen_model(out_albedo_features)
        #3 c.Estimate lighting
        all_features = torch.cat((out_features, out_normal_features, out_albedo_features), dim=1)
        #predict SH
        predicted_sh = self.light_estimator_model(all_features)

        # 4. Generate shading
        out_shading = get_shading(predicted_normal, predicted_sh)

        # 5. Reconstruction of image
        out_recon = reconstruct_image(out_shading, predicted_albedo)

        return predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon

    def fix_weights(self):

        dfs_freeze(self.conv_model)
        dfs_freeze(self.normal_residual_model)
        dfs_freeze(self.normal_gen_model)
        dfs_freeze(self.albedo_residual_model)
        dfs_freeze(self.light_estimator_model)

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

#Base methods for creating convnet
def get_skipnet_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding = padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def get_skipnet_deconv(in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding = padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

