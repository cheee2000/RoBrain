import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from config import cfg


class BrainDecoder(nn.Module):
    def __init__(self, dim_fmri, dim_fea, dim_hidden=8192):
        super(BrainDecoder, self).__init__()
        self.layer1 = nn.Linear(dim_fmri, dim_hidden)
        self.relu1 = nn.ReLU()
        # self.layer2 = nn.Linear(dim_hidden, dim_hidden)
        # self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(dim_hidden, dim_fea)

    def forward(self, x):
        h = self.relu1(self.layer1(x))
        # h = self.relu2(self.layer2(h))
        out = self.layer3(h)
        return out


####################################################################
# ------------------------- Generator -------------------------------
####################################################################
class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        ngf = cfg.GAN.GF_DIM
        self.deconv1 = nn.ConvTranspose2d(cfg.BRAIN.DIMENSION, ngf * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(ngf * 8)
        self.relu1 = nn.ReLU()
        # (ngf*8) x 4 x 4
        self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(ngf * 4)
        self.relu2 = nn.ReLU()
        # (ngf*4) x 8 x 8
        self.deconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(ngf * 2)
        self.relu3 = nn.ReLU()
        # (ngf*2) x 16 x 16
        self.deconv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(ngf)
        self.relu4 = nn.ReLU()
        # (ngf) x 32 x 32
        self.deconv5 = nn.ConvTranspose2d(ngf, 3, 4, 2, 1)
        self.tanh = nn.Tanh()
        #  3 x 64 x 64

    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    # forward method
    def forward(self, cond_vec):
        x = cond_vec.unsqueeze(2).unsqueeze(3)
        x = self.relu1(self.deconv1_bn(self.deconv1(x)))
        x = self.relu2(self.deconv2_bn(self.deconv2(x)))
        x = self.relu3(self.deconv3_bn(self.deconv3(x)))
        x = self.relu4(self.deconv4_bn(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x


####################################################################
# ------------------------- Discriminator --------------------------
####################################################################
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = cfg.GAN.DF_DIM
        self.main = nn.Sequential(
            #  3 x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # (ndf*8) x 4 x 4
        )
        self.enc_feat = nn.Conv2d(ndf * 8, ndf, 4, 1, 0)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    def forward(self, image, enc_feat=False):
        img_code = self.main(image)
        out_cond = self.logits(img_code).view(-1)
        if enc_feat:
            feats = self.enc_feat(img_code)
            feats = feats.view(feats.shape[0], -1)
            return out_cond, feats
        return out_cond


####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        model = models.inception_v3(pretrained=True)
        # url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        # print('Load pretrained model from ', url)

        self.define_module(model)

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

    def forward(self, x):
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        # x = self.Mixed_7a(x)
        # # 8 x 8 x 1280
        # x = self.Mixed_7b(x)
        # # 8 x 8 x 2048
        # x = self.Mixed_7c(x)
        # # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=8)
        # # 1 x 1 x 2048
        # # x = F.dropout(x, training=self.training)
        # # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # # 2048

        features = F.avg_pool2d(features, kernel_size=17)
        features = features.view(features.shape[0], -1)
        return features
