import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# np.set_printoptions(threshold=np.nan)
from math import exp
from typing import Callable, Any, Optional, Tuple, List
from .blocks import *



###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], size=256):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, load_size=size)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, load_size=size)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, load_size=size)
    elif netG == 'inception':
        net = Inception(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, load_size=size)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, load_size=size)
    elif netG == 'HRUnet':
        net = HRUnet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, load_size=size)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'global':     # classify if each pixel is real or fake
        net = GlobalDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class f1_loss(nn.Module):
            # assert y_true.ndim == 1
            # assert y_pred.ndim == 1 or y_pred.ndim == 2
            
            # if y_pred.ndim == 2:
            #     y_pred = y_pred.argmax(dim=1)

     def __init__(self):   

        super(f1_loss, self).__init__()     
                
    
     def __call__(self, y_pred, y_true):
        y_true=(y_true>0.1).float() * 1
        y_pred=(y_pred>0.1).float() * 1
        # print(y_pred)

        add=y_true+y_pred
        sub=y_true-y_pred

        tp=(add == 2.).sum()
        tn=(add == 0.).sum()
        fp=(sub == -1.).sum()
        fn=(sub == 1.).sum()
        # sth,tp=torch.unique(add, return_counts=True)
        # print(sth,tp(3))

        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        
        epsilon = 1e-7
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        # print(tp,tn,fp,fn)
        
        f1 = 2* (precision*recall) / (precision + recall + epsilon)
        return 1/f1


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, load_size=256, padding_type='reflect',  expansion_factor = 2):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc * expansion_factor , kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

        self.S_enhancer = nn.Sequential(
                              nn.Linear(load_size,512),
                              nn.Linear(512,load_size)
                )

        self.final_CNN =  nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(output_nc * expansion_factor,output_nc, kernel_size=3),
                            nn.Tanh()
            )


    def forward(self, input, gt):

        x_feat  = self.model(input)
        gt_feat = self.model(gt)

        try : 

            U, S, Vh = torch.linalg.svd(x_feat, full_matrices=False)
        except :

            U, S, Vh = torch.linalg.svd(x_feat + + 1e-4*x_feat.mean()*torch.rand_like(x_feat), full_matrices=False)

        S_real = torch.linalg.svdvals(gt_feat)

        S_e =  self.S_enhancer(S)

        y = U @ torch.diag_embed(S_e) @ Vh 

        out = self.final_CNN(y) 

        return out, S, S_e, S_real, x_feat, gt_feat, y


class SVDBlock(nn.Module):

    def __init__(self):

        super(SVDBlock, self).__init__()


        self.S_enhancer = nn.Sequential(
                              nn.Linear(262,512),
                              nn.Linear(512,262)
                )


    def forward(self, input):
        U, S, Vh = torch.linalg.svd(input, full_matrices=False)

        S_e =  self.S_enhancer(S)

        y = U @ torch.diag_embed(S_e) @ Vh

        return y 


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, load_size=256, expansion_factor = 2):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc * expansion_factor, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.S_enhancer = nn.Sequential(
                              nn.Linear(load_size,512),
                              nn.Linear(512,load_size)
                )

        self.final_CNN =  nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(output_nc * expansion_factor,output_nc, kernel_size=3),
                            nn.Tanh()
            )
        
    def forward(self, input, gt):
        """Standard forward"""
        x_feat  = self.model(input)
        gt_feat = self.model(gt)

        try : 

            U, S, Vh = torch.linalg.svd(x_feat, full_matrices=False)
        except :

            U, S, Vh = torch.linalg.svd(x_feat + + 1e-4*x_feat.mean()*torch.rand_like(x_feat), full_matrices=False)

        S_real = torch.linalg.svdvals(gt_feat)

        S_e =  self.S_enhancer(S)

        y = U @ torch.diag_embed(S_e) @ Vh

        out =  self.final_CNN(y)

        return out, S, S_e, S_real, x_feat, gt_feat, y



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



class HRUnet(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, load_size=256, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        
        super(HRUnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(16)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(32)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # convs for global features
        # input 32x32x128 output 16x16x128
        self.conv51 = nn.Conv2d(128, 128, 5, stride=2, padding=2)

        # input 16x16x128 output 8x8x128
        self.conv52 = nn.Conv2d(128, 128, 5, stride=2, padding=2)

        # input 8x8x128 output 1x1x128
        self.conv53 = nn.Conv2d(128, 128, 8, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(1, 1),
            nn.SELU(inplace=True),
            nn.Linear(1, 1),
        )

        # input 32x32x128 output 32x32x128
        # the global features should be concatenated to the feature map after this layer
        # the output after concat would be 32x32x256
        self.conv6 = nn.Conv2d(128, 196, 5, stride=1, padding=2)

        # input 32x32x256 output 32x32x128
        self.conv7 = nn.Conv2d(196, 128, 5, stride=1, padding=2)

        # deconvolutional layers
        # input 32x32x128 output 64x64x128
        self.dconv1 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        )

        # input 64x64x256 ouput 128x128x128
        self.dconv2 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        )

        # input 128x128x192 output 256x256x64
        self.dconv3 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(192),
            nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1)
        )

        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(96),
            nn.ConvTranspose2d(96, 32, 4, stride=2, padding=1)
        )

        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 16, 5, stride=1, padding=2)
        )

        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, 5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x0 = self.conv1(x)

        # input 512x512x16 to output 256x256x32
        x1 = self.conv2(x0)

        # input 256x256x32 to output 128x128x64
        x2 = self.conv3(x1)

        # input 128x128x64 to output 64x64x128
        x3 = self.conv4(x2)

        # input 64x64x128 to output 32x32x128
        x4 = self.conv5(x3)

        # # convolutions for global features
        # # input 32x32x128 to output 16x16x128
        # x51 = self.conv51(x4)

        # # # input 16x16x128 to output 8x8x128
        # # x52 = self.conv52(x51)

        # # # input 8x8x128 to output 1x1x128
        # # x53 = self.conv53(x52)
        # # # x53 = self.fc(x53)
        # # x53_temp = torch.cat([x53] * 32, dim=2)
        # # x53_temp = torch.cat([x53_temp] * 32, dim=3)

        # input 32x32x128 to output 32x32x196
        x5 = self.conv6(x4)

        # input 32x32x196 to output 32x32x128
        x5 = self.conv7(x5)

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(x5)

        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(torch.cat([xd, x3], dim=1))

        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(torch.cat([xd, x2], dim=1))

        # input 256x256x96 to output 512x512x32
        xd = self.dconv4(torch.cat([xd, x1], dim=1))

        # input 512x512x48 to output 512x512x16
        xd = self.conv8(torch.cat([xd, x0], dim=1))

        # input 512x512x16 to output 512x512x3
        xd = self.conv9(xd)

        # Residuals
        xd = xd
        return xd



class Inception(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, load_size=256):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Inception, self).__init__()
        # construct unet structure
        conv_depths=conv_depths=(64, 128, 256, 512, 1024)
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'
        in_channels=input_nc
        out_channels=output_nc * expansion_factor

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])
        # encoder_layers2 = []
        # encoder_layers2.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        # encoder_layers2.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
        #                        for i in range(len(conv_depths)-2)])
        self.inception_block=DeepWInet(conv_depths[len(conv_depths)-2],32)
        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(conv_depths[1], conv_depths[0], out_channels))


        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        # self.encoder_layers2=self.encoder_layers
        # self.encoder_layers2 = nn.Sequential(*encoder_layers2)
        self.center = nn.Sequential(Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2]),
                                    DeepWInet(conv_depths[len(conv_depths)-2],32))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        self.model = nn.Sequential(
            self.encoder_layers,
            self.center,
            self.decoder_layers
            )


        self.S_enhancer = nn.Sequential(
                              nn.Linear(load_size,512),
                              nn.Linear(512,load_size)
                )

        self.final_CNN =  nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(out_channels,output_nc, kernel_size=3),
                            nn.Tanh()
            )


    def basic_forward(self, input, return_all=False):
        
        img=input
        img=[img]
        

        for enc_layer in self.encoder_layers:
            img.append(enc_layer(img[-1]))

        x_enc=img #+att


        x_dec = [self.center(x_enc[-1])]
        
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec

    def forward(self, input, gt, return_all=False):

        x = self.basic_forward(input) #6, 12, 256, 256
        g = self.basic_forward(gt) #bs, 12, 256, 256

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        S_real = torch.linalg.svdvals(g)

        S_e =  self.S_enhancer(S)

        y = U @ torch.diag_embed(S_e) @ Vh

        out = self.final_CNN(y) #6, 12, 256, 256 => 6, 3, 256, 256

        return out , S_e, S_real



def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)




class DeepWInet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DeepWInet, self).__init__()
        #BatchNorm=nn.BatchNorm1d
       
        if conv_block is None:
            conv_block = nn.Conv2d

           
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=7,stride=1,padding='same')


       
        self.branch5x5_1 = conv_block(in_channels, 64, kernel_size=5,stride=1,padding='same')
        self.branch5x5_2 = conv_block(64, 64, kernel_size=5,stride=1,padding='same')
       
       
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=3,stride=1,padding='same')
        self.branch3x3dbl_2 = conv_block(64, 64, kernel_size=3,stride=1,padding='same')
        self.branch3x3dbl_3 = conv_block(64, 64, kernel_size=3,stride=1,padding='same')
        self.branch3x3dbl_4 = conv_block(64, 64, kernel_size=3,stride=1,padding='same')
       
        self.branchdblSINGLE = conv_block(in_channels, 64, kernel_size=1,stride=1,padding='same')
 
        self.branch1x1_v2 = conv_block(256, 128, kernel_size=7,stride=1,padding='same')

       
        self.branch5x5_1_v2 = conv_block(256, 128, kernel_size=5,stride=1,padding='same')
        self.branch5x5_2_v2 = conv_block(128, 128, kernel_size=5,stride=1,padding='same')
       

        self.branch3x3dbl_1_v2 = conv_block(256, 128, kernel_size=3,stride=1,padding='same')
        self.branch3x3dbl_2_v2 = conv_block(128, 128, kernel_size=3,stride=1,padding='same')
        self.branch3x3dbl_3_v2 = conv_block(128, 128, kernel_size=3,stride=1,padding='same')
        self.branch3x3dbl_4_v2 = conv_block(128, 128, kernel_size=3,stride=1,padding='same')
       
       
        self.branchdblSINGLE_v2 = conv_block(256, 128, kernel_size=1,stride=1,padding='same')
       
       
        #self.branch_pool = conv_block(in_channels, 2*pool_features, kernel_size=1)
       
       

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_4(branch3x3dbl)
       
        branchdblSINGLE = self.branchdblSINGLE(x)
       
        Maxpool=nn.MaxPool2d(3, stride=1)

       # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        #branch_pool = self.branch_pool(branch_pool)

        #outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = [branchdblSINGLE, branch1x1, branch5x5, branch3x3dbl]
        y = torch.cat(outputs, 1)
        y = Maxpool(y)
        y = self.forward2(y)
       
        return y



    def forward2(self, x):
       
        branch1x1_v2 = self.branch1x1_v2(x)

        branch5x5_v2 = self.branch5x5_1_v2(x)
        branch5x5_v2 = self.branch5x5_2_v2(branch5x5_v2)

        branch3x3dbl_v2 = self.branch3x3dbl_1_v2(x)
        branch3x3dbl_v2 = self.branch3x3dbl_2_v2(branch3x3dbl_v2)
        branch3x3dbl_v2 = self.branch3x3dbl_3_v2(branch3x3dbl_v2)
        branch3x3dbl_v2 = self.branch3x3dbl_4_v2(branch3x3dbl_v2)
       
        branchdblSINGLE_v2 = self.branchdblSINGLE_v2(x)

       # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        #branch_pool = self.branch_pool(branch_pool)

        #outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputsF = [branchdblSINGLE_v2, branch1x1_v2, branch5x5_v2, branch3x3dbl_v2]
        return outputsF

    def forward(self, x):
        outputs = self._forward(x)
        #return torch.cat(outputs, 1)
        return torch.cat(outputs, 1)






class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



class GlobalDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(GlobalDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

       #  Convolutional layers
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(16)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(32)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(64)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 32x32x128  output 16x16x128
        # the output of this layer we need layers for global features
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 16x16x128  output 1x1x128
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 1, 16),
            nn.LeakyReLU(inplace=True)
        )

        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1(x)

        # input 512x512x16 to output 256x256x32
        x = self.conv2(x)

        # input 256x256x32 to output 128x128x64
        x = self.conv3(x)

        # input 128x128x64 to output 64x64x128
        x = self.conv4(x)

        # input 64x64x128 to output 32x32x128
        x = self.conv5(x)

        # input 32x32x128 to output 16x16x128
        x = self.conv6(x)

        # input 16x16x128 to output 1x1x1
        x = self.conv7(x)
        # print(x.shape)

        x = self.fc(x)

        return x


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
