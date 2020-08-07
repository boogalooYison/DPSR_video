import os.path
import time
import numpy as np
from scipy.io import loadmat
import torch
import cv2


from config.config import cfg
from denoising.dpsr.models.network_srresnet import SRResNet
from denoising.dpsr.utils import utils_deblur
from denoising.dpsr.utils import utils_logger
from denoising.dpsr.utils import utils_image as util




# -- check the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class DPSR:
    def __init__(self, weight, noise_level=8./255., n_channels=3, upscale=4, act_mode='R', upsample_mode='pixelshuffle', method='DPSRGAN'):

        load_model_time = time.time()

        # -- load the model and noise, channels, upscale, act_mode, upsample_mode, method    
        self.noise_level = noise_level
        self.n_channels = n_channels
        self.upscale = upscale
        self.act_mode = act_mode
        self.upsample_mode = upsample_mode
        self.method = method

        if method=='DPSRGAN':
            assert upscale == 4
        else:
            assert upscale in [2, 3, 4]

        self.model = self.load_model(weight)

        loading_time = time.time() - load_model_time

        print('Load model in ', str(loading_time), 's')


    def load_model(self, weight):
        print(weight)

        # -- instance for the model
        model = SRResNet(in_nc=self.n_channels+1, out_nc=self.n_channels, nc=96, nb=16, upscale=self.upscale, act_mode=self.act_mode, upsample_mode=self.upsample_mode)
        
        # -- load the model state dict
        model.load_state_dict(torch.load(weight), strict=True)
        
        # -- model eval method
        model.eval()

        # -- cancle the gradient
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        # -- adjust the model device
        model = model.to(device)
        print('Finish load model...')

        return model

    def denoising(self, frame, kernel=None, use_iter=False):


        # -- resize the img
        frame = util.img_resize(frame)

        # -- convert the type of the img
        h, w = frame.shape[:2]
        frame = util.uint2single(frame)

        if not use_iter:
            
            # -- convert the type of the img
            img_L = util.single2tensor4(frame)

            # -- do super-resolution
            noise_level_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)), dtype=torch.float).mul_(self.noise_level)
            img_L = torch.cat((img_L, noise_level_map), dim=1)
            img_L = img_L.to(device)
            img_E = self.model(img_L)

            img_E = util.tensor2single(img_E)
            img_E = util.single2uint(img_E[:h*self.upscale, :w*self.upscale])

        else:
            # -- load the iter times
            iter_num = cfg.MODEL.ITERS

            # -- load the blur kernel
            if os.path.exists(os.path.join(cfg.PATH.KERNEL, kernel+'_kernel.mat')):
                k = loadmat(os.path.join(cfg.PATH.KERNEL, kernel+'.mat'))['kernel']
                k = k.astype(np.float64)
                k /= k.sum()
            elif os.path.exists(os.path.join(cfg.PATH.KERNEL, kernel+'_kernel.png')):
                k = cv2.imread(os.path.join(cfg.PATH.KERNEL, kernel+'_kernel.png'), 0)
                k = np.float64(k)
                k /= k.sum()
            else:
                k = utils_deblur.fspecial('gaussian', 3, 0.5)
                iter_num = 5

            # -- handle the boundary
            img = utils_deblur.wrap_boundary_liu(frame, utils_deblur.opt_fft_size([frame.shape[0]+k.shape[0]+1, frame.shape[1]+k.shape[1]+1]))

            img_E = self.iteration_dpsr(img, k, iter_times=iter_num)
            img_E = util.single2uint(img_E[:h*self.upscale, :w*self.upscale])

        return img_E

    def iteration_dpsr(self, img, k, iter_times=15):
        
        # -- get upperleft, denominator
        upperleft, denominator = utils_deblur.get_uperleft_denominator(img, k)

        # get rhos and sigmas
        rhos, sigmas = utils_deblur.get_rho_sigma(sigma=max(0.255/255.0, self.noise_level), iter_num=iter_times)

        # -- convert the params type
        z = img
        rhos = np.float32(rhos)
        sigmas = np.float32(sigmas)

        # -- main iteration
        for i in range(iter_times):

            # --------------------------------
            # step 1, Eq. (9) // FFT
            # --------------------------------
            rho = rhos[i]
            if i != 0:
                z = util.imresize_np(z, 1/self.upscale, True)

            z = np.real(np.fft.ifft2((upperleft + rho*np.fft.fft2(z, axes=(0, 1)))/(denominator + rho), axes=(0, 1)))

            # --------------------------------
            # step 2, Eq. (12) // super-resolver
            # --------------------------------
            sigma = torch.from_numpy(np.array(sigmas[i]))
            img_L = util.single2tensor4(z)

            noise_level_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)), dtype=torch.float).mul_(sigma)
            img_L = torch.cat((img_L, noise_level_map), dim=1)
            img_L = img_L.to(device)

            # with torch.no_grad():
            z = self.model(img_L)
            z = util.tensor2single(z)
        
        return z
