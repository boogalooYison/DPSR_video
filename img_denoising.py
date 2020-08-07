import os
import time
import cv2
import torch
import denoising.dpsr.dpsr as dpsr

from denoising.dpsr.utils import utils_image as util
from denoising.dpsr.models.network_srresnet import SRResNet
from config.config import cfg


def main():

    # instance for the dpsr model
    dpsr_model = dpsr.DPSR(
                cfg.PATH.WEIGHT, 
                noise_level=cfg.IMG.NOISE, 
                n_channels=cfg.IMG.CHANNEL, 
                upscale=cfg.MODEL.UPSCALE,
                act_mode=cfg.MODEL.ACT,
                upsample_mode=cfg.MODEL.UPSAMPLE,
                method=cfg.MODEL.EXCUTE)

    # walk through the img dir
    for file in os.listdir(cfg.PATH.IMAGE):
        # -- split the imgname and ext
        img_name, ext = os.path.splitext(file)

        # -- read the img
        img = util.imread_uint(os.path.join(cfg.PATH.IMAGE, file), n_channels=cfg.IMG.CHANNEL)

        start_time = time.time()

        # -- denoise the image
        img_denoise = dpsr_model.denoising(img, kernel=img_name)

        denoising_time = time.time() - start_time

        print('Finish processing \'' + file + '\'')
        print('Time used: {}'.format(denoising_time))

        # -- save the image
        util.imsave(img_denoise, os.path.join(cfg.PATH.SAVEI, img_name+ext))

if __name__ == '__main__':
    main()