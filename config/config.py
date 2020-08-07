# from easydict import EasyDict as edict
cfg                       = dict()

# img settings
cfg['IMG']                   = dict()
cfg['IMG']['NOISE']             = 8./255.
cfg['IMG']['CHANNEL']           = 3

# model settings
cfg['MODEL']                 = dict()
cfg['MODEL']['UPSCALE']         = 2
cfg['MODEL']['UPSAMPLE']        = 'pixelshuffle'
cfg['MODEL']['ACT']             = 'R'
cfg['MODEL']['EXCUTE']          = 'DPSR'  # DPSRGAN | DPSR
cfg['MODEL']['ITERS']           = 15

# path settings
cfg['PATH']                  = dict()
cfg['PATH']['IMAGE']            = 'data/noisy/img'
cfg['PATH']['VIDEO']            = 'data/noisy/video'
cfg['PATH']['pull_rtsp']        = 'rtmp://58.200.131.2:1935/livetv/gxtv'
#cfg.PATH.pull_rtsp        = '/home/ssh124/RKB_2TB/LLL_night_vision/dataset/cockatoo.mp4'
cfg['PATH']['SAVEI']            = 'data/denoised/img'
cfg['PATH']['SAVEV']            = 'data/denoised/video'
cfg['PATH']['push_rtsp']        = 'rtsp://172.16.45.1/test.sdp'
cfg['PATH']['WEIGHT']           = 'denoising/dpsr/weights/{}x{}.pth'.format(cfg['MODEL']['EXCUTE'],cfg['MODEL']['UPSCALE'])
cfg['PATH']['KERNEL']           = 'data/noisy/kernel'
# print(cfg)
