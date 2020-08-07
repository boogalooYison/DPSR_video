#!/usr/bin/env python
# encoding:utf-8

import cv2
import time
import subprocess as sp
# import multiprocessing as mp
import torch.multiprocessing as mp

import denoising.dpsr.dpsr as dpsr
from config.config import cfg


def read_frame(flag,q_command,q_frame):
    # instance for the dpsr model
    dpsr_model = dpsr.DPSR(
        cfg.PATH.WEIGHT,
        noise_level=cfg.IMG.NOISE,
        n_channels=cfg.IMG.CHANNEL,
        upscale=cfg.MODEL.UPSCALE,
        act_mode=cfg.MODEL.ACT,
        upsample_mode=cfg.MODEL.UPSAMPLE,
        method=cfg.MODEL.EXCUTE)

    cap = cv2.VideoCapture(cfg.PATH.pull_rtsp)
    if not cap.isOpened():
        print('Cannot open the camera.')
        cap.release()

    push_rtsp_url = cfg.PATH.push_rtsp
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    flag_command = True
    while True:
        ret,frame = cap.read()
        if not ret:
            break

        #cv2.imshow("frame",frame)
        start_time = time.time()
        # -- denoising
        img_denoise = dpsr_model.denoising(frame)

        denoising_time = time.time() - start_time

        print('denoising time in ', denoising_time, 's')
        #cv2.imshow("denoising",img_denoise);cv2.wiatKey()
        #cv2.imwrite("test.png",img_denoise)

        if flag_command:
            width = img_denoise.shape[1]
            height = img_denoise.shape[0]

            command = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps//4),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'rtsp',
                push_rtsp_url
            ]
            q_command.put(command)
            flag_command = False

        q_frame.put(img_denoise)
        #p.stdin.write(img_denoise.tostring())

    #cv2.destroyAllWindows()
    cap.release()
    flag.value = 0
    print("read_frame finish")


def push_frame(flag,q_command,q_frame):
    command = []
    while flag.value:
        if q_command.empty():
            time.sleep(1)
            continue
        command = q_command.get()
        break
    print("command:",command)
    p = sp.Popen(command, stdin=sp.PIPE)
    while flag.value:
        if q_frame.empty():
            time.sleep(1)
            continue
        frame = q_frame.get()
        p.stdin.write(frame.tostring())
    print("push_frame finish")

def run():
    mp.set_start_method('spawn')
    # mp.set_start_method('forkserver')

    flag = mp.Value("i",1)
    # q_command = mp.Queue()
    # q_frame = mp.Queue()
    q_command = mp.Manager().Queue()
    q_frame = mp.Manager().Queue()
    p1 = mp.Process(target=read_frame,args=(flag,q_command,q_frame))
    p2 = mp.Process(target=push_frame,args=(flag,q_command,q_frame))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("run finish")

if __name__ == '__main__':
    run()