import os
import time
import cv2

import denoising.dpsr.dpsr as dpsr
from config.config import cfg

def main(video_name):

    # -- get the video path
    video_path = os.path.join(cfg.PATH.VIDEO, video_name)

    # -- check the video type(video or cam)
    if video_name is None:
        print('There is no video or the path is incorrect.')
        return
    videoCapture = cv2.VideoCapture(video_path)
    if not videoCapture.isOpened():
        print('Cannot open the video.')
        videoCapture.release()
        return

    # -- cv2 info
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
        fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', 'V')
    else:
        fh = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')

    print(fw, fh, fps)

    # -- instance for video writer
    videoWriter = cv2.VideoWriter(os.path.join(cfg.PATH.SAVEV, video_name), fourcc, fps, (int(fw/2), int(fh/2)))

    # -- instance for the dpsr model
    dpsr_model = dpsr.DPSR(
                cfg.PATH.WEIGHT, 
                noise_level=cfg.IMG.NOISE, 
                n_channels=cfg.IMG.CHANNEL, 
                upscale=cfg.MODEL.UPSCALE,
                act_mode=cfg.MODEL.ACT,
                upsample_mode=cfg.MODEL.UPSAMPLE,
                method=cfg.MODEL.EXCUTE)

    # -- process
    while True:

        # -- capture the video frame
        _, frame = videoCapture.read()

        # -- check the video frame
        if frame is None:
            break

        start_time = time.time()

        # -- denoise the image
        h, w = frame.shape[:2]
        #frame = cv2.resize(frame, (int(w/2), int(h/2)))
        frame = dpsr_model.denoising(frame)

        denoising_time = time.time() - start_time

        # -- process the res
        print('denoising time in ', denoising_time, 's')
        cv2.imshow('video', frame)
        videoWriter.write(frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    videoCapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('vlc-record-2019-09-09-19h37m43s-rtsp___192.168.1.88_554_main-.mp4')