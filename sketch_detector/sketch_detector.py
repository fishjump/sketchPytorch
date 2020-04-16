import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from . import helper
from .sketch_pytorch import sketch_pytorch


mod = sketch_pytorch(os.path.split(os.path.realpath(__file__))[0] + '/sketch_pytorch_weight.npy')
mod.cuda()
mod.eval()

def get(path):
    img = cv2.imread(path)
    line_mat = pre_process(img)
    helper.show_active_img_and_save('sketchKeras_colored', line_mat, 'sketchKeras_colored.jpg')
    line_mat = np.amax(line_mat, 2)
    helper.show_active_img_and_save_denoise_filter2('sketchKeras_enhanced', line_mat, 'sketchKeras_enhanced.jpg')
    helper.show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat, 'sketchKeras_pured.jpg')
    helper.show_active_img_and_save_denoise('sketchKeras', line_mat, 'sketchKeras.jpg')
    cv2.waitKey(0)
    return

def pre_process(img):
    width = float(img.shape[1])
    height = float(img.shape[0])
    new_width = 0
    new_height = 0
    if (width > height):
        img = cv2.resize(img, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        img = cv2.resize(img, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512

    img = img.transpose((2, 0, 1))
    light_map = np.zeros(img.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = helper.get_light_map_single(img[channel])
    light_map = helper.normalize_pic(light_map)
    light_map = helper.resize_img_512_3d(light_map)
    line_mat = mod(torch.tensor(light_map, dtype=torch.float).cuda())
    line_mat = line_mat.permute((1, 2, 3, 0)).cpu().detach().numpy()[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    
    return line_mat

def get_colored_sketch(img):
    img = pre_process(img)
    return helper.get_colored_sketch(img).reshape(img.shape[0], img.shape[1], 1)

def get_pured_sketch(img):
    img = pre_process(img)
    img = np.amax(img, 2)
    return helper.get_pured_sketch(img).reshape(img.shape[0], img.shape[1], 1)

def get_enhanced_sketch(img):
    img = pre_process(img)
    img = np.amax(img, 2)
    return helper.get_enhanced_sketch(img).reshape(img.shape[0], img.shape[1], 1)

def get_sketch(img):
    img = pre_process(img)
    img = np.amax(img, 2)
    return helper.get_sketch(img).reshape(img.shape[0], img.shape[1], 1)