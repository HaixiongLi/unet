import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet

from utils.dataset import BasicDataset


Pixel=[(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)]


num_classes = 20

input_img = './000000_10.png'
save_bin_dir = 'city_bin_res_04.png'
save_color_dir = 'city_color_res_04.png'


parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', '-m', default='./checkpoints_cityscapes_20class_50epochs/CP_epoch50.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

args = parser.parse_args()

def predict_img_color(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        index = torch.max(probs.cpu(),0)[1]
        index = np.array(index).astype('uint8')
    return index

def mask_to_image(mask):
    a = (mask).astype(np.uint8)
    b = Image.fromarray(a)
    return b

def mask_to_colorimage(original_image,firename):
    original_image = (original_image).astype(np.uint8)
    image=np.zeros(shape=(np.shape(original_image)[0],np.shape(original_image)[1],3),dtype=np.uint8)
    for m in range(np.shape(original_image)[0]):
        for n in range(np.shape(original_image)[1]):
            class_num=int(original_image[m,n])
            if class_num==19:
                color=(0,0,0)
            else:
                color=Pixel[class_num]
            image[m,n,0]=color[0]
            image[m, n, 1] = color[1]
            image[m, n, 2] = color[2]

    img = Image.fromarray(image.astype('uint8')).convert('RGB')
    img.save(firename)

if __name__ == "__main__":

    in_file = input_img

    net = UNet(n_channels=3, n_classes=num_classes)
    logging.info("Loading model {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    img = Image.open(in_file)

    mask = predict_img_color(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

    result = mask_to_image(mask)
    result.save(save_bin_dir)
    result = mask_to_colorimage(mask,save_color_dir)

    logging.info("Mask saved to {}".format(save_bin_dir))
