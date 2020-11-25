import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from PIL import Image,ImageOps
from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset_cityscapes import cityscapes
from torch.utils.data import DataLoader, random_split
from utils.metrics import SegmentationMetric
from utils.transform import Relabel, ToLabel, Colorize
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage


num_classes = 19

metric = SegmentationMetric(num_classes)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
imaglescale=1   #图像尺寸为1

# dir_img = '/home/lab2/work/lhx/data/data_semantics/training/image_2_resize/'
# dir_mask = '/home/lab2/work/lhx/data/data_semantics/training/id/'
# dir_checkpoint = 'checkpoints_kitti_20class_50epochs/'
dir_checkpoint = 'checkpoints_cityscapes_8class_50epochs/'

datadir = '/media/lab2/机械盘1/Dataset/cityscapes/datasets/'


parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                    help='Number of epochs', dest='epochs')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=3,
                    help='Batch size', dest='batchsize')
parser.add_argument('--datadir', default="/media/lab2/机械盘1/Dataset/cityscapes/datasets/")
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--height', type=int, default=512)
parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                    help='Learning rate', dest='lr')
parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                    help='Load model from a .pth file')
parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                    help='Downscaling factor of the images')
parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                    help='Percent of the data that is used as validation (0-100)')

args = parser.parse_args()


# Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc = enc
        self.augment = augment
        self.height = height
        pass

    def __call__(self, input, target):
        # do something to both images
        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if (self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255)  # pad label filling with 255
            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height / 8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 7)(target)

        return input, target

def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp=True,
              img_scale=imaglescale):


    # dataset = cityscapes(dir_img, dir_mask, img_scale)

    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    enc = False
    assert os.path.exists(datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(enc, augment=True, height=args.height)#512)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)#512)
    dataset_train = cityscapes(args.datadir,co_transform,'train')
    dataset_val = cityscapes(args.datadir,co_transform_val,'val')
    print(dataset_val)
    n_train = int(len(dataset_train))
    n_val = int(len(dataset_val))

    train_loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=batch_size, shuffle=False)




    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    global_step = 0
    train_fwiou_sum = 0.0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=10, min_lr=1e-8)

    criterion = nn.CrossEntropyLoss(ignore_index=255)


    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type =  torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                true_masks = true_masks.squeeze(1)
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)

                for i in range(masks_pred.shape[0]):
                    pre = masks_pred[i, :, :, :].argmax(axis=0).cpu()
                    label = true_masks[i, :, :].cpu()
                    metric.addBatch(pre, label)
                    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
                    train_fwiou_sum += FWIoU
                train_fwiou = train_fwiou_sum/true_masks.shape[0]
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                writer.add_scalar('FWIOU/train_FWIou', train_fwiou,global_step)

                # pbar.set_postfix(**{'loss(batch)':loss.item()},**{'fwiou (batch)':train_fwiou})
                pbar.set_postfix(**{'fwiou':train_fwiou},**{'loss':loss.item()})

                # pbar.set_postfix()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score, val_FWIoU= eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation cross entropy: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)

                    writer.add_scalar('FWIOU/test_FWIou', val_FWIoU,global_step)

                    writer.add_images('images', imgs, global_step)


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=num_classes, bilinear=True)




    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


