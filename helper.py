import numpy as np

CITYSCAPE_PALLETE = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (0, 0, 0)]

width = 1245
height = 375

def logits2image(logits):
    # logits = logits.item()
    image = np.zeros([3,logits.shape[0],logits.shape[1]]).astype(np.uint8)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if(any(logits[i,j] == 255)):
                image[:,i,j] = CITYSCAPE_PALLETE[19]
            else:
                image[:,i,j] = CITYSCAPE_PALLETE[logits[i,j]]
    # image = logits.numpy.astype(np.uint8)
    return image