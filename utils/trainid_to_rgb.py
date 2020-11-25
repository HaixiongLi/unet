import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
Name=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
Label=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
Pixel=[(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)]

# label_matrix=np.load('./000000_10.npy')
def trainid_to_RGB(original_image,firename):
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
    img.show()
    img.save(firename)



# plt.imshow(original_image)
# plt.show()
