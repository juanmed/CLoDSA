from __future__ import absolute_import
from .technique import PositionInvariantTechnique
import cv2
import numpy as np
import os
import random

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

class randomObjectNonOcclusionTechnique(PositionInvariantTechnique):

    def __init__(self,parameters):
        PositionInvariantTechnique.__init__(self, parameters)
        self.parameters = parameters

    def apply(self, image):
        return image

    def apply2(self, image, maskLabels):

        # list all files and get only images
        rand_img_files = os.listdir(self.parameters["random_images_dir"])
        rand_img_files = [f for f in rand_img_files if ((".jpg" in f) or (".png" in f))]
        # load and resize one random background image
        rand_img = cv2.imread(os.path.join(self.parameters["random_images_dir"], random.sample(rand_img_files,1)[0]))
        # add random image overlaid on random instance. For this, make
        # a list of random 1s and 0s the same length as maskLabels. For
        # those idx where the list is 1, place a random image on the
        # segment with the same idx in maskLabels.
        draw_not_draw = [0] * len(maskLabels)
        while (1.0*np.sum(draw_not_draw))/len(draw_not_draw) < self.parameters["rate"]:
            i = random.randint(0,len(draw_not_draw)-1)
            draw_not_draw[i] = 1

        #print(draw_not_draw)
        for j, (mask, label) in enumerate(maskLabels):
            if draw_not_draw[j] == 1:
                # overlay image
                contours , hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                c, s, a = cv2.minAreaRect(contours[0]) # ((center_x,center_y),(width,height),angle)
                rand_img = cv2.imread(os.path.join(self.parameters["random_images_dir"], random.sample(rand_img_files,1)[0]), cv2.IMREAD_COLOR)
                r_factor = self.parameters["resize_factor"]
                rand_img = cv2.resize(rand_img, (int(s[0]*r_factor),int(s[1]*r_factor)))
                rand_img = rotate_image(rand_img, a+1)
                rand_img_mask = np.zeros_like(image)
                rand_w = random.randint(-s[0]//2,s[0]//2) # random location inside minAreaRect
                rand_h = random.randint(-s[1]//2,s[1]//2)
                h_min = max(0,int(c[1]+rand_h) - rand_img.shape[0]//2)
                h_max = min(rand_img_mask.shape[0],int(c[1]+rand_h) + rand_img.shape[0]//2)
                w_min = max(0,int(c[0]+rand_w) - rand_img.shape[1]//2)
                w_max = min(rand_img_mask.shape[1],int(c[0]+rand_w) + rand_img.shape[1]//2)
                #h_min = int(c[1]+rand_h) - rand_img.shape[0]//2
                #h_max = int(c[1]+rand_h) + rand_img.shape[0]//2
                #w_min = int(c[0]+rand_w) - rand_img.shape[1]//2
                #w_max = int(c[0]+rand_w) + rand_img.shape[1]//2
                #print(h_max, h_min, w_max, w_min)
                #print(h_max- h_min, w_max- w_min,rand_img.shape)
                #print(c)

                dmask = np.dstack((mask,mask,mask))

                try:
                    rand_img_mask[h_min:h_max,w_min:w_max,:] = rand_img[0:h_max - h_min, 0:w_max-w_min]
                except:
                    pass

                cond = np.logical_and(dmask, rand_img_mask)
                image = np.where(cond,rand_img_mask,image)

            else:
                continue

        return image