import json
import os
from pycocotools.coco import COCO 
import numpy as np
import skimage.io as io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

raw_dir = "../clodsa/images/color"
output_dir = "../random_instances/color/"

def main():
    # the coco annotation file is named "train.json" and
    # is located inside raw_dir
    split = "train"
    label_dir = os.path.join(raw_dir,'annotations.json')
    with open(label_dir,'r') as label_file:
        labels = json.load(label_file)
    # load dataset
    print(labels.keys())
    coco_anns = COCO(label_dir)
    # get categories and print
    cats = coco_anns.loadCats(coco_anns.getCatIds())
    category_names = [category['name'] for category in cats]
    print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
    print(cats)
    # get all images containing given categories, select one at random
    catIds = coco_anns.getCatIds(catNms=['icebox','box','pouch', 'sack','bottle']);
    print(catIds)
    #print(catIds)
    imgIds = coco_anns.getImgIds(catIds=catIds);
    print(imgIds)
    imgIds= [2,3]
    for imgId in imgIds[:]:
        img_data = coco_anns.loadImgs(imgId)[0]
        # select one annotation, read image and draw image with annotation
        annIds = coco_anns.getAnnIds(imgIds = img_data['id'], catIds=catIds, iscrowd=None)
        anns = coco_anns.loadAnns(annIds)
        img_path = os.path.join(raw_dir,img_data['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #dimg_path = img_path.replace("color","depth")
        #dimg = cv2.imread(dimg_path, cv2.IMREAD_UNCHANGED)
        #print(np.mean(dimg))
            
        found = False
        
        data= []
        for j, ann in enumerate(anns):
            x,y,w,h = [int(_) for _ in ann['bbox']]
            bbox_area = w*h
            seg_area = ann['area']
            fill_rate = (seg_area * 1.0) / bbox_area
            #print(fill_rate)
            data.append(fill_rate)
            if fill_rate > 0.9:
                found = True
                #ann = anns[np.random.randint(0,len(anns))]
                #print(ann)
                x,y,w,h = [int(_) for _ in ann['bbox']]
                #print(x,y,w,h)
                instance_patch_rgb = img[y:y+h,x:x+w,:]
                #instance_patch_depth = dimg[y:y+h,x:x+w]
                patch_name = str(ann['category_id']) +"_"+str(j)+"_"+ img_data['file_name']
                cv2.imwrite(os.path.join(output_dir,patch_name),instance_patch_rgb)
                #cv2.imwrite(os.path.join(output_dir.replace("color","depth"),patch_name),instance_patch_depth)
                #store segmentation mask
                cnt = np.array([int(_) for _ in ann['segmentation'][0]]).reshape(-1,2)

                mask = np.zeros((img.shape[0],img.shape[1]))
                mask = cv2.drawContours(mask, [cnt], 0, 255, -1)
                mask = mask[y:y+h,x:x+w]
                cv2.imwrite(os.path.join(output_dir.replace("color","mask"),patch_name),mask)           

        print(np.max(data))



    #fig = plt.figure()
    #ax1 = fig.add_subplot(1,1,1)
    #ax1.imshow(img)
    #coco_anns.showAnns(anns)
    #plt.show()

if __name__ == '__main__':
    main()