from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
import cv2

image_directory = 'output/'
annotation_file = 'output/annotation.json'
#image_directory = '/home/fer/repos/CLoDSA/clodsa/results/merged/images/'
#annotation_file = '/home/fer/repos/CLoDSA/clodsa/results/merged/images/merged.json'
image_directory = '/home/fer/Downloads/01.final_dataset/polygon/train/image/'
annotation_file = '/home/fer/Downloads/01.final_dataset/polygon/train/SKKU_polygon_train.json'
#image_directory = 'images/minjae/'
#annotation_file = 'images/minjae/annotations.json'
example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
#print("Categories:", categories)
category_names = [category['name'] for category in categories]
#print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
print(category_names)

#category_names = set([category['supercategory'] for category in categories])
#print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['circle'])
image_ids = example_coco.getImgIds(catIds=category_ids)

cat_map = {1:"icebox", 2:"box", 3:"pouch", 4:"sack", 5:"bottle", 6: "cube"}


for image_id in image_ids[0:1]:
    image_data = example_coco.loadImgs(image_id)[0]
    print("Currently viewing: {}".format(image_directory + image_data['file_name']))
    image = cv2.imread(image_directory + image_data['file_name'], cv2.IMREAD_UNCHANGED)
    if ( image.shape[0]>0) and (image.shape[1] > 0) and (image is not None):
        pass
    else:
        print( " ##### ESTA IMAGEN NO ESTA ######")
        break



a = True
same = False
idx = 0
while a:
    if same == False:
        idx = np.random.randint(0, len(image_ids))
        idx += 1
    image_data = example_coco.loadImgs(image_ids[idx])[0]

    same= False
    print("Currently viewing: {}".format(image_directory + image_data['file_name']))
    image = io.imread(image_directory + image_data['file_name'])
    plt.imshow(image); plt.axis('off')
    pylab.rcParams['figure.figsize'] = (20.0, 20.0)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    example_coco.showAnns(annotations, draw_bbox=False)

    #print(annotations[0])
    ax = plt.gca()
    for ann in annotations:
        ann_id = ann["category_id"]
        name = cat_map[ann_id]
        bbox = ann["bbox"]
        ax.text(bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2, name,fontsize=12)



    plt.show()
    #cv2.imshow('img',cv2.resize(image,None,fx=0.25,fy=0.25))
    k = input("Opciones: ")
    if k=="q":  # normally -1 returned,so don't print it
        print (k)
        break
    elif k == "t":
        same = True
        print(k)
        continue
    else:
        print (k) # else print its value
        continue

