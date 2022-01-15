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
image_directory = '/home/fer/Downloads/5classes_stacked_random_sack_oscd/5classes_stacked_random_sack_oscd/train/'
annotation_file = '/home/fer/Downloads/5classes_stacked_random_sack_oscd/5classes_stacked_random_sack_oscd/annotations/new.json'
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
print("Cat IDS: {}".format(category_ids))
image_ids = example_coco.getImgIds(catIds=category_ids)

cat_map = {1:"icebox", 2:"box", 3:"pouch", 4:"sack", 5:"bottle", 6: "cube"}

ice = 0
box = 0 
pouch = 0 
sack = 0 
bottle = 0 
for image_id in image_ids[:0]:
    image_data = example_coco.loadImgs(image_id)[0]
    #print("Currently viewing: {}".format(image_directory + image_data['file_name']))
    try:
        image = cv2.imread(image_directory + image_data['file_name'], cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print("\nImage load Problem:  image_id {}, {}".format(image_id, image_directory + image_data['file_name']))
        print(str(e))
        continue

    height, width, _ = image.shape
    if ( image.shape[0]>0) and (image.shape[1] > 0) and (image is not None):
        pass
    else:
        print( " ##### ESTA IMAGEN NO ESTA ######")
        break
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    #annotations = example_coco.loadAnns(annotation_ids)
    target = [x for x in example_coco.loadAnns(annotation_ids) if x['image_id'] == image_id]
    masks = [example_coco.annToMask(obj).reshape(-1) for obj in target]
    
    try:
        masks = np.vstack(masks)
        masks = masks.reshape(-1, height, width)
    except Exception as e:
        print("\nMask stack Problem:  image_id {}, {}".format(image_id, image_directory + image_data['file_name']))
        print(str(e))
        continue


    for ann in target:
        ann_id = ann["category_id"]
        if ann_id == 1:
            ice += 1
        elif ann_id == 2:     
            box += 1
        elif ann_id == 3:
            pouch += 1
        elif ann_id == 4:
            sack += 1
        elif ann_id == 5:
            bottle +=1
        else:
            print("*** ID EXTRANO *** ", ann_id)

print("Boxes:",  box)
print("Icebox: ", ice)
print("Pouch: ", pouch)
print("Sack: ", sack)
print("Bottle: ", bottle)


a = True
same = False
idx = 0
while a:
    if same == False:
        idx = np.random.randint(0, len(image_ids))
        idx += 1
    idx = 10219
    image_data = example_coco.loadImgs(idx)[0]
    #image_data = example_coco.loadImgs(image_ids[idx])[0]

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

