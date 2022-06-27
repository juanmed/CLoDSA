from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pylab
import cv2

image_directory = 'output/'
annotation_file = 'output/annotation.json'
image_directory = '/home/rise/Downloads/unloading_datasets/zerowaste/zerowaste-f-final/splits_final_deblurred/train/data/'
annotation_file = '/home/rise/Downloads/unloading_datasets/zerowaste/zerowaste-f-final/splits_final_deblurred/train/labels.json'
#image_directory = '/home/rise/'
#annotation_file = '/home/rise/'
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
print("Total images: ",len(image_ids))

cat_map = {1:"icebox", 2:"box", 3:"pouch", 4:"sack", 5:"bottle", 6: "cube"}

for k, cat in enumerate(category_names):
    cat_map[k+1] = cat

print(cat_map)   

ice = 0
box = 0 
pouch = 0 
sack = 0 
bottle = 0 
other = 0
total_anns = 0
num_anns = []
for image_id in image_ids[:5]:
    image_data = example_coco.loadImgs(image_id)[0]
    print("Currently viewing: {}".format(image_directory + image_data['file_name']))
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
        continue

    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    total_anns += len(annotation_ids)
    #assert len(annotation_ids) > 0, "{} has no annotations".format(image_data['file_name'])
    if len(annotation_ids) <= 0:
        print("{} has no annotations".format(image_data['file_name']))
    num_anns.append(len(annotation_ids))
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
        elif ann_id == 6:
            other +=1
        else:
            print("*** ID EXTRANO *** ", ann_id)

print(cat_map[1],":",  box)
print(cat_map[2],": ", ice)
print(cat_map[3],": ", pouch)
print(cat_map[4],": ", sack)
print(cat_map[5],": ", bottle)
print("Total annotations", total_anns)
print("Average annotation per image: ",total_anns/len(image_ids))

fig =plt.figure(figsize=(5,3))
ax1 = fig.add_subplot(1,1,1)
ax1.hist(num_anns,rwidth = 0.2, label = "", log=True)
#ax1.hist(num_anns, rwidth = 0.3, cumulative=True,label = "cumulative", color = "green")
ax1.set_title("Annotation frequency for validation split")
ax1.set_xlabel("Number of annotations")
ax1.set_ylabel("Image count")
ax1.legend(loc = "best")
#plt.hist(num_anns, rwidth = 0.3)
plt.tight_layout()
plt.show()

a = True
same = False
idx = 0
while a:
    if same == False:
        idx = np.random.randint(0, len(image_ids))
        idx += 1
    #idx = 10219
    #image_data = example_coco.loadImgs(idx)[0]
    image_data = example_coco.loadImgs(image_ids[idx])[0]

    same= False
    print("Currently viewing: {}".format(image_directory + image_data['file_name']))
    h = image_data['height']
    w = image_data['width']
    image = io.imread(image_directory + image_data['file_name'])
    plt.imshow(image); plt.axis('off')
    #pylab.rcParams['figure.figsize'] = (20.0, 20.0)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    example_coco.showAnns(annotations, draw_bbox=False)

    #print(annotations[0])
    ax = plt.gca()
    for ann in annotations:
        ann_id = ann["category_id"]
        name = cat_map[ann_id]
        bbox = ann["bbox"]
        if bbox[0] > w:
            print("bbox fuera de image : bbox x {}, width {}".format(bbox[0],w))
        if bbox[1] > h:
            print("bbox fuera de image : bbox y {}, height {}".format(bbox[1],h))
        ax.text(bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2, name,fontsize=12)
        ax.add_patch(Rectangle((bbox[0], bbox[1]),bbox[2], bbox[3],color="yellow", fill=False))
        ax.add_patch(Rectangle((0, 0),5, 5,color="red", fill=True))



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

