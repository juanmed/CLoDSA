import json

ann_dir = "/home/rise/Downloads/unloading_datasets/results/merged/annotations/merged.json"
#ann_dir = "/home/rise/repos/CLoDSA/clodsa/val2.json"
with open(ann_dir,'r') as coco_file:
    coco_labels = json.load(coco_file)


print(coco_labels.keys())
print(coco_labels['categories'])

anns = coco_labels['annotations']
seg_corr = 0
new_anns = []
for ann in anns:
    append = True
    x,y,w,h = ann['bbox']
    if len(ann['segmentation'][0]) <= 4:
        print(ann['segmentation'])
        ann['segmentation'][0].append(ann['segmentation'][0][0])
        ann['segmentation'][0].append(ann['segmentation'][0][1])
        seg_corr += 1
        append = False
        print("Invalid segmentation < 4 points")
    
    #ann['bbox'] = [x - 304,y - 228, w, h]
    elif (x < 0) or (y < 0) or (w < 0) or (h<0):
        print("Invalid X, Y ",x,y,w,h)
        append = False
    else:
        pass


    if append:
        new_anns.append(ann)

coco_labels['annotations'] = new_anns
print("Total errors:", seg_corr)


new_ann = "new.json"
with open(new_ann, 'w') as file:
  json.dump(coco_labels, file)

file.close()

with open(new_ann,'r') as coco_file:
    coco_labels = json.load(coco_file)

anns = coco_labels['annotations']
seg_corr = 0
for ann in anns:
    x,y,w,h = ann['bbox']
    if len(ann['segmentation'][0]) <= 4:
        print(ann['segmentation'])
        seg_corr += 1
    elif (x < 0) or (y < 0) or (w < 0) or (h<0):
        print("Invalid X, Y ",x,y,w,h)
        seg_corr += 1
    else:
        pass
print("Total errors 2nd pass:", seg_corr)
