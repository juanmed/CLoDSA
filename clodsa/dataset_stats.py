import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import math
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.cbook import get_sample_data
import cv2
from pycocotools.coco import COCO

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', '-a', type=str, default='examples/instances_val2017.json')
    parser.add_argument('--result', '-r', default='examples/coco_instances_results.json', type=str)
    parser.add_argument('--name', '-n', default='', type=str)
    parser.add_argument('--show', type=bool, default=True)
    parser.add_argument('--normalize', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():

    args = arg()
    example_coco = COCO(args.annotation)

    categories = example_coco.loadCats(example_coco.getCatIds())
    #print("Categories:", categories)
    category_names = [category['name'] for category in categories]
    #print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
    print(category_names)

    category_ids = example_coco.getCatIds(catNms=['circle'])
    print("Cat IDS: {}".format(category_ids))
    image_ids = example_coco.getImgIds(catIds=category_ids)

    bbox_cx = []
    bbox_cy = []
    step = 0.03

    bbox_rcx = []
    bbox_rcy = []

    for image_id in image_ids[:]:
        image_data = example_coco.loadImgs(image_id)[0]

        w = image_data['width']
        h = image_data['height']

        annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
        annotations = example_coco.loadAnns(annotation_ids)
        for p in annotations:
            try:
                normalized_cx = np.min([ (p['bbox'][0] + (p['bbox'][2]//2))/(w*1.0), 1.0]) # gt or pred can have bbox going out of image...
                normalized_cy = np.min([ (p['bbox'][1] + (p['bbox'][3]//2))/(h*1.0), 1.0])
                assert normalized_cx <= 1.0, "Dato CX extrano {}, x{}, y{}, w{}, h{}, cx{}".format(w,p['bbox'][0] ,p['bbox'][1] ,p['bbox'][2],p['bbox'][3], normalized_cx )
                assert normalized_cy <= 1.0, "Dato CY extrano {}, y{}, x{}, h{}, w{},cy{}".format(h,p['bbox'][1] ,p['bbox'][0] ,p['bbox'][3],p['bbox'][2], normalized_cy )
                assert normalized_cx >= 0.0, "Dato CX extrano {}, x{}, y{}, w{}, h{},cx{}".format(w,p['bbox'][0] ,p['bbox'][1] ,p['bbox'][2],p['bbox'][3], normalized_cx )
                assert normalized_cy >= 0.0, "Dato CY extrano {}, y{}, x{}, h{}, w{},cy{}".format(h,p['bbox'][1] ,p['bbox'][0] ,p['bbox'][3],p['bbox'][2], normalized_cy )
                bbox_cx.append( normalized_cx) #normalize to width and height
                bbox_cy.append( normalized_cy)
                bbox_rcx.append((p['bbox'][0] + p['bbox'][2]//2))
                bbox_rcy.append((p['bbox'][1] + p['bbox'][3]//2))
            except Exception as e:
                print("Algo raro")

    plt.hist(bbox_cx, label= "x center (norm)", alpha = 0.5)
    plt.hist(bbox_cy, label = "y center(norm)", alpha = 0.5)
    plt.legend()
    plt.xlabel("normalized bbox coordinate")
    #print("bbox_cx : ",len(bbox_cx),len(bbox_cy))
    #img = cv2.imread('C:\\Users\\lenovo\\Documents\\projects\\unloading\\failure.tar\\failure\\textureless_brackground.png', cv2.IMREAD_COLOR)
    img = cv2.imread('/home/rise/Desktop/failure/textureless_brackground.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx = 0.125, fy = 0.125)
    img = img.astype('float32')/255

    # setup the figure and axes
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    # grid in normalized image coordinate and step of 0.1... or whatever
    
    _x = np.arange(0.,1.,step) * img.shape[0]
    _y = np.arange(0.,1.,step) * img.shape[1]
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    print("xx :",_xx.shape, _yy.shape)
    #top = x + y
    top, count = np.meshgrid(_x, _y)
    print("top: ",top.shape,count.shape)
    count = count*0.

    #for k, (iou, cx, cy) in enumerate(zip(ious, bbox_cx, bbox_cy)):
    for k, (cx, cy) in enumerate(zip( bbox_cx, bbox_cy)):

        x_coor = int(np.floor(cx/(step + step*0.01 ) ))
        y_coor = int(np.floor(cy/(step + step*0.01 ) ))
        #top[y_coor][x_coor] += iou 
        count[y_coor][x_coor] += 1.

    #top = top/count  # take average of each cell by dividing by count of elements
    #top[np.isnan(top)] = 0

    #print("top : {}\n{}".format(top.shape,top))
    #print("count : {}\n{}".format(count.shape,count))
    #print("x: {}\n{}".format(x.shape,x))
    #print("y: {}\n{}".format(y.shape,y))

    bottom = np.zeros_like(count)
    width = step * img.shape[0]
    depth = step * img.shape[1]
    colors = plt.cm.coolwarm(count.flatten()/float(count.max()))
    ax1.bar3d(x, y, bottom.ravel(), width, depth, count.ravel(), color = colors, edgecolor='black', linewidth=0.1, alpha = 0.7)
    #ax1.scatter(x, y, top.ravel(), color = colors, edgecolor='black', linewidth=0.1)
    #ax1.plot_wireframe(_xx, _yy, top, rstride=int(1/step), cstride=int(1/step))
    ax1.set_title('BBox count as function of position in image')
    ax1.set_xlabel("Y")
    ax1.set_ylabel("X")
    ax1.set_zlabel("BBox count")
    #ax1.set_zlim(-2.01, count.max())

    #stepX, stepY = 1. / (img.shape[0]), 1. / (img.shape[1])

    x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    #X1, Y1 = np.meshgrid(X1, Y1)
    # stride args allows to determine image quality 
    # stride = 1 work slow
    #ax1.plot_surface(X1, Y1, np.zeros_like(X1), rstride=4, cstride=4, facecolors=img)
    ax1.plot_surface(x, y, np.atleast_2d(-count.max()), rstride=2, cstride=2, facecolors=img)

    plt.show()


    """
    with open(path, 'r') as args.annotation:
        cocojson = json.load(json_file)
    
    images = cocojson['images']
    anns   = cocojson['annotations']
    cats   = cocojson['categories'] if 'categories' in cocojson else None
    """

if __name__ == '__main__':
    main()