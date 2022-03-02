import json




def readCOCOJSON(jsonPath):
    with open(jsonPath) as f:
        data = json.load(f)

    info = {} # data['info']
    licenses = {} #data['licenses']
    categories = data['categories']
    images = data['images']
    annotations = data['annotations']

    dictImages = {}
    dictAnnotations = {}
    for image in images:
        dictImages[image['id']] = (image['file_name'],(image['width'],image['height']))
        dictAnnotations[image['id']] = []

    for annotation in annotations:
        dictAnnotations[annotation['image_id']].append((annotation['category_id'],
                                                        annotation['segmentation'][0]))

    return (info,licenses,categories,dictImages,dictAnnotations)

remap = {"box":"icebox"}    

image_directory = '/home/fer/Downloads/unloader_stacked/color/'
annotation_file = '/home/fer/Downloads/unloader_stacked/unloader_rgbd_20211103-10.json'