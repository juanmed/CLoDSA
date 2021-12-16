from matplotlib import pyplot as plt
import sys

from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import cv2
#%matplotlib inline

PROBLEM = "instance_segmentation"
ANNOTATION_MODE = "coco"
INPUT_PATH = "./images/"
GENERATION_MODE = "sequential"
OUTPUT_MODE = "coco"
OUTPUT_PATH= "output/"
augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH})
transformer = transformerGenerator(PROBLEM)


#for angle in [90,180]:



t = createTechnique("random_object_non_occlusion", 
					{"random_images_dir":"/home/fer/repos/CLoDSA/random_instances/",
					"rate":0.5,
					"resize_factor":0.4})
augmentor.addTransformer(transformer(t))

#t = createTechnique("resize", {"x":20,"y":20})
#augmentor.addTransformer(transformer(t))


#rotate = createTechnique("random_rotate", {"range":[-90,90]})
#augmentor.addTransformer(transformer(rotate))

#flip0 = createTechnique("random_object_occlusion",
#						{"random_images_dir":"/home/fer/repos/CLoDSA/random_instances/color"})
#flip1 = createTechnique("flip",{"flip":1})
#flipm1 = createTechnique("flip",{"flip":-1})
#augmentor.addTransformer(transformer(flip0))
#augmentor.addTransformer(transformer(flip1))
#augmentor.addTransformer(transformer(flipm1))


#blur = createTechnique("blurring", {"ksize" : 5})
#augmentor.addTransformer(transformer(blur))

#gamma = createTechnique("gamma",{"gamma":1.5})
#augmentor.addTransformer(transformer(gamma))

#ea = createTechnique("elastic",{})
#augmentor.addTransformer(transformer(ea))

#for shearval in range(1,5,1):
#shear = createTechnique("shearing", {"a":0.05})
#augmentor.addTransformer(transformer(shear))

#none = createTechnique("none",{})
#augmentor.addTransformer(transformer(none))

rotate = createTechnique("random_rotate", {"range":[-90,90]})
augmentor.addTransformer(transformer(rotate))

#sp = createTechnique("salt_and_pepper", {"low" : 0,"up":50})
#augmentor.addTransformer(transformer(sp))

#do = createTechnique("dropout", {"percentage" : 0.2})
#augmentor.addTransformer(transformer(do))
#bkg_img_dir = "/home/fer/repos/CLoDSA/random_backgrounds/VOCdevkit/VOC2012/JPEGImages/"
#br = createTechnique("background_replacement", {"background_images_dir": bkg_img_dir})
#augmentor.addTransformer(transformer(br))





augmentor.applyAugmentation()


