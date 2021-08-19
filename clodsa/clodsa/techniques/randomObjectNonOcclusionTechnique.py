from __future__ import absolute_import
from .technique import PositionInvariantTechnique
import cv2

class randomObjectNonOcclusionTechnique(PositionInvariantTechnique):

    def __init__(self,parameters):
        PositionInvariantTechnique.__init__(self, parameters)
        self.parameters = parameters

    def apply(self, image):
        return image

    def apply2(self, image, maskLabels):

        # list all files and get only images
        rand_img_files = os.listdir(self.parameters["random_images_dir"])
        rand_img_files = [f for f in bkg_img_files if ((".jpg" in f) or (".png" in f))]
        # load and resize one random background image
        rand_img = cv2.imread(os.path.join(self.parameters["random_images_dir"], random.sample(bkg_img_files,1)[0]))
        
        return rand_img