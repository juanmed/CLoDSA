from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
from .technique import PositionVariantTechnique
import cv2

class cropByCoordinatesAugmentationTechnique(PositionVariantTechnique):

    # percentage is a value between 0 and 1
    # startFrom indicates the starting point of the cropping,
    # the possible values are TOPLEFT, TOPRIGHT, BOTTOMLEFT,
    # BOTTOMRIGHT, and CENTER
    def __init__(self,parameters):
        PositionVariantTechnique.__init__(self, parameters)
        if 'xywh' in list(parameters.keys()):
            self.xywh = float(parameters["xywh"])
        else:
            self.xywh = (0,0,256,256)

    def apply(self, image):
        (h,w) = image.shape[:2]
        x,y,w,h = self.xywh
        crop = image[y:y_h,x:x+w]
        return crop


# Example
# technique = cropAugmentationTechnique(0.5,'TOPRIGHT')
# image = cv2.imread("LPR1.jpg")
# cv2.imshow("original",image)
# cv2.imshow("new",technique.apply(image))
# cv2.waitKey(0)