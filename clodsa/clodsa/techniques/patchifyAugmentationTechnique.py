from .technique import PositionVariantTechnique
import cv2
from cropByCoordinatesAugmentationTechnique import cropByCoordinatesAugmentationTechnique as Cropper


class patchifyAugmentationTechnique(PositionVariantTechnique):

	def __init__(self, parameters):
		PositionVariantTechnique.__init__(self, parameters)
        if 'patchify' in list(parameters.keys()):
            self.f = parameters['patchify']
        else:
            self.f = 1

	def apply(self,image):
		patches_coords = self.f(image)
		patches = []
		for k, patch_coord in enumerate(patches_coords):
			x,y,w,h = patch_coord
			patches.append(image[y:y+h, x:x+w])
		return patches if len(patches)!=1 else patches[0]

