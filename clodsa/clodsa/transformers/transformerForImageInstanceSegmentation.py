from __future__ import absolute_import
from .transformer import Transformer
from ..techniques.technique import PositionVariantTechnique, BackgroundReplaceTechnique, RandomRotateTechnique
import random



class TransformerForImageInstanceSegmentation(Transformer):

    def __init__(self,technique,dictLabels=None):
        Transformer.__init__(self,technique,dictLabels)

    def transform(self, image,maskLabels):
        if (isinstance(self.technique, RandomRotateTechnique)):
            angle = random.randint(0,360)
            return [self.technique.apply2(image, angle),
                    [(self.technique.apply2(mask, angle), self.transformLabel(label))
                     for mask,label in maskLabels ]]

        elif (isinstance(self.technique, BackgroundReplaceTechnique)):
            return [self.technique.apply2(image, maskLabels),
                    [(mask, self.transformLabel(label))
                     for mask, label in maskLabels]]

        elif (isinstance(self.technique, PositionVariantTechnique)):
            return [self.technique.apply(image),
                    [(self.technique.apply(mask), self.transformLabel(label))
                     for mask,label in maskLabels ]]

        else:
            return [self.technique.apply(image),
                    [(mask, self.transformLabel(label))
                     for mask, label in maskLabels]]