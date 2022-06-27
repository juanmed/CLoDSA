from __future__ import absolute_import
from .technique import RandomObjectOcclusionTechnique
import cv2
import numpy as np
import os
import random
import copy
#from nest2D import Point, Box, Item, nest, SVGWriter

class randomObjectOclussionTechnique(RandomObjectOcclusionTechnique):

    def __init__(self,parameters):
        RandomObjectOcclusionTechnique.__init__(self, parameters)
        self.parameters = parameters
        self.input_image_name = ""
        #self.sw = SVGWriter()

    def apply(self, image):
        return image

    def plot_solution(self, all_rects, pal_812, pal_1012):
        # Plot
        plt.figure(figsize=(10,10))
        # Loop all rect
        for rect in all_rects:
            b, x, y, w, h, rid = rect
            x1, x2, x3, x4, x5 = x, x+w, x+w, x, x
            y1, y2, y3, y4, y5 = y, y, y+h, y+h,y

            # Pallet type
            if [w, h] == pal_812:
                color = '--k'
            else:
                color = '--r'

            plt.plot([x1,x2,x3,x4,x5],[y1,y2,y3,y4,y5], color)
        plt.axis('equal')
        plt.show()

    # Function Solver
    def solver(self, n_812, n_1012, bins):
        # Pallets to load
        rectangles = [pal_812 for i in range(n_812)] + [pal_1012 for i in range(n_1012)]
        
        # Build the Packer
        pack = newPacker(mode = packer.PackingMode.Offline, bin_algo = packer.PackingBin.Global,
                         rotation=True)

        # Add the rectangles to packing queue
        for r in rectangles:
            pack.add_rect(*r)

        # Add the bins where the rectangles will be placed
        for b in bins:
            pack.add_bin(*b)

        # Start packing
        pack.pack()
        
        # Full rectangle list
        all_rects = pack.rect_list()

        # Pallets with dimensions
        all_pals = [sorted([p[3], p[4]]) for p in all_rects]

        # Count number of 80 x 120 
        p_812, p_1012 = all_pals.count(pal_812), all_pals.count(pal_1012)
        print("{:,}/{:,} Pallets 80 x 120 (cm) | {:,}/{:,} Pallets 100 x 120 (cm)".format(p_812, n_812, p_1012, n_1012))
        
        return all_rects, all_pals

    def containerize(self, rectangles, bins, rot):
        from rectpack import newPacker

        packer = newPacker(rotation=rot)

        # Add the rectangles to packing queue
        for i, r in enumerate(rectangles):
            packer.add_rect(*r, i)

        # Add the bins where the rectangles will be placed
        for b in bins:
            packer.add_bin(*b)

        # Start packing
        packer.pack()

        # number of rectangles packed into first bin
        nrect = len(packer.rect_list())

        #output = []
        #for r in packer.rect_list():
            #rect = packer[0][i]
            #output.append(r)
            #x = rect.x
            #y = rect.y
            #w = rect.width 
            #h = rect.height
            #rid = rect.rid
        return packer.rect_list()

    def left_image(self):
        n = int(self.input_image_name.split(".")[0])%2
        return n

    def down_image(self):
        n = int(self.input_image_name.split(".")[0]) < 72
        return n

    def irregular_containerize(self, objects, bins, params):
        item_list = []
        for k, obj in enumerate(objects):
            mask = obj[3]
            cnt, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imwrite("/tmp/k.png", mask)
            #print(len(cnt),cnt[0].shape)
            ps = []
            for p in cnt[0].reshape(-1,2):
                ps.append(Point(p[0]*1000,p[1]*1000))
            item_list.append(Item(ps))
        package = nest(item_list, Box(bins[0]*1000, bins[1]*1000))
        print(package)
        self.sw.write_packgroup(package)
        self.sw.save()




    def apply2(self, image, maskLabels):

        # first, lets remove all original objects
        # combine all masks
        joint_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        #print("Original masks: ",len(maskLabels))
        for j, (mask,label) in enumerate(maskLabels):
            joint_mask += mask

        # create a copy of the joint mask before removing all instances
        # in original image. Then dialte a little because some pixels still remain
        tmp = copy.deepcopy(joint_mask)
        kernel = np.ones((5, 5), 'uint8')
        dilate_img = cv2.dilate(tmp, kernel, iterations=1)
        image = np.where(np.dstack((tmp,tmp,tmp)),
                                    0,image)
        #cv2.imwrite("/home/fer/repos/CLoDSA/random_instances/mask_" + str(j)+".png", joint_mask)

        # get min,max coordinates of combined mask to find to box region to fill
        # up with new instances
        idx = np.where(joint_mask>0)
        h_min = np.min(idx[0])
        h_max = np.max(idx[0])
        w_min = np.min(idx[1])
        w_max = np.max(idx[1])

        # nah... better hard code the region to fill up with images
        if self.down_image():
            if self.left_image():
                h_min = 200
                h_max = 1000
                w_min = 240
                w_max = 980
            else: 
                h_min = 200
                h_max = 1000
                w_min = 510
                w_max = 1200
        else:
            if self.left_image():
                h_min = 200
                h_max = 1080
                w_min = 0
                w_max = 1100
            else: 
                h_min = 200
                h_max = 1080
                w_min = 510
                w_max = 1440        

        # read random image patches
        rgb_img_files = os.listdir(self.parameters["random_images_dir"])
        rgb_img_files = [f for f in rgb_img_files if ((".jpg" in f) or (".png" in f))]
        rgb_img_files = [f for f in rgb_img_files if not("5_"in f)]

        newMaskLabels = []

        # Type 1 augmentation: place random objects in random positions
        if (0.2 > 0.5): 

            total_area = (h_max - h_min) * (w_max - w_min)
            objects = []
            rectangles  = []
            occupied_area = 0
            factor = 1.0
            while occupied_area < int(factor*total_area):

                rgb_img_path = random.sample(rgb_img_files,1)[0]
                rgb_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"], rgb_img_path), cv2.IMREAD_COLOR)
                depth_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"].replace("color","color"), rgb_img_path), cv2.IMREAD_UNCHANGED)
                mask_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"].replace("color","mask"), rgb_img_path), cv2.IMREAD_GRAYSCALE)

                occupied_area += rgb_img_patch.shape[0]*rgb_img_patch.shape[1]
                if occupied_area < int(factor*total_area):
                    objects.append((rgb_img_path, rgb_img_patch, 
                        depth_img_patch, mask_img_patch))
                    rectangles.append((rgb_img_patch.shape[1],rgb_img_patch.shape[0]))
                else:
                    break
            print("Area Ratio: ",occupied_area,"/",total_area,occupied_area/total_area)
            print("Total Objects to Fill: ", len(rectangles))

            bins = [(w_max - w_min, h_max - h_min)]
            rects =  self.containerize(rectangles, bins, True)
            print("Total Objects Filled: ",len(rects))
            
            for rect in rects:
                bid, bx, by, w, h, rid = rect#.x, rect.y,rect.width,rect.height,rect.rid
                path, rgb_p, depth_p, mask_p = objects[rid]

                # if box was rotated during packing, rotate source patch and mask.. should
                # include depth at some point
                if (h,w) != (rgb_p.shape[0],rgb_p.shape[1]):
                    rgb_p = cv2.rotate(rgb_p, cv2.ROTATE_90_CLOCKWISE)
                    mask_p = cv2.rotate(mask_p, cv2.ROTATE_90_CLOCKWISE)
                    assert (h,w) == (rgb_p.shape[0],rgb_p.shape[1])

                #  create new mask
                newmask = np.zeros_like(joint_mask, np.uint8)
                mask_p_h, mask_p_w = mask_p.shape
                newmask[h_max - by - mask_p_h:h_max - by, w_min + bx:w_min + bx + mask_p_w] = mask_p

                # extract category
                #print("Path: ",path)
                category = int(path[0])

                # modify origina image
                newimgpatch = np.zeros_like(image)

                newimgpatch[h_max - by - mask_p_h:h_max - by, w_min + bx:w_min + bx + mask_p_w,:] = rgb_p
                image = np.where(np.dstack((newmask,newmask,newmask)), newimgpatch, image)

                newMaskLabels.append((newmask, category))

        else: # Type 2 augmentation: chose n type of random object and place them in discrete regions
            n = random.randint(2,5)

            objects = []
            h_sum = 0
            for i in range(n):
                rgb_img_path = random.sample(rgb_img_files,1)[0]
                rgb_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"], rgb_img_path), cv2.IMREAD_COLOR)
                depth_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"].replace("color","color"), rgb_img_path), cv2.IMREAD_UNCHANGED)
                mask_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"].replace("color","mask"), rgb_img_path), cv2.IMREAD_GRAYSCALE)

                # randomize size, rotation
                if (random.random() > 0.5):
                    rgb_img_patch = cv2.rotate(rgb_img_patch, cv2.ROTATE_90_CLOCKWISE)
                    mask_img_patch = cv2.rotate(mask_img_patch, cv2.ROTATE_90_CLOCKWISE)
                    factor = random.random() * 0.1 + 0.7
                    rgb_img_patch = cv2.resize(rgb_img_patch, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
                    mask_img_patch = cv2.resize(mask_img_patch, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

                h_sum += rgb_img_patch.shape[0]
                rect = (rgb_img_patch.shape[1],rgb_img_patch.shape[0])
                objects.append((rgb_img_path,rgb_img_patch,mask_img_patch,rect))

            # verify all objects can be fit into available area
            total_height = h_max - h_min
            if h_sum < total_height:
                pass
            else:
                objects.pop(-1)
                #print("Le hicimos POP")

            # divide regions:
            remaining_h = total_height
            outputs = []
            for k, object_ in enumerate(objects):
                #print("Forma: ",object_[1].shape)
                h_ = object_[1].shape[0]
                n = (w_max - w_min)//object_[1].shape[1]
                if k==(len(objects)-1):
                    h_ = remaining_h
                    n *= int(h_//object_[1].shape[0])
                bins = [(w_max - w_min, h_)]
                
                rects = [object_[3]]*n
                #print("Rects: ",rects)
                remaining_h -= h_
                out =  self.containerize(rects, bins, False)
                outputs.append((out,bins))

            assert len(outputs) == len(objects)

            remaining_h = h_max
            idx = 0
            for k, output in enumerate(outputs):
                #idx = random.randint(0,len(outputs)-1)
                rects, bins = output
                #print(" INDEX: ",k)
                for rect in rects:
                    bid, bx, by, w, h, rid = rect#.x, rect.y,rect.width,rect.height,rect.rid
                    path, rgb_p, mask_p, or_rect = objects[k]
                    #print("Or:",rgb_p.shape, "Out:",w,h)

                    # extract category
                    #print("Path: ",path)
                    category = int(path[0])

                    # if box was rotated during packing, rotate source patch and mask.. should
                    # include depth at some point
                    if (h,w) != (rgb_p.shape[0],rgb_p.shape[1]):
                        rgb_p = cv2.rotate(rgb_p, cv2.ROTATE_90_CLOCKWISE)
                        mask_p = cv2.rotate(mask_p, cv2.ROTATE_90_CLOCKWISE)
                        assert (h,w) == (rgb_p.shape[0],rgb_p.shape[1])
                        assert (h,w) == (mask_p.shape[0],mask_p.shape[1])

                    #  create new mask
                    newmask = np.zeros_like(joint_mask, np.uint8)
                    mask_p_h, mask_p_w = mask_p.shape
                    m = newmask[remaining_h - by - mask_p_h:remaining_h - by, w_min + bx:w_min + bx + mask_p_w]
                    #print("M: ",m.shape)
                    newmask[remaining_h - by - mask_p_h:remaining_h - by, w_min + bx:w_min + bx + mask_p_w] = mask_p
                    # modify origina image
                    newimgpatch = np.zeros_like(image)

                    newimgpatch[remaining_h - by - mask_p_h:remaining_h - by, w_min + bx:w_min + bx + mask_p_w,:] = rgb_p
                    image = np.where(np.dstack((newmask,newmask,newmask)), newimgpatch, image)

                    newMaskLabels.append((newmask, category))
                remaining_h -= mask_p_h

        """
        else:
            total_area = (h_max - h_min) * (w_max - w_min)
            objects = []
            occupied_area = 0
            factor = 1.0
            while occupied_area < int(factor*total_area):

                rgb_img_path = random.sample(rgb_img_files,1)[0]
                rgb_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"], rgb_img_path), cv2.IMREAD_COLOR)
                depth_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"].replace("color","color"), rgb_img_path), cv2.IMREAD_UNCHANGED)
                mask_img_patch = cv2.imread(os.path.join(self.parameters["random_images_dir"].replace("color","mask"), rgb_img_path), cv2.IMREAD_GRAYSCALE)

                occupied_area += rgb_img_patch.shape[0]*rgb_img_patch.shape[1]
                if occupied_area < int(factor*total_area):
                    objects.append((rgb_img_path, rgb_img_patch, 
                        depth_img_patch, mask_img_patch))
                else:
                    break
            print("Area Ratio: ",occupied_area,"/",total_area,occupied_area/total_area)

            bins = [w_max - w_min, h_max - h_min]
            rects =  self.irregular_containerize(objects, bins, True)
            #print("Total Objects Filled: ",len(rects))
            
            for rect in []:
                bid, bx, by, w, h, rid = rect#.x, rect.y,rect.width,rect.height,rect.rid
                path, rgb_p, depth_p, mask_p = objects[rid]

                # if box was rotated during packing, rotate source patch and mask.. should
                # include depth at some point
                if (h,w) != (rgb_p.shape[0],rgb_p.shape[1]):
                    rgb_p = cv2.rotate(rgb_p, cv2.ROTATE_90_CLOCKWISE)
                    mask_p = cv2.rotate(mask_p, cv2.ROTATE_90_CLOCKWISE)
                    assert (h,w) == (rgb_p.shape[0],rgb_p.shape[1])

                #  create new mask
                newmask = np.zeros_like(joint_mask, np.uint8)
                mask_p_h, mask_p_w = mask_p.shape
                newmask[h_max - by - mask_p_h:h_max - by, w_min + bx:w_min + bx + mask_p_w] = mask_p

                # extract category
                #print("Path: ",path)
                category = int(path[0])

                # modify origina image
                newimgpatch = np.zeros_like(image)

                newimgpatch[h_max - by - mask_p_h:h_max - by, w_min + bx:w_min + bx + mask_p_w,:] = rgb_p
                image = np.where(np.dstack((newmask,newmask,newmask)), newimgpatch, image)

                newMaskLabels.append((newmask, category))
        """
        return [image, newMaskLabels]
        














