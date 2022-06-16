# Implementation for training 
# python pills.py --mode train

# Implementation for prediction 
# python pills.py --mode predict -e pills/net/
# python pills.py --mode predict -e pills/pills_hand/

# # import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import json
import cv2
import os
import pdb  # mz pdb.set_trace() #mz

# initialize the dataset path, images path, and annotations file path
DATASET_PATH = os.path.abspath("pills")
IMAGES_PATH = os.path.sep.join([DATASET_PATH, "images"])
ANNOT_PATH = os.path.sep.join([DATASET_PATH, "via_region_data.json"])
# initialize the amount of data to use for training
TRAINING_SPLIT = 0.75
# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))
idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
trainIdxs = idxs[:i]
valIdxs = idxs[i:]

# initialize the class names dictionary
CLASS_NAMES = {1: "round"}
# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "mask_rcnn_coco.h5"
# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = "pills_logs"


class PillsConfig(Config):
    # give the configuration a recognizable name
    NAME = "pills"
    # set the number of GPUs to use training along with the number of
    # images per GPU (which may have to be tuned depending on how
    # much memory your GPU has)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # set the number of steps per training epoch
    STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
    # number of classes (+1 for the background)
    NUM_CLASSES = len(CLASS_NAMES) + 1


class PillsInferenceConfig(PillsConfig):

    # set the number of GPUs and images per GPU (which may be
    # different values than the ones used for training)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # set the minimum detection confidence (used to prune out false
    # positive detections)
    DETECTION_MIN_CONFIDENCE = 0.9


class PillsDataset(utils.Dataset):
    def __init__(self, imagePaths, annotPath, classNames, width=1024):
        # call the parent constructor
        super().__init__(self)
        # store the image paths and class names along with the width
        # we'll resize images to
        self.imagePaths = imagePaths
        self.classNames = classNames
        self.width = width
        # load the annotation data
        self.annots = self.load_annotation_data(annotPath)

    def load_annotation_data(self, annotPath):
        # load the contents of the annotation JSON file (created
        # using the VIA tool) and initialize the annotations
        # dictionary
        annotations = json.loads(open(annotPath).read())
        annots = {}

        # loop over the file ID and annotations themselves (values)
        for (fileID, data) in sorted(annotations.items()):
            # store the data in the dictionary using the filename as
            # the key
            annots[data["filename"]] = data

        # return the annotations dictionary
        return annots

    def load_pills(self, idxs):
        # loop over all class names and add each to the 'pills'
        # dataset
        for (classID, label) in self.classNames.items():
            self.add_class("pills", classID, label)

        # loop over the image path indexes
        for i in idxs:
            # extract the image filename to serve as the unique
            # image ID
            imagePath = self.imagePaths[i]
            filename = imagePath.split(os.path.sep)[-1]
            # load the image and resize it so we can determine its
            # width and height (unfortunately VIA does not embed
            # this information directly in the annotation file)
            image = cv2.imread(imagePath)
            (origH, origW) = image.shape[:2]
            image = imutils.resize(image, width=self.width)
            (newH, newW) = image.shape[:2]

            # add the image to the dataset
            self.add_image("pills", image_id=filename,
                           width=newW, height=newH,
                           orig_width=origW, orig_height=origH,
                           path=imagePath)

    def load_image(self, imageID):
        # grab the image path, load it, and convert it from BGR to
        # RGB color channel ordering
        p = self.image_info[imageID]["path"]
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize the image, preserving the aspect ratio
        image = imutils.resize(image, width=self.width)

        # return the image
        return image

    def load_mask(self, imageID):
        # grab the image info and then grab the annotation data for
        # the current image based on the unique ID
        info = self.image_info[imageID]
        annot = self.annots[info["id"]]
        # allocate memory for our [height, width, num_instances] array
        # where each "instance" effectively has its own "channel"
        masks = np.zeros((info["height"], info["width"],
                          len(annot["regions"])), dtype="uint8")
        # loop over each of the annotated regions
        for (i, region) in enumerate(annot["regions"]):
            # allocate memory for the region mask
            regionMask = np.zeros(masks.shape[:2], dtype="uint8")
            # grab the shape and region attributes
            sa = region["shape_attributes"]
            ra = region["region_attributes"]
            # scale the center (x, y)-coordinates and radius of the
            # circle based on the dimensions of the resized image
            ratio = info["width"] / float(info["orig_width"])
            cX = int(sa["cx"] * ratio)
            cY = int(sa["cy"] * ratio)
            r = int(sa["r"] * ratio)
            # draw a circular mask for the region and store the mask
            # in the masks array
            cv2.circle(regionMask, (cX, cY), r, 1, -1)
            masks[:, :, i] = regionMask
            # pdb.set_trace() #mz

        # return the mask array and class IDs, which for this dataset
        # is all 1's
        return (masks.astype("bool"), np.ones((masks.shape[-1],),
                                              dtype="int32"))


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True,
                    help="either 'train', 'predict', or 'investigate'")
    ap.add_argument("-w", "--weights",
                    help="optional path to pretrained weights")
    ap.add_argument("-i", "--image",
                    help="optional path to input image to segment")
    ap.add_argument("-e", "--example",
                    help="optional path to input image to segment")
    args = vars(ap.parse_args())

# check to see if we are training the Mask R-CNN
    if args["mode"] == "train":
        # load the training dataset
        trainDataset = PillsDataset(IMAGE_PATHS, ANNOT_PATH,
                                    CLASS_NAMES)
        trainDataset.load_pills(trainIdxs)
        trainDataset.prepare()
        # load the validation dataset
        valDataset = PillsDataset(IMAGE_PATHS, ANNOT_PATH,
                                  CLASS_NAMES)
        valDataset.load_pills(valIdxs)
        valDataset.prepare()

        # initialize the training configuration
        config = PillsConfig()
        config.display()
        
        # initialize the model and load the COCO weights so we can
        # perform fine-tuning
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=LOGS_AND_MODEL_DIR)
        model.load_weights(COCO_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])

        # train *just* the layer heads
        model.train(trainDataset, valDataset, epochs=10,
                    layers="heads", learning_rate=config.LEARNING_RATE)
        
                    
        # unfreeze the body of the network and train *all* layers
        pdb.set_trace() #mz
        model.train(trainDataset, valDataset, epochs=20,
                    layers="all", learning_rate=config.LEARNING_RATE / 10)

        # check to see if we are predicting using a trained Mask R-CNN
    elif args["mode"] == "predict":
        # initialize the inference configuration
        config = PillsInferenceConfig()
        # initialize the Mask R-CNN model for inference
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=LOGS_AND_MODEL_DIR)
        # load our trained Mask R-CNN
        weights = args["weights"] if args["weights"] else model.find_last()
        model.load_weights(weights, by_name=True)

        # MZ: loading examples
        #pdb.set_trace()  # mz
        EXAMPLE_PATHS = sorted(list(paths.list_images(args["example"])))

        for p in EXAMPLE_PATHS:
            image = cv2.imread(p)

            # load the input image, convert it from BGR to RGB channel
            # ordering, and resize the image
            #image = cv2.imread(args["image"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = imutils.resize(image, width=1024)

            # perform a forward pass of the network to obtain the results
            r = model.detect([image], verbose=1)[0]
            # loop over of the detected object's bounding boxes and
            # masks, drawing each as we go along
            for i in range(0, r["rois"].shape[0]):
                mask = r["masks"][:, :, i]
                image = visualize.apply_mask(image, mask,
                                             (1.0, 0.0, 0.0), alpha=0.5)
                image = visualize.draw_box(image, r["rois"][i],
                                           (1.0, 0.0, 0.0))

            # convert the image back to BGR so we can use OpenCV's
            # drawing functions
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # pdb.set_trace() #mz

            # loop over the predicted scores and class labels
            for i in range(0, len(r["scores"])):
                # extract the bounding box information, class ID, label,
                # and predicted probability from the results
                (startY, startX, endY, end) = r["rois"][i]
                classID = r["class_ids"][i]
                label = CLASS_NAMES[classID]
                score = r["scores"][i]
                # draw the class label and score on the image
                text = "{}: {:.4f}".format(label, score)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # resize the image so it more easily fits on our screen
            image = imutils.resize(image, width=512)

            # save image
            result_path='pills/pills_hand_labeled/'+p.split('/')[-1]
            #pdb.set_trace()  # mz
            cv2.imwrite(result_path, image)

            # show the output image
            #cv2.imshow("Output", image)
            #cv2.waitKey(0)

