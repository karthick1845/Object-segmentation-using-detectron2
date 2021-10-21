from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2 import model_zoo


import cv2
import numpy


class Detector:
    def __init__(self,model_type = "OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        #Load model and pretrained model
        if model_type == "OD":
            #Object detection

            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        elif model_type == "IS": # Instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        elif model_type == "PS" :   #Panoptic segmentation

            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")




        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = 'cpu'

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self,imagePath):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":

            prediction = self.predictor(image)

            viz = Visualizer(image[:,:,::-1],metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            instance_mode= ColorMode.IMAGE)
            output = viz.draw_instance_predictions(prediction["instances"].to("cpu"))

        else :
            prediction,segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(prediction.to("cpu"),segmentInfo)
        cv2.imshow("Result",output.get_image()[:,:,::-1])
        cv2.waitKey(0)


    def onVideo(self,videopath):
        cap = cv2.VideoCapture(videopath)

        if (cap.isOpened()==False):
            print("Error opening the file..")
            return
        (sucess, image ) = cap.read()

        while sucess :
            if self.model_type != "PS":

                prediction = self.predictor(image)

                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                 instance_mode=ColorMode.IMAGE)
                output = viz.draw_instance_predictions(prediction["instances"].to("cpu"))

            else:
                prediction, segmentInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(prediction.to("cpu"), segmentInfo)
            cv2.imshow("Result", output.get_image()[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (sucess,image) = cap.read()


