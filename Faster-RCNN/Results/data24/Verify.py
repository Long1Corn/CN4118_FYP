import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
import matplotlib
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode

with open(r'E:\PyProjects\Faster-RCNN\Database\data24\val\data_test.txt', 'r') as in_file:
    testset_dicts = json.load(in_file)
for dict in testset_dicts:
    for ann in dict["annotations"]:
        ann["bbox_mode"] = BoxMode.XYXY_ABS

DatasetCatalog.register("mol_val", lambda d="val": testset_dicts)
MetadataCatalog.get("mol_val").set(thing_classes=["L", "R"])
mol_metadata = MetadataCatalog.get("mol_val")


cfg = get_cfg()
# cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ("mol_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
cfg.MODEL.RPN.NMS_THRESH = 0.8
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.6, 0.8]
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = r"E:\PyProjects\Faster-RCNN\Results\data24\output\model_final.pth"
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 60000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 20000
cfg.TEST.DETECTIONS_PER_IMAGE = 500
# cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 100000
# cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 100000
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512

predictor = DefaultPredictor(cfg)


# im = cv2.imread(r"C:\Users\Laptop\Desktop\P1.jpg")
# im2 = cv2.imread(r"C:\Users\Laptop\Desktop\P2.jpg")
# outputs = predictor(im2)
#
# v = Visualizer(im,
#                metadata=mol_metadata,
#                scale=1.5,
#                instance_mode=ColorMode.SEGMENTATION
#                )
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow("1", v.get_image())
# cv2.waitKey(0)
# # #
for d in random.sample(testset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=mol_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("n", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
#
# evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "mol_val")
# trainer = DefaultTrainer(cfg)
# inference_on_dataset(trainer.model, val_loader, evaluator)
