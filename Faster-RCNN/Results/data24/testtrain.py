import json
# import random
# import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.evaluation import COCOEvaluator

with open(r'E:\PyProjects\Faster-RCNN\Database\data32\train\data_test.txt', 'r') as in_file:
    dataset_dicts = json.load(in_file)
# with open(r'E:\PyProjects\Faster-RCNN\Database\data32\val\data_test.txt', 'r') as in_file:
#     testset_dicts = json.load(in_file)


DatasetCatalog.register("mol_train" , lambda d="train": dataset_dicts)
MetadataCatalog.get("mol_train").set(thing_classes=["L", "R"])
# DatasetCatalog.register("mol_val" , lambda d="val": testset_dicts)
# MetadataCatalog.get("mol_val").set(thing_classes=["L", "R"])
mol_metadata = MetadataCatalog.get("mol_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("mol_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "E:\PyProjects\Faster-RCNN\Results\data24\output\model_final.pth"  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = (500, 1000)
cfg.INPUT.MIN_SIZE_TRAIN = 600
cfg.INPUT.MAX_SIZE_TRAIN = 1000
cfg.SOLVER.MOMENTUM = 0.8
cfg.SOLVER.MAX_ITER = 1500   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.WEIGHT_DECAY = 0.0005
# cfg.MODEL.PIXEL_MEAN = [125, 125, 125]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.6, 0.8]
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# cfg.INPUT.CROP["ENABLED"] = True
# cfg.INPUT.CROP.SIZE = [0.8, 0.8]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
cfg.MODEL.RPN.BOUNDARY_THRESH = 20
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 60000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 20000
cfg.MODEL.RPN.NMS_THRESH = 0.8
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
cfg.MODEL.RPN.LOSS_WEIGHT = 1.0

# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
# cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
# cfg.TEST.EVAL_PERIOD = 0
# cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 30000
# cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 6000
# evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
# trainer.test(cfg,evaluators=evaluator)
trainer.train()

#
# #
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
#
# evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "mol_val")
# inference_on_dataset(trainer.model, val_loader, evaluator)