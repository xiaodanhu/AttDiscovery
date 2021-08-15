import torch, torchvision
print('torch version:', torch.__version__, torch.cuda.is_available())
print('torch_vision version:',  torchvision.__version__)
# assert torch.__version__.startswith("1.9")

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_backbone, build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.poolers import ROIPooler

from glob import glob

def get_region_proposals(inputs, raw_images, k=10, model_zoo_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device ='cuda:3'):
    
    def get_image_patches(raw_image, proposals):
        patches = []
        for box in proposals:
            x1,y1,x2,y2 = box.to("cpu")
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            patch = raw_image[y1:y2,x1:x2]
            patches.append(patch)
        return patches

    cfg = get_cfg()
    # set cuda device
    cfg.MODEL.DEVICE = device

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_name)

    '''set up model'''
    # predictor = DefaultPredictor(cfg)
    model = build_model(cfg) 
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) 
    model.eval()

    '''inference'''
    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN
        
        top_k_proposals = [instance.proposal_boxes[:k] for instance in proposals]
        # print(f'top {k} proposals:\n', top_k_proposals)
        for i in range(len(inputs)): 
            inputs[i]['proposals'] = top_k_proposals[i]
            inputs[i]['raw_image'] = raw_images[i]
            inputs[i]['raw_image_patches'] = get_image_patches(raw_images[i],top_k_proposals[i])
        
        '''get pred and features from region proposals'''
        # features_ = [features[f] for f in model.roi_heads.box_in_features]
        # box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        # box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        # predictions = model.roi_heads.box_predictor(box_features)
        # pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals) #  First, it applies box deltas to read just the proposal boxes. Then it computes Non-Maximum Suppression to remove non-overlapping boxes (while also applying other hyper-settings such as score threshold). Finally, it ranks top-k boxes according to their scores. 
        # pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # # output boxes, masks, scores, etc
        # pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        # # features of the proposed boxes
        # feats = box_features[pred_inds]
    return inputs, cfg

def get_detectron_inputs(img_path_list):
    inputs = []
    raw_images = []
    for im_path in img_path_list:
        im = cv2.imread(im_path)
        raw_images.append(im)
        height, width = im.shape[:2] # desired height and width, can be different from im.shape
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        print(image.size())
        inputs.append({"image": image, "height": height, "width": width})
    return inputs, raw_images

if __name__ == '__main__':
    output_dir = '/shared/nas/data/m1/wangz3/multimedia_attribute/AttDiscovery/test_output'
    input_image_dir = '/shared/nas/data/m1/wangz3/multimedia_attribute/AttDiscovery/test_images'
    
    img_path_list = sorted(glob(os.path.join(input_image_dir,'*')))
    '''load inputs'''
    inputs, raw_images = get_detectron_inputs(img_path_list)

    '''get region proposal'''
    inputs, cfg = get_region_proposals(inputs, raw_images)

    '''visualize'''
    for i in range(len(inputs)):
        im = raw_images[i]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2) # bgr->rgb
        top_proposals = list(inputs[i]['proposals'])
        for box_idx in range(len(top_proposals)):
            box = top_proposals[box_idx]
            box_coord = box.to("cpu")
            out = v.draw_box(box_coord, alpha=0.8, edge_color='g', line_style='-')
            out = v.draw_text(f'{box_idx}',(box_coord[0],box_coord[1]))
        # out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))

        cv2.imwrite(os.path.join(output_dir, f'out_box_{i}.png'),out.get_image()[:, :, ::-1])

        # visualize patches
        for j in range(len(inputs[i]['raw_image_patches'])):
            x1,y1,x2,y2 = top_proposals[j].to("cpu")
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y1)
            cv2.imwrite(os.path.join(output_dir, f'patches/img_{i}_patch_{j}_{x1}-{y1}-{x2}-{y2}.png'),inputs[i]['raw_image_patches'][j])

