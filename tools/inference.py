# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
#
# Shengli's changes

import argparse
import functools
import json
import operator
import os
import os.path as op
from tkinter import W
from typing import Optional
import uuid

import blobfile as bf
import cv2
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from PIL import Image
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.config import sg_cfg
from scene_graph_benchmark.scene_parser import SceneParser
from tools.demo.visual_utils import draw_bb, draw_rel
from tqdm import tqdm
from mpi4py import MPI


def split_by_rank(data):
    data = list(data)
    count = len(data)

    n_samples_per_rank = max(count // MPI.COMM.Get_size(), 1)

    start_idx = n_samples_per_rank * MPI.COMM.Get_rank()
    end_idx = n_samples_per_rank * (MPI.COMM.Get_rank() + 1)

    return data[start_idx:end_idx]

def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_image(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and
    # width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)

    with torch.no_grad():
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device("cpu"))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]

    if isinstance(model, SceneParser):
        prediction_pred = prediction.prediction_pairs
        relations = prediction_pred.get_field("idx_pairs").tolist()
        relation_scores = prediction_pred.get_field("scores").tolist()
        predicates = prediction_pred.get_field("labels").tolist()
        prediction = prediction.predictions

    prediction = prediction.resize((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()

    if isinstance(model, SceneParser):
        rt_box_list = []
        if "attr_scores" in prediction.extra_fields:
            attr_scores = prediction.get_field("attr_scores")
            attr_labels = prediction.get_field("attr_labels")
            rt_box_list = [
                {
                    "rect": box,
                    "class": cls,
                    "conf": score,
                    "attr": attr[attr_conf > 0.01].tolist(),
                    "attr_conf": attr_conf[attr_conf > 0.01].tolist(),
                }
                for box, cls, score, attr, attr_conf in zip(
                    boxes, classes, scores, attr_labels, attr_scores
                )
            ]
        else:
            rt_box_list = [
                {"rect": box, "class": cls, "conf": score}
                for box, cls, score in zip(boxes, classes, scores)
            ]
        rt_relation_list = [
            {"subj_id": relation[0], "obj_id": relation[1], "class": predicate + 1, "conf": score}
            for relation, predicate, score in zip(relations, predicates, relation_scores)
        ]
        return {"objects": rt_box_list, "relations": rt_relation_list}
    else:
        if "attr_scores" in prediction.extra_fields:
            attr_scores = prediction.get_field("attr_scores")
            attr_labels = prediction.get_field("attr_labels")
            return [
                {
                    "rect": box,
                    "class": cls,
                    "conf": score,
                    "attr": attr[attr_conf > 0.01].tolist(),
                    "attr_conf": attr_conf[attr_conf > 0.01].tolist(),
                }
                for box, cls, score, attr, attr_conf in zip(
                    boxes, classes, scores, attr_labels, attr_scores
                )
            ]

        return [
            {"rect": box, "class": cls, "conf": score}
            for box, cls, score in zip(boxes, classes, scores)
        ]

def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        "white",
        "black",
        "blue",
        "green",
        "red",
        "brown",
        "yellow",
        "small",
        "large",
        "silver",
        "wooden",
        "wood",
        "orange",
        "gray",
        "grey",
        "metal",
        "pink",
        "tall",
        "long",
        "dark",
        "purple",
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {"blonde": "blond"}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]


def run(config_file: str, image_directory: str, working_directory: Optional[str] = None, output_dir: Optional[str] = None, opts: Optional[dict] =None):
    ###
    # Load configurations
    ###
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.DEVICE = f"cuda:{MPI.COMM.Get_rank()}"
    cfg.freeze()

    ###
    # Load Images
    ###
    img_files = bf.listdir(image_directory)
    allowed_extensions = {"jpg", "jpeg", "png"}
    img_files = []
    for f in bf.listdir(image_directory):
        ext = f.split(".")[-1].lower()
        if ext not in allowed_extensions:
            continue
        full_path = bf.join(image_directory, f)
        img_files.append(full_path)

    if len(img_files) == 0:
        raise RuntimeError(f"No images found in {image_directory}")

    ###
    # Create working directory
    ###
    if working_directory is None:
        working_directory = f"sgb_working_dir/{uuid.uuid4()}"

    if not bf.exists(working_directory):
        bf.makedirs(working_directory)

    ###
    # Output Directory
    ###
    if output_dir is None:
        output_dir = cfg.OUTPUT_DIR

    if not bf.exists(output_dir):
        bf.makedirs(output_dir)

    ###
    # Load Model
    ###
    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)

    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    ###
    # Load datasets and label map
    ###
    dataset_labelmap = {}
    with bf.BlobFile(cfg.DATASETS.LABELMAP_FILE, "r") as f:
        dataset_allmap = json.load(f)
        dataset_labelmap = {int(val): key for key, val in dataset_allmap["label_to_idx"].items()}

    visual_labelmap = None
    ###
    # Load in the correct relations, if germane
    ###
    if cfg.MODEL.ATTRIBUTE_ON:
        dataset_attr_labelmap = {
            int(val): key for key, val in dataset_allmap["attribute_to_idx"].items()
        }

    if cfg.MODEL.RELATION_ON:
        dataset_relation_labelmap = {
            int(val): key for key, val in dataset_allmap["predicate_to_idx"].items()
        }

    transforms = build_transforms(cfg, is_train=False)

    image_summaries = []
    for img_file in tqdm(split_by_rank(img_files)):
        ###
        # Copy images to our local
        ###
        image_name = bf.basename(img_file)
        local_image_path = os.path.join(working_directory, image_name)
        bf.copy(img_file, local_image_path)
        cv2_img = cv2.imread(local_image_path)

        detections = detect_objects_on_single_image(model, transforms, cv2_img)

        if isinstance(model, SceneParser):
            relationship_detections = detections["relations"]
            detections = detections["objects"]


        for obj in detections:
            obj["class"] = dataset_labelmap[obj["class"]]
        if visual_labelmap is not None:
            detections = [d for d in detections if d["class"] in visual_labelmap]
        if cfg.MODEL.ATTRIBUTE_ON:
            for obj in detections:
                obj["attr"], obj["attr_conf"] = postprocess_attr(
                    dataset_attr_labelmap, obj["attr"], obj["attr_conf"]
                )
        if cfg.MODEL.RELATION_ON:
            for rel in relationship_detections:
                rel["class"] = dataset_relation_labelmap[rel["class"]]
                subj_rect = detections[rel["subj_id"]]["rect"]
                rel["subj_center"] = [
                    (subj_rect[0] + subj_rect[2]) / 2,
                    (subj_rect[1] + subj_rect[3]) / 2,
                ]
                obj_rect = detections[rel["obj_id"]]["rect"]
                rel["obj_center"] = [
                    (obj_rect[0] + obj_rect[2]) / 2,
                    (obj_rect[1] + obj_rect[3]) / 2,
                ]

        rects = [d["rect"] for d in detections]
        scores = [d["conf"] for d in detections]
        if cfg.MODEL.ATTRIBUTE_ON:
            attribute_list = [d["attr"] for d in detections]
            attr_labels = [",".join(d["attr"]) for d in detections]
            attr_scores = [d["attr_conf"] for d in detections]
            classes = [d["class"] for d in detections]
            labels = [attr_label + " " + d["class"] for d, attr_label in zip(detections, attr_labels)]
        else:
            labels = [d["class"] for d in detections]

        draw_bb(cv2_img, rects, labels, scores)

        if cfg.MODEL.RELATION_ON:
            rel_subj_centers = [r["subj_center"] for r in relationship_detections]
            rel_obj_centers = [r["obj_center"] for r in relationship_detections]
            rel_scores = [r["conf"] for r in relationship_detections]
            rel_labels = [r["class"] for r in relationship_detections]
            draw_rel(cv2_img, rel_subj_centers, rel_obj_centers, rel_labels, rel_scores)

        save_file_fn = bf.basename(img_file) + ".detect" + op.splitext(img_file)[-1]
        save_file = bf.join(output_dir, save_file_fn)

        cv2.imwrite(save_file, cv2_img)
        print("saved img results to: {}".format(save_file))

        relationship_detections_output = None
        if cfg.MODEL.RELATION_ON:
            relationship_detections_output = []
            rel_subj_centers = [r["subj_center"] for r in relationship_detections]
            rel_obj_centers = [r["obj_center"] for r in relationship_detections]
            rel_scores = [r["conf"] for r in relationship_detections]
            rel_labels = [r["class"] for r in relationship_detections]

        attr_detections_output = None
        if cfg.MODEL.ATTRIBUTE_ON:
            attr_detections_output = []
            for label, clazz, score, rect, a_label, a_score in zip(labels, classes, scores, rects, attribute_list, attr_scores):
                attributes = []
                for attribute_label, attribute_score in zip(a_label, a_score):
                    attribute_obj = {
                        "label": attribute_label,
                        "score": attribute_score,
                    }
                    attributes.append(attribute_obj)

                object_detection_output = {
                    "label": label,
                    "class": clazz,
                    "score": score,
                    "attributes": attributes,
                    "bounding_box": rect
                }
                attr_detections_output.append(object_detection_output)

            json_output = op.splitext(save_file)[0] + ".json"

        full_output = {
            "image": img_file,
            "annotated_image": save_file,
            "attributes": attr_detections_output,
            "relationships": relationship_detections_output,
        }

        json_result = json.dumps(full_output)
        image_summaries.append(full_output)
        with bf.BlobFile(json_output, "w") as f:
            f.write(json_result)
            print("saved img results to: {}".format(json_output))


    ###
    # Write output on main thread
    ###
    if MPI.COMM.Get_rank() == 0:
        summary = functools.reduce(
            operator.concat, MPI.COMM.gather(image_summaries, root=0)
        )
        output = {
            "cfg": cfg,
            "metadata": {
                "args": {
                    "config_file": config_file,
                    "image_directory": image_directory,
                    "working_directory": working_directory,
                    "output_directory": output_dir,
                    "opts": opts
                },
                "cfg": cfg,
                "mpi": {
                    "size": MPI.COMM.Get_size()
                }
            },
            "images": summary
        }
        json_full_result = json.dumps(output)
        with bf.BlobFile(json_output, "w") as f:
            f.write(json_full_result)
            print("saved summary of all images to: {}".format(json_output))

def main():
    parser = argparse.ArgumentParser(description="Object-Attribute Detection")
    parser.add_argument("--config_file", metavar="FILE", help="path to config file")
    parser.add_argument("--img_dir", metavar="DIR", help="path to images")
    parser.add_argument("--output_dir", metavar="DIR", help="path to output")
    parser.add_argument("--working_directory", metavar="DIR", help="path to output")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()
    run(
        config_file=args.config_file,
        image_directory=args.img_dir,
        working_directory=args.working_directory,
        output_dir=args.output_dir,
        opts=args.opts
    )


if __name__ == "__main__":
    main()