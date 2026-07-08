"""
detect.py
Runs a frozen pretrained YOLO detector over a folder of images and dumps
the detections (boxes, classes, confidences) to a JSON file.

This is run ONCE per image set so that training/evaluation do not have to
re-run the detector every epoch.

The detector is used purely as a frozen proposal engine — no training or
fine-tuning happens here, which matches the project's zero-retraining premise.

Output format (detections.json):
{
    "<image_id>": [
        {"class_id": 47, "class_name": "cup",    "bbox": [x1,y1,x2,y2], "conf": 0.83},
        {"class_id": 73, "class_name": "laptop", "bbox": [x1,y1,x2,y2], "conf": 0.91},
        ...
    ],
    ...
}

bbox is in absolute pixel [x1, y1, x2, y2] format (same convention the rest
of the pipeline uses after bbox_xywh_to_xyxy).
"""

import os
import json
import argparse
from glob import glob


def image_path_to_id(path):
    """
    COCO val2017/train2017 filenames are the zero-padded image id, e.g.
    000000397133.jpg -> 397133. We strip the extension and parse the int so
    the id lines up with COCO's GT 'image_id' field.

    If your filenames are NOT COCO-style integers, this falls back to the raw
    filename stem (a string), which still works as long as detection ids and
    GT ids are matched the same way.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def run_detector(image_dir, out_path, weights, conf, iou, classes_keep=None, limit=None):
    from ultralytics import YOLO  # imported here so the rest of the repo doesn't need it

    model = YOLO(weights)               # frozen pretrained weights, COCO-trained
    names = model.names                 # {class_id: class_name}

    image_paths = sorted(
        glob(os.path.join(image_dir, "*.jpg"))
        + glob(os.path.join(image_dir, "*.png"))
        + glob(os.path.join(image_dir, "*.jpeg"))
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    print(f"Found {len(image_paths)} images in {image_dir}")
    print(f"Running {weights} at conf={conf}, NMS iou={iou} ...")

    detections = {}
    for i, path in enumerate(image_paths):
        # conf + NMS are applied INSIDE ultralytics; we never implement suppression.
        results = model.predict(path, conf=conf, iou=iou, verbose=False)
        r = results[0]

        img_id = image_path_to_id(path)
        dets = []
        for box in r.boxes:
            cls_id = int(box.cls.item())
            if classes_keep is not None and cls_id not in classes_keep:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "class_id":   cls_id,
                "class_name": names[cls_id],
                "bbox":       [x1, y1, x2, y2],
                "conf":       float(box.conf.item()),
            })
        detections[str(img_id)] = dets

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(image_paths)} images done")

    with open(out_path, "w") as f:
        json.dump(detections, f)

    n_boxes = sum(len(v) for v in detections.values())
    print(f"\nSaved {n_boxes} detections over {len(detections)} images -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True,
                    help="Folder of images, e.g. ./coco/val2017")
    ap.add_argument("--out", default="detections.json",
                    help="Where to write detections JSON")
    ap.add_argument("--weights", default="yolov8m.pt",
                    help="Ultralytics weights name or path (auto-downloads if missing)")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="Detector confidence threshold (low = more boxes, more noise)")
    ap.add_argument("--iou", type=float, default=0.7,
                    help="NMS IoU threshold")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of images (for quick tests)")
    args = ap.parse_args()

    run_detector(args.image_dir, args.out, args.weights, args.conf, args.iou,
                 classes_keep=None, limit=args.limit)


if __name__ == "__main__":
    main()
