"""Synthetic test: GT labels (distance & multi modes), global-threshold baseline, scaler."""
import json, tempfile, os
import numpy as np
import importlib
import data_pipeline as dp

W = H = 1000
categories = [{"id":47,"name":"cup"},{"id":73,"name":"laptop"},
              {"id":1,"name":"person"},{"id":3,"name":"car"},{"id":49,"name":"knife"}]
def xywh(cx,cy,w=40,h=40): return [cx-w/2,cy-h/2,w,h]
def xyxy(cx,cy,w=40,h=40): return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

images = [{"id":i,"width":W,"height":H} for i in (1,2,3,4)]
annotations = [
    {"image_id":1,"category_id":47,"bbox":xywh(500,500)},   # real cup
    {"image_id":1,"category_id":73,"bbox":xywh(520,520)},   # real laptop (close+overlap)
    {"image_id":2,"category_id":47,"bbox":xywh(100,100)},
    {"image_id":2,"category_id":73,"bbox":xywh(900,900)},   # far
    {"image_id":3,"category_id":73,"bbox":xywh(500,500)},   # laptop only (cup will be spurious)
    {"image_id":4,"category_id":1, "bbox":xywh(500,500)},   # person
    {"image_id":4,"category_id":3, "bbox":xywh(620,620)},   # car at cdist 0.120
]
coco = {"categories":categories,"images":images,"annotations":annotations}
detections = {
    "1":[{"class_id":47,"class_name":"cup","bbox":xyxy(500,500),"conf":0.90},
         {"class_id":73,"class_name":"laptop","bbox":xyxy(520,520),"conf":0.88}],
    "2":[{"class_id":47,"class_name":"cup","bbox":xyxy(100,100),"conf":0.80},
         {"class_id":73,"class_name":"laptop","bbox":xyxy(900,900),"conf":0.85}],
    "3":[{"class_id":47,"class_name":"cup","bbox":xyxy(505,505),"conf":0.18},   # spurious
         {"class_id":73,"class_name":"laptop","bbox":xyxy(500,500),"conf":0.92}],
    "4":[{"class_id":1,"class_name":"person","bbox":xyxy(500,500),"conf":0.95},
         {"class_id":3,"class_name":"car","bbox":xyxy(620,620),"conf":0.95}],
}

def run(mode):
    dp.HAZARD_MODE = mode
    with tempfile.TemporaryDirectory() as d:
        ap, dpath = os.path.join(d,"a.json"), os.path.join(d,"d.json")
        json.dump(coco, open(ap,"w")); json.dump(detections, open(dpath,"w"))
        feats, labels, ids, base = dp.extract_pairs(ap, dpath)
    return {int(i):(int(l),int(b)) for f,l,i,b in zip(feats,labels,ids,base)}, feats

print("baseline global threshold =", round(dp.GLOBAL_BASELINE_THRESHOLD,3))

for mode in ("distance","multi"):
    m, feats = run(mode)
    print(f"\n[{mode}]  (label, baseline) per image:", m)
    # img1: real close+overlap -> hazard in both modes; baseline (cdist .02<.133) fires
    assert m[1]==(1,1), m[1]
    # img2: far -> safe; baseline 0
    assert m[2]==(0,0), m[2]
    # img3: spurious cup -> true 0; baseline fires (close) -> the confidence case
    assert m[3]==(0,1), m[3]
    # img4: person-car cdist .120; per-class thr .10 -> NOT close -> label 0;
    #        global baseline thr .133 -> .120<.133 -> baseline FIRES (1). The per-class gap.
    assert m[4]==(0,1), m[4]

# scaler sanity
_, feats = run("multi")
mean, std = dp.fit_scaler(feats)
Z = dp.apply_scaler(feats, mean, std)
assert Z.shape == feats.shape and dp.FEATURE_DIM == feats.shape[1] == 15
print("\nALL ASSERTIONS PASSED — feature dim", feats.shape[1])
print("img3 = confidence case (spurious low-conf cup); img4 = per-class threshold gap")
print("Both are cases the single-threshold baseline gets WRONG and the MLP can learn.")
