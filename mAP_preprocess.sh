#!/bin/bash



if [ -d "./mAP-master/input/ground-truth" ]; then
    rm -r ./mAP-master/input/ground-truth
fi

if [ -d "./mAP-master/input/detection-results" ]; then
    rm -r ./mAP-master/input/detection-results
fi

cp -r ./Data/test_annot/. ./mAP-master/input/ground-truth

python3 ./convert_gt_xml.py

mkdir ./mAP-master/input/detection-results
python3 ./mAP_detection_data.py
