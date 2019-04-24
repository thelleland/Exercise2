#!/bin/bash

mkdir ./Data
cd ./Data
mkdir train_imgs
mkdir train_annot
mkdir valid_imgs
mkdir valid_annot
mkdir test_imgs
mkdir test_annot


cd ..

python3 image-simulator-master/imagesim.py -n 2000 -b source_data/Backgrounds -c source_data/Classes -o Data/train_imgs
echo "Finished simlulating training images."

python3 image-simulator-master/imagesim.py -n 500 -b source_data/Backgrounds -c source_data/Classes -o Data/valid_imgs
echo "Finished simulating validation images."

python3 image-simulator-master/imagesim.py -n 500 -b source_data/Backgrounds -c source_data/Classes -o Data/test_imgs
echo "Finished simulating test images."

cd Data

python3 ../make_voc_data.py -i ./train_imgs/ -o ./train_annot/
echo "Made VOC data for taining set."

python3 ../make_voc_data.py -i ./valid_imgs/ -o ./valid_annot/
echo "Made VOC data for validation set."

python3 ../make_voc_data.py -i ./test_imgs/ -o ./test_annot/
echo "Made VOC data for test set."

echo "Deleting annotation files"

for f in train_imgs/*.txt
do
    rm $f
done

for f in valid_imgs/*.txt
do
    rm $f
done

for f in test_imgs/*.txt
do
    rm $f
done

echo "Finished deleting annotations."
