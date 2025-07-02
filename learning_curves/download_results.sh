#!/bin/bash
read -p "Enter property: " prop
cd $prop
for i in {10,100,1000}
do
for j in {0..3}
do
index=$((1000*$j+$i))
mkdir $index
scp myriad:~/Scratch/abs_spectra/PV_paper/learning_curve/${prop}/${index}/test/val_preds.csv ${index}
done
done

