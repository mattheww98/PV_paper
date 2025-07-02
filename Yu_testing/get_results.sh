#!/bin/bash
read -p "Enter property: " prop
if [ ! -d $prop ]
then
mkdir $prop
fi
cd $prop
scp myriad:~/Scratch/abs_spectra/PV_paper/properties/$prop/yu_test/val_preds.csv val_preds.csv
cd ..

