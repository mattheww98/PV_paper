#!/bin/bash
read -p "Enter property: " prop
if [ ! -d $prop ]
then
mkdir $prop
fi
cd $prop
scp myriad:~/Scratch/abs_spectra/PV_paper/properties/${prop}_unscaled/yu_test/val_preds.csv unscaled/val_preds.csv
scp myriad:~/Scratch/abs_spectra/PV_paper/properties/${prop}_unscaled/zero_inflated/yu_test/val_preds.csv ZI/val_preds.csv
cd ..

