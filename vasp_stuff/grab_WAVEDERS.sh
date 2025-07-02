#!/bin/bash
read -p "Enter material: " mat
scp uccaahw@young.rc.ucl.ac.uk:~/Scratch/test_atomate/hybrid_optics/key_optics/${mat}/optics/WAVEDER ${mat}/hse/optics/
