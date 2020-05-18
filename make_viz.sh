#!/usr/bin/env python
##############################################################
# Author:    Yasuko Matsubara 
# Email:     yasuko@sanken.osaka-u.ac.jp
# URL:       https://www.dm.sanken.osaka-u.ac.jp/~yasuko/
# Date:      2020-01-01
#------------------------------------------------------------#
# Copyright (C) 2020 Yasuko Matsubara & Yasushi Sakurai
# OrbitMap is freely available for non-commercial purposes
##############################################################
# use nohup, if needed
echo ""
echo "============================"
echo " OrbitMap (make_viz.sh)   "
echo "============================"
echo ""

outdir=$1
echo "------------------"
echo "outdir:" $outdir
echo "------------------"


if [ "$#" -ne 1 ]; then
echo " Usage: cmd [outdir] "
exit 1
fi

for fn in `ls $outdir`; do
	echo $fn


# create dirs
outdirC=$outdir$fn"/est/"
outdirU=$outdir$fn"/scan/"

ls $outdirC
ls $outdirU

#--------------------#
# viz
if [ 1 = 1 ]; then
python3 main_cc_viz.py -o $outdirC
python3 main_cc_viz.py -o $outdirU
fi
#--------------------#


done
