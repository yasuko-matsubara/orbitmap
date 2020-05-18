#!/bin/sh
##############################################################
# Author:    Yasuko Matsubara 
# Email:     yasuko@sanken.osaka-u.ac.jp
# URL:       https://www.dm.sanken.osaka-u.ac.jp/~yasuko/
# Date:      2020-01-01
#------------------------------------------------------------#
# Copyright (C) 2020 Yasuko Matsubara & Yasushi Sakurai
# OrbitMap is freely available for non-commercial purposes
##############################################################
echo ""
echo "============================"
echo " OrbitMap (main_om.sh)   "
echo "============================"
echo ""

mode=$1
outdir=$2
seqfnORG=$3
lstep=$4
est_st=$5
est_n=$6
cast_st=$7
cast_n=$8
mscale=$9

echo "------------------"
echo "mode:" $mode
echo "outdir:" $outdir
echo "sequence fn (org):" $seqfnORG
echo "lstep:"  $lstep
echo "mscale:" $mscale
echo "est_st:" $est_st
echo "est_n:"  $est_n
echo "cast_st:" $cast_st
echo "cast_n:" $cast_n
echo "------------------"


if [ "$#" -ne 9 ]; then
echo " Usage: cmd [mode] [outdir] [seqfn] [lstep] [est_st] [est_n] [cast_st] [cast_n] [mscale]"
exit 1
fi

# run-mode: e.g., 1-1-1, 1-0-0, 1-1-2, ... 
IFS='-'; set -- $mode 
mode_est=$1; mode_scan=$2; mode_viz=$3


# create dirs
outdirC=$outdir"est/"
outdirU=$outdir"scan/"

# input sequence
seqfn=$outdir"seq"
mkdir $outdir
#python3 main_norm.py -i $seqfnORG -o $seqfn #-w $wsize -s $swd  
python3 main_norm.py -i $seqfnORG -o $seqfn -s $lstep #-w $wsize -s $swd  
# (wsize: samling window size, swd: smooth window size) - default: wsize=1, swd=1


#--------------------#
# estimate (run_est)
if [ $mode_est = 1 ]; then
mkdir $outdirC
python3 main_om.py -i $seqfn -l $lstep -p $mscale -o $outdirC -t $est_st -n $est_n #> $outdirC"log.txt"
fi
#--------------------#
# scan (run_scan)
if [ $mode_scan = 1 ]; then
mkdir $outdirU
python3 main_om.py -i $seqfn -l $lstep -p $mscale -o $outdirU -t $cast_st -n $cast_n -m $outdirC #> $outdirU"logi.txt"
fi
#--------------------#
# viz
if [ $mode_viz = 1 ]; then
python3 main_om_viz.py -o $outdirC #-a 1
python3 main_om_viz.py -o $outdirU #-a 1
fi
#--------------------#
# viz
if [ $mode_viz = 2 ]; then
python3 main_om_viz.py -o $outdirC -a 1
python3 main_om_viz.py -o $outdirU -a 1
fi
#--------------------#



