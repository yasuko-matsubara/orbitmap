#!/bin/sh
##############################################################
# Author:    Yasuko Matsubara 
# Email:     yasuko@sanken.osaka-u.ac.jp
# URL:       https://www.dm.sanken.osaka-u.ac.jp/~yasuko/
# Date:      2020-04-24
#------------------------------------------------------------#
# Copyright (C) 2020 Yasuko Matsubara & Yasushi Sakurai
# OrbitMap is freely available for non-commercial purposes
##############################################################



OUTDIR="./_out/tmp/"

#-------------------#                      
# input parameter settings
# outdir: 	output directory
# seqfnORG: 	original sequence file name
# mscale: 	multi-scale-modeling (default: 1)
# est_st:	start estimation (preprocessing, find opt rho, etc.)
# est_n:   	estimation length (preprocessing, find opt rho, etc.)
# cast_st: 	start forecast (real-time modeling and forecasting)
# cast_n:  	forecast length (real-time modeling and forecasting)
# lstep:	lsteps-ahead forecasting
#-------------------#                      


#--- synthetic data
seqfnORG="_dat/synthetic_X3"
outdir="./_out/tmp/syn/"
lstep=10; est_st=0; est_n=500; cast_st=$est_n; cast_n=$est_n; 
lstep=10  
#-------------------#                      


#--- google trends data (13weeks/3months-ahead forecasting)
#seqfnORG="../../DATA/google/seq/outdoor.dat"
#outdir="./_out/tmp/google_t_o/"
#lstep=13; est_st=0; est_n=260; cast_st=$est_n; cast_n=$est_n;

#--- google trends data (13weeks/3months-ahead forecasting)
#seqfnORG="../../DATA/google/seq/sports.dat"
#outdir="./_out/tmp/google_t_s/"
#lstep=13; est_st=0; est_n=260; cast_st=$est_n; cast_n=$est_n;




#-------------------#                      
mkdir $outdir
echo "------------------"
echo "OrbitMap START: "
echo "label:" $label
echo "outdir:" $outdir
echo "------------------"
#-------------------#                      



#-------------------#                      


mscale=1     
mode="1-1-1"
#mode="0-0-1"
#mode="1-1-1" # est + scan + viz
sh main_om.sh $mode $outdir $seqfnORG $lstep $est_st $est_n $cast_st $cast_n $mscale  #> $outdir"log.txt"

echo "=================="
echo " OrbitMap END   "
echo "=================="





