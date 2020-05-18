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
import sys
import numpy as np
import tool as tl
import time
MAXSIZE=10
MINSIZE=1
FONTSIZE=64
#------------------------------------------------#
DBG=tl.NO #YES
def write_dot(CGraph, outfn):
    if(DBG): tl.msg("Write graph %s.dot"%(outfn))
    #--------------------------------#
    f=open("%s.dot"%(outfn),"w")
    f.write("digraph G {\n")
    #f.write("node [fontname=Arial, fontsize=12, color=white, style=filled];")
    f.write("bgcolor=\"#ffffff00\"\n") # RGBA (with alpha)
    f.write("ratio=2.0\n")
    #f.write("splines=true\n")
    #f.write("overlap=scale\n")
    f.write("graph [size=\"4.0,8.0\", center=true];") #viz quality notfinA
    #f.write("graph [size=\"4,4\",page=\"4.0,4.0\", center=true];") #viz quality notfinA
    #f.write("graph [dpi = 80];") #viz quality notfinA
    f.write("node [color=white, style=filled, fontsize=%d];"%(FONTSIZE))
    f.write("edge [fontsize=%d];"%(FONTSIZE))

    #f.write("size =\"4,4\";\n")
    for rcd in CGraph['MD']:
        if(not 'col' in rcd): rcd['col']='lightgray'
        if(not 'cnt' in rcd):rcd['cnt']=1.0
        if(DBG): lbl='#%d (%d)'%(rcd['id'], rcd['cnt'])
        else: lbl='#%d'%(rcd['id'])
        size=min(MINSIZE+np.log2(rcd['cnt']+1),MAXSIZE)
        f.write("%s [label=\" %s \", shape=circle, width=%f, fillcolor=%s];\n"%(rcd['id'], lbl, size, rcd['col']))
    for rcd in CGraph['MS']:
        #if(not 'col' in rcd): rcd['col']='dimgrey'
        if(not 'col' in rcd): rcd['col']='black'
        if(not 'shape' in rcd): rcd['shape']='solid'
        if(not 'cnt' in rcd):rcd['cnt']=1.0
        if(DBG): lbl='%d (%d)'%(rcd['tcnt'], rcd['cnt'])
        else: lbl='%d'%(rcd['cnt'])
        #lbl=''
        size=min(MINSIZE+(rcd['cnt']),MAXSIZE)
        f.write("%s -> %s [label=\" %s \", color=%s, style=%s, penwidth=%f];\n"%(rcd['fr'], rcd['to'], lbl, rcd['col'], rcd['shape'], size))
    f.write("}\n")
    f.close()
    #--------------------------------#
    # convert dot -> pdf
    tl.os.system("dot -T pdf %s.dot -o %s.pdf"%(outfn, outfn))
    tl.os.system("dot -T png %s.dot -o %s.png"%(outfn, outfn))
    #tl.os.system("dot -T eps %s.dot -o %s.eps"%(outfn, outfn))

def _sample0():
    rcds_MD=[];
    rcds_MD.append({'id':1, 'cnt':2})
    rcds_MD.append({'id':2, 'cnt':2})
    rcds_MD.append({'id':3, 'cnt':2, 'col':'powderblue'})
    rcds_MD.append({'id':4, 'cnt':2, 'col':'tomato'})
    rcds_MS=[];
    rcds_MS.append({'fr':1, 'to':2, 'cnt':2}) 
    rcds_MS.append({'fr':2, 'to':3, 'cnt':2}) 
    rcds_MS.append({'fr':3, 'to':4, 'cnt':2}) 
    rcds_MS.append({'fr':4, 'to':1, 'cnt':1}) 
    return (rcds_MD, rcds_MS)    


def _sample1():
    rcds_MD=[];
    #rcds_MD.append({'id':1, 'lbl':5, 'col':'tomato', 'shape':'bold','size':5.0})
    rcds_MD.append({'id':1, 'cnt':2, 'col':'tomato'})
    rcds_MD.append({'id':2, 'cnt':4})
    rcds_MD.append({'id':3, 'cnt':8})
    rcds_MD.append({'id':4, 'cnt':16})
    rcds_MD.append({'id':5, 'cnt':32})
    rcds_MD.append({'id':6, 'cnt':64})
    rcds_MS=[];
    rcds_MS.append({'fr':1, 'to':2, 'cnt':1}) #, 'col':'red'})
    rcds_MS.append({'fr':1, 'to':1, 'cnt':2}) 
    rcds_MS.append({'fr':2, 'to':4, 'cnt':4}) 
    rcds_MS.append({'fr':3, 'to':4, 'cnt':8}) 
    rcds_MS.append({'fr':4, 'to':5, 'cnt':16}) 
    rcds_MS.append({'fr':5, 'to':6, 'cnt':32}) 
    rcds_MS.append({'fr':6, 'to':1, 'cnt':64}) 
    return (rcds_MD, rcds_MS)    

#---------------#
#     main      #
#---------------#
if __name__ == "__main__":

    fn="../output/tmp/graph_sample"

    tl.msg("graphviz")
    outdir='../output/tmp/graph_sample'
    (rcds_MD, rcds_MS)=_sample0()
    #(rcds_MD, rcds_MS)=_sample1()    
    write_dot({'MD':rcds_MD, 'MS':rcds_MS}, outdir) 
    print(rcds_MD)
    print(rcds_MS)


