#!/bin/bash
#Copyright (C) 2023 Hong cao, Jiahua He, Tao Li, Hao Li, Sheng-You Huang and Huazhong University of Science and Technology


# Users need to properly set the following variables after installation
#######################################################################
    EMInfo_home=""
    activate=""
    EMInfo_env=""
    #EMInfo_home="/home/hcao/data_gu02/papers/paper1/package/20240326/EMInfo_use"
    #activate="/home/hcao/data_gu02/anaconda3/bin/activate"
    #EMInfo_env="pth2"
#######################################################################


source $activate $EMInfo_env

if [ $# -lt 2 ];then
	echo ""
	echo "EMInfo - Improvement of Cryo-EM maps by deep learning
Huang Lab @ HUST, http://huanglab.phys.hust.edu.cn/EMInfo/"
	echo ""
	echo "USAGE: `basename $0` in_map.mrc contour [options]"
	echo ""
	echo "Descriptions:"
	echo "    in_map.mrc  : Input EM density map in MRC2014 format"
	echo ""
	echo "    contour     : The contour level of input EM density map file"
	echo ""
	echo "    -f          : Select which fold of the EXP model to use"
	echo "                  default: 'None'"
    echo ""
	echo "    -g          : ID(s) of GPU devices to use.  e.g. '0' for GPU #0, and '2,3,6' for GPUs #2, #3, and #6"
	echo "                  default: '0'"
    echo ""
	echo "    -s          : The step of the sliding window for cutting the input map into overlapping boxes. Its value should be an integer within [12, 48]"
	echo "                  default: '12'"
    echo ""
	echo "    -b          : Number of boxes input into EMReady in one batch. Users can adjust 'batch_size' according to the VRAM of their GPU devices."
	echo "                  default: '16'"
    echo ""
	echo "    --evaluate  : Evaluate the F1-score and Q4-accuracy of annotated map"
	echo ""
	echo "    -p          : Input the path of pdb file for evaluate"
	echo "                  default: 'None'"
    echo ""
	echo "    --use_cpu   : Run EMReady on CPU instead of GPU"
	echo ""

        exit 1
fi

in_map=$1
contour=$2
type_model="EXP"
fold_exp=none
gpu_id="0"
stride=12
batch_size=16
evaluate=""
pdb_path=none
use_cpu=""
model_state_dict_dir=$EMInfo_home"/model_state_dicts"

while [ $# -gt 2 ];do
    case $3 in
    -t)
        shift
        type_model=$3;;
    -f)
        shift
        fold_exp=$3;;
    -g)
        shift
        gpu_id=$3;;
    -s)
        shift
        stride=$3;;
    -b)
        shift
        batch_size=$3;;
    --evaluate)
        evaluate="--evaluate";;
    -p)
        shift
        pdb_path=$3;;
    --use_cpu)
        use_cpu="--use_cpu";;
    *)
    	echo " ERROR: wrong command argument \"$3\" !!"
    	echo " Type \"$0\" for help !!"
        exit 2;;
    esac
    shift
done

python ${EMInfo_home}/eminfo.py -mi $in_map -cl $contour -t $type_model -f $fold_exp -g $gpu_id -s $stride -b $batch_size $evaluate -p $pdb_path $use_cpu -md $model_state_dict_dir
