"""
Copyright (C) 2023 Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import json
import math
import torch
import mrcfile
import warnings
import argparse
import numpy as np
from torch import nn
from sklearn.metrics import f1_score
from sklearn.neighbors import BallTree

from model import NestedUNet
from interp3d import interp3d
from utils import parse_map, write_map, cleanpdb, read_atoms, split_map_into_overlapped_boxes, get_map_from_overlapped_boxes, get_SS_type_atoms, assign_predicted_SS_to_atoms, del_out_of_contour, Get_path_according_to_type


# threads
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def get_args():
    parser = argparse.ArgumentParser(description="Obtain the parameters required for prediction.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mapin", "-mi", type=str, required=True, help="Input EM density map file")
    parser.add_argument("--contour", "-cl", type=float, required=True, help="The contour level of input EM density map file")
    parser.add_argument("--type", "-t", type=str, default="EXP", help="Select the type of input EM density map file")
    parser.add_argument("--fold", "-f", type=str, default=None, help="Select which fold of the model to use")
    parser.add_argument("--gpu", "-g", type=str, default="0", help="ID(s) of GPU devices to use")
    parser.add_argument("--stride", "-s", type=int, default=12, help="The step of the sliding window for cutting the input map into overlapping boxes. Its value should be an integer within [12,48]")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size for predict")
    parser.add_argument("--evaluate", action='store_true', default=False, help="choose whether to evaluate")
    parser.add_argument("--pdb", "-p", type=str, default=None, help="Input the path of pdb file")
    parser.add_argument("--use_cpu", action='store_true', default=False, help="Whether to use CPU instead of GPU")
    parser.add_argument("--model_dir", "-md", type=str, required=True, help="Directory name of the state dictionary files for parameters of the trained model")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mapin_path = args.mapin
    contour = args.contour
    map_type = args.type
    fold = args.fold
    gpu_id = args.gpu
    stride = args.stride
    batch_size = args.batch_size
    evaluate = args.evaluate
    pdb_path = args.pdb
    use_cpu = args.use_cpu
    model_dir = args.model_dir
    name = mapin_path.split("/")[-1].split(".")[0]

    #################################################################################
    ################################ set parameters #################################
    #################################################################################
    # set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu_id is None:
        n_gpus = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("# Running on CPU", flush=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"# Running on {n_gpus} GPU(s)", flush=True)
        else:
            print("# CUDA is not available", flush=True)
            sys.exit()

    # set data parameters
    ignorestart = False
    apix = 1.0
    box_size = 48
    assert box_size % 2 == 0
    stride = stride
    n_classes = 4


    # check whether to evaluate
    if evaluate:
        if pdb_path is None:
            print("# pdb file input is missing", flush=True)
            sys.exit()
        mapout_path = "./Pred_Eva"
        if not os.path.exists(mapout_path):
            print(f"# mkdir {mapout_path}", flush=True)
            os.mkdir(mapout_path)
        else:
            print(f"# {mapout_path} already exists", flush=True)
    else:
        mapout_path = "./Pred"
        if not os.path.exists(mapout_path):
            print(f"# mkdir {mapout_path}", flush=True)
            os.mkdir(mapout_path)
        else:
            print(f"# {mapout_path} already exists", flush=True)


    # initialize the model
    model = NestedUNet(n_classes = 4,
                       init_channels = 32,
                       in_channels = 1)

    if gpu_id is not None:
        torch.cuda.empty_cache()
        model = model.cuda()
        if n_gpus > 1:
            model = nn.DataParallel(model)

   
    #################################################################################
    ########################### get path according to type ##########################
    #################################################################################
    model_path, mapout_all_path, mapout_cx_path, npy_all_path, repdb_path, result_path = Get_path_according_to_type(map_type, mapout_path, name, fold, model_dir)
    if pdb_path != "none":
        cleanpdb(pdb_path, repdb_path)


    #################################################################################
    ################################### load mapin ##################################
    ################################################################################# 
    print(f"# Load mapin from {mapin_path}", flush=True)
    map, origin, nxyz, voxel_size = parse_map(mapin_path, ignorestart=False, apix=apix)
    try:
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
    except AssertionError:
        origin_shift =  ( np.round(origin / voxel_size) - origin / voxel_size ) * voxel_size
        map, origin, nxyz, voxel_size = parse_map(in_map, ignorestart=False, apix=apix, origin_shift=origin_shift)
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    print(f"# Map dimensions = {nxyz}", flush=True)

    below_threshold_mask = np.where(map <= contour, n_classes, -1)
    print(f"# Get the mask from mapin with contour {contour}", flush=True)

    boxes, ncx, ncy, ncz = split_map_into_overlapped_boxes(map, box_size, stride, dtype=np.float32, padding=0.0)
    n_boxes = len(boxes)
    print(f"# Split map into {n_boxes} overlapped boxes", flush=True)

    del_indexs = []
    max_density = np.percentile(map[map > 0], 99.999)
    boxes_norm = np.zeros((n_boxes, box_size, box_size, box_size), dtype=np.float32)
    for i, box in enumerate(boxes):
        if box.max() <= 0.0:
            del_indexs.append(i)
            continue
        boxes_norm[i] = box
    boxes_norm = boxes_norm.clip(min=0.0, max=max_density) / max_density

    boxes = np.delete(boxes_norm, del_indexs, axis=0)
    keep_indexs = np.delete( np.arange(n_boxes, dtype=np.int64), del_indexs, axis=0 )
    n_boxes_keep = len(boxes)
    print(f"# Get {n_boxes_keep} positive boxes", flush=True)

    X = torch.autograd.Variable(torch.FloatTensor(boxes), requires_grad=False).view(-1, 1, box_size, box_size, box_size)
    del boxes
    n_data = len(X)
    n_batches = math.ceil(n_data / batch_size)


    #################################################################################
    ################################### prediction ##################################
    #################################################################################
    print(f"\n-----------------------------------------------------------------\n# Start prediction", flush=True)

    if os.path.exists(mapout_all_path):
        print(f"{mapout_all_path} already exists.", flush=True)
    else:
        if map_type == "EXP" and fold == "none":
            assert len(model_path) == 5
            for index, model_file in enumerate(model_path):
                print(f"# Load model from {model_file}", flush=True)
                if not use_cpu:
                    model_state_dict = torch.load(model_file)
                else:
                    model_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
                model.load_state_dict(model_state_dict)
                model.eval()
                boxes_pred_keep = np.zeros( (n_boxes_keep, n_classes, box_size, box_size, box_size), dtype=np.float32 )
                with torch.no_grad():
                    for i in range(n_batches):
                        X_batch = X[i * batch_size : (i+1) * batch_size]
                        if gpu_id is not None:
                            X_batch = X_batch.cuda()
        
                        Y_pred = model(X_batch)
                        Y_pred = Y_pred.cpu().detach().numpy()
                        boxes_pred_keep[i * batch_size : (i+1) * batch_size] = Y_pred
        
                boxes_pred = np.zeros( (n_boxes, n_classes, box_size, box_size, box_size), dtype=np.float32 )
                boxes_pred[keep_indexs] = boxes_pred_keep
                del boxes_pred_keep
        
                map_pred = get_map_from_overlapped_boxes(boxes_pred, n_classes, ncx, ncy, ncz, box_size, stride, nxyz)
                np.save(npy_all_path[index], map_pred)
                print(f"# The prediction of fold{index+1} is finished", flush=True)
                del map_pred
    
            map_pred = 0
            for i in range(5):
                map_pred += np.load(npy_all_path[i])
            map_pred = np.argmax(map_pred, axis=0)
            map_pred = np.where(map_pred < below_threshold_mask, below_threshold_mask, map_pred)
            write_map(mapout_all_path, map_pred.astype(np.float32), voxel_size, origin=origin)
            for i in range(n_classes):
                write_map(mapout_cx_path[i], np.where(map_pred==i, 1, 0).astype(np.float32), voxel_size, origin=origin)
            print("# End prediction", flush=True)
            
        else:
            print(f"# Load model from {model_path}", flush=True)
            if not use_cpu:
                model_state_dict = torch.load(model_path)
            else:
                model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state_dict)

            model.eval()
            boxes_pred_keep = np.zeros( (n_boxes_keep, n_classes, box_size, box_size, box_size), dtype=np.float32 )
            with torch.no_grad():
                for i in range(n_batches):
                    X_batch = X[i * batch_size : (i+1) * batch_size]
                    if gpu_id is not None:
                        X_batch = X_batch.cuda()
    
                    Y_pred = model(X_batch)
                    Y_pred = Y_pred.cpu().detach().numpy()
                    boxes_pred_keep[i * batch_size : (i+1) * batch_size] = Y_pred
    
            boxes_pred = np.zeros( (n_boxes, n_classes, box_size, box_size, box_size), dtype=np.float32 )
            boxes_pred[keep_indexs] = boxes_pred_keep
            del boxes_pred_keep
    
            map_pred = get_map_from_overlapped_boxes(boxes_pred, n_classes, ncx, ncy, ncz, box_size, stride, nxyz)
            map_pred = np.argmax(map_pred, axis=0)
            map_pred = np.where(map_pred < below_threshold_mask, below_threshold_mask, map_pred)
            write_map(mapout_all_path, map_pred.astype(np.float32), voxel_size, origin=origin)
            for i in range(n_classes):
                write_map(mapout_cx_path[i], np.where(map_pred==i, 1, 0).astype(np.float32), voxel_size, origin=origin)
            print("# End prediction", flush=True)


    #################################################################################
    ################################### evaluation ##################################
    #################################################################################
    if evaluate:
        print(f"\n-----------------------------------------------------------------\n# Start evaluation", flush=True)
        print(f"The predictions are saved in {result_path}", flush=True)
        coords, secstr = get_SS_type_atoms(repdb_path)
        map_pred, origin, nxyz, _ = parse_map(mapout_all_path, False, apix=apix)
 
        map_pred = map_pred.astype(np.int64)
        predict_grids = np.argwhere(map_pred < n_classes)
        predict_grids = predict_grids[:,::-1] # zyx -> xyz
        predict_grids = predict_grids * voxel_size + origin
        predictions = map_pred[map_pred < n_classes]
        secstr_pred = assign_predicted_SS_to_atoms(predict_grids, predictions, coords, full=n_classes)
        assert(len(secstr) == len(secstr_pred))

        # voxel level evaluation
        with open(result_path, "w") as f:
            f.write(f"voxel level evaluation\n")
        tree = BallTree(coords, leaf_size=2)
        distances, indices = tree.query(predict_grids, k=1)
    
        del_indices = []
        ground_trueths = np.zeros(len(predictions), dtype=np.int64)
        for i, index in enumerate(indices):
            distance = distances[i][0]
            if distance > 3.0:
                del_indices.append(i)
                continue
            ground_trueths[i] = secstr[index[0]]
    
        ground_trueths = np.delete(ground_trueths, del_indices)
        predictions = np.delete(predictions, del_indices)
    
        n_H = np.sum(np.where(ground_trueths==0,1,0))
        n_S = np.sum(np.where(ground_trueths==1,1,0))
        n_C = np.sum(np.where(ground_trueths==2,1,0))
        n_N = np.sum(np.where(ground_trueths==3,1,0))

        with open(result_path, "a") as f:
            f.write(f"n_H = {n_H}\nn_S = {n_S}\nn_C = {n_C}\nn_N = {n_N}\n")
    
        f1_voxel_classes = []
        num_f1 = [n_H, n_S, n_C, n_N]
        for i in range(n_classes):
            if num_f1[i] == 0:
                f1_voxel_classes.append(-1)
            else:
                f1_voxel_classes.append(f1_score(ground_trueths, predictions, labels=[i], average='weighted'))
    
        f1_voxel_weighted = (f1_voxel_classes[0]*n_H + f1_voxel_classes[1]*n_S + f1_voxel_classes[2]*n_C + f1_voxel_classes[3]*n_N) / float(n_H+n_S+n_C+n_N)
    
        ### residue level evaluation
        with open(result_path, "a") as f:
            f.write(f"\nresidue level evaluation\n")
        del_indices = np.where(secstr_pred >= n_classes) # unassigned atoms
        with open(result_path, "a") as f:
            f.write(f"{len(del_indices[0])} unassigned atom(s) is(are) removed from evaluation\n")
        coords = np.delete(coords, del_indices, axis=0)
        secstr = np.delete(secstr, del_indices)
        secstr_pred = np.delete(secstr_pred, del_indices)
        assert(len(coords) == len(secstr) == len(secstr_pred))
    
        del_indices = del_out_of_contour(mapin_path, contour, coords, secstr) # delete atoms out of contour
        with open(result_path, "a") as f:
            f.write(f"{len(del_indices)} out-of-contour atom(s) is(are) removed from evaluation\n")
        secstr = np.delete(secstr, del_indices)
        secstr_pred = np.delete(secstr_pred, del_indices)
    
        n_H = np.sum(np.where(secstr==0,1,0))
        n_S = np.sum(np.where(secstr==1,1,0))
        n_C = np.sum(np.where(secstr==2,1,0))
        n_N = np.sum(np.where(secstr==3,1,0))

        with open(result_path, "a") as f:
            f.write(f"n_H = {n_H}\nn_S = {n_S}\nn_C = {n_C}\nn_N = {n_N}\n")
    
        q4_accuracy_classes = []
        num_q4 = [n_H, n_S, n_C, n_N]
    
        for i in range(n_classes):
            if num_q4[i] == 0:
                q4_accuracy_classes.append(-1)
            else:
                q4_accuracy_classes.append(np.sum(np.where(secstr == i, 1, 0) * np.where(secstr_pred == i, 1, 0) == 1) / np.sum(secstr == i))
        q4_accuracy_weighted = (q4_accuracy_classes[0]*n_H + q4_accuracy_classes[1]*n_S + q4_accuracy_classes[2]*n_C + q4_accuracy_classes[3]*n_N) / float(n_H+n_S+n_C+n_N)
    
        ### write results
        with open(result_path, "a") as f:
            title = ["f1_weight", "f1_Helix", "f1_Sheet", "f1_Coil", "f1_NA",
                   "Q4_weight", "Q4_Helix", "Q4_Sheet", "Q4_Coil", "Q4_NA" ]
            f.write(f"\n{title[0]:12s}{title[1]:12s}{title[2]:12s}{title[3]:12s}{title[4]:12s}{title[5]:12s}{title[6]:12s}{title[7]:12s}{title[8]:12s}{title[9]:12s}\n")
            #################################################################
            f.write(f"{f1_voxel_weighted:<.8f}  ")
            for i in range(n_classes):
                if num_f1[i] == 0:
                    f.write(f"{'nan':<10s}  ")
                else:
                    f.write(f"{f1_voxel_classes[i]:<.8f}  ")
            f.write(f"{q4_accuracy_weighted:<.8f}  ")
            for i in range(n_classes):
                if num_q4[i] == 0:
                    f.write(f"{'nan':<10s}  ")
                else:
                    f.write(f"{q4_accuracy_classes[i]:<.8f}  ")


if __name__ == "__main__":
    main()
