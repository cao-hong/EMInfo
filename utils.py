'''
Copyright (C) 2023 Hong Cao, Jiahua He, Tao Li, Hao Li, Sheng-You Huang and Huazhong University of Science and Technology

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
'''

import os
import sys
import mrcfile
import warnings
import numpy as np
from math import floor, ceil
from Bio.PDB import PDBParser
from Bio import BiopythonWarning
from sklearn.neighbors import BallTree

from interp3d import interp3d

def parse_map(map_path, ignorestart=False, apix=None, origin_shift=None):
    ### parse_map
    mrc = mrcfile.open(map_path, mode="r")
    map = np.asarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    nxyzstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    cella = (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z)
    nxyz = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)
    mapcrs = np.asarray([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=np.int64)
    mrc.close()

    ### check orthogonal
    try:
        assert np.all(angle == 90.0)
    except AssertionError:
        print(f"# Input grid is not orthogonal. EXIT.")
        sys.exit()

    ### check mapcrs
    try:
        assert np.all(mapcrs == np.asarray([1,2,3], dtype=np.int64))
    except AssertionError:
        sort = np.asarray([0,1,2], np.int64)
        for i in range(3):
            sort[mapcrs[i] -1] = i
        nxyzstart = np.asarray(nxyzstart[i] for i in sort)
        nxyz = np.asarray([nxyz[i] for i in sort])
        map = np.transpose(map, axes = 2-sort[::-1])

    ### shift origin according to nxyzstart
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    ### check apix
    if apix is not None:
        target_voxel_size = np.asarray([apix for _ in range(3)], dtype=np.float32)
        try:
            assert np.all(voxel_size == target_voxel_size)
        except AssertionError:
            interp3d.del_mapout()
            if origin_shift is not None:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
                origin += origin_shift
            else:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, nxyz[2], nxyz[1], nxyz[0])
            map = interp3d.mapout
            nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
            voxel_size = target_voxel_size
    assert np.all(nxyz == np.asarray([map.shape[2], map.shape[1], map.shape[0]], dtype=np.int64))

    return map, origin, nxyz, voxel_size



def write_map(map_path, map, voxel_size, origin=(0.0, 0.0, 0.0), nxyzstart=(0.0, 0.0, 0.0)):
    mrc = mrcfile.new(map_path, overwrite=True)
    mrc.set_data(map)
    mrc.voxel_size = [voxel_size[i] for i in range(3)]
    (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z) = origin
    (mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart) = nxyzstart
    mrc.close()



def split_map_into_overlapped_boxes(map, box_size, stride, dtype=np.float32, padding=0.0):
    assert stride < box_size
    map_shape = np.shape(map)
    padded_map = np.full((map_shape[0]+2*box_size, map_shape[1]+2*box_size, map_shape[2]+2*box_size), padding, dtype=dtype)
    padded_map[box_size:box_size+map_shape[0], box_size:box_size+map_shape[1], box_size:box_size+map_shape[2]] = map
    boxes_list = []
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < map_shape[2] + box_size):
        next_box = padded_map[cur_x:cur_x+box_size, cur_y:cur_y+box_size, cur_z:cur_z+box_size]
        cur_x += stride
        if (cur_x + stride >= map_shape[0] + box_size):
            cur_y += stride
            cur_x = start_point
            if (cur_y + stride >= map_shape[1] + box_size):
                cur_z += stride
                cur_x, cur_y = start_point, start_point
        boxes_list.append(next_box)
    n_boxes = len(boxes_list)
    ncx, ncy, ncz = [ceil(map_shape[i] / stride) for i in range(3)]
    assert (n_boxes == ncx * ncy * ncz)
    boxes = np.asarray(boxes_list, dtype=dtype)
    return boxes, ncx, ncy, ncz


def get_map_from_overlapped_boxes(boxes, n_channels, ncx, ncy, ncz, box_size, stride, nxyz):
    map = np.zeros((n_channels,
                    (ncx - 1) * stride + box_size, \
                    (ncy - 1) * stride + box_size, \
                    (ncz - 1) * stride + box_size), dtype=np.float32)
    denominator = np.zeros((n_channels,
                            (ncx - 1) * stride + box_size, \
                            (ncy - 1) * stride + box_size, \
                            (ncz - 1) * stride + box_size), dtype=np.float32) # should clip to 1
    result = np.zeros((n_channels, nxyz[2], nxyz[1], nxyz[0]) ,dtype=np.float32)

    for channel in range(n_channels):
        i = 0
        for z_steps in range(ncz):
            for y_steps in range(ncy):
                for x_steps in range(ncx):
                    map[channel,
                        x_steps * stride : x_steps * stride + box_size,
                        y_steps * stride : y_steps * stride + box_size,
                        z_steps * stride : z_steps * stride + box_size] += boxes[i, channel]
                    denominator[channel,
                                x_steps * stride : x_steps * stride + box_size,
                                y_steps * stride : y_steps * stride + box_size,
                                z_steps * stride : z_steps * stride + box_size] += 1
                    i += 1
    for channel in range(n_channels):
        result[channel] = (map[channel] / denominator[channel].clip(min=1))[stride : nxyz[2] + stride, stride : nxyz[1] + stride, stride : nxyz[0] + stride]
    return result


def read_atoms(pdb_path, hetatm = False, ignore_H = True):
    with open(pdb_path, "r") as f:
        lines = f.readlines()
    atoms = []
    headers = ["ATOM  "]
    if hetatm:
        headers.append("HETATM")

    for line in lines:
        if line[0:6] not in headers:
            continue
        elif ignore_H and line[77:78] == "H":
            continue
        else:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        atoms.append([x, y, z])
    atoms = np.asarray(atoms, dtype = np.float32)
    return atoms


def get_zone_map(mapout_all_path, pdb_path, mapout_cx_path, n_classes, r_zone=3.0):
    mrc = mrcfile.open(mapout_all_path, mode = "r")
    map = np.asarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    nxyzstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    cella = (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z)
    nxyz = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)
    mapcrs = np.asarray([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=np.int64)
    mrc.close()

    ###########################################################################
    ################## assert that this is a preprocessed map #################
    ###########################################################################
    assert np.all(angle == 90.0)
    assert np.all(mapcrs == np.asarray([1,2,3], dtype=np.int64))
    try:
        assert np.all(nxyzstart == 0)
    except:
        origin += nxyzstart * voxel_size
    try:
        assert np.all(np.shape(map) == nxyz[::-1])
    except:
        sort = np.asarray([0,1,2], dtype=np.int64)
        for i in range(3):
            sort[mapcrs[i] - 1] = i
        nxyzstart = np.asarray(nxyzstart[i] for i in sort)
        nxyz = np.asarray(nxyz[i] for i in sort)
        map = np.transpose(map, axes = 2 - sort[::-1])
    try:
        assert np.all(voxel_size == 1.0)
    except:
        apix = 1.0
        target_voxel_size = np.asarray([apix for _ in range(3)], dtype=np.float32)
        interp3d.del_mapout()
        interp3d.cubic(data, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, nxyz[2], nxyz[1], nxyz[0])
        data = interp3d.mapout
        nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
        voxel_size = target_voxel_size

    #############################################################################
    ################################### zone ####################################
    #############################################################################
    # zone box
    atoms = read_atoms(pdb_path)
    coords = atoms - origin    # [N, 3]
    nxyz_min = np.maximum(np.floor((np.min(coords, axis=0) - r_zone) / voxel_size), 0).astype(np.int64)
    nxyz_max = np.minimum(np.ceil((np.max(coords, axis=0) + r_zone) / voxel_size), nxyz).astype(np.int64)
    assert np.all(nxyz_min < nxyz_max)
    origin += np.multiply(nxyz_min, voxel_size)
    nxyz = nxyz_max - nxyz_min    # map shape after zone box
    map = map[nxyz_min[2] : nxyz_max[2], nxyz_min[1] : nxyz_max[1], nxyz_min[0] : nxyz_max[0]]
    assert np.all(np.shape(map) == nxyz[::-1])
    print(f"map shape after zone box = {map.shape}")

    # zone map
    mask_map = np.full(nxyz[::-1], -100, dtype = np.int64)
    for atom in atoms:
        coord = atom - origin
        lower = np.floor( (coord - r_zone) / voxel_size ).astype(np.int32)
        upper = np.ceil ( (coord + r_zone) / voxel_size ).astype(np.int32)
        for x in range(lower[0], upper[0] + 1):
            for y in range(lower[1], upper[1] + 1):
                for z in range(lower[2], upper[2] + 1):
                    if 0 <= x < nxyz[0] and 0 <= y < nxyz[1] and 0 <= z < nxyz[2]:
                        if mask_map[z, y, x] == -100:
                            vector = np.array([x, y, z], dtype=np.float32) * voxel_size - coord
                            dist = np.sqrt(vector@vector)
                            if dist < r_zone:
                                mask_map[z, y, x] = 0
    zone_map = map + mask_map
    zone_map = np.where(zone_map > -1, zone_map, 4).astype(np.int64)

    for i in range(n_classes):
        write_map(mapout_cx_path[i], np.where(zone_map==i, 1, 0).astype(np.float32), voxel_size, origin=origin)

    return 1


def del_out_of_contour(file_name, contour, coords, secstr):
    ### parse MRC file
    mrc = mrcfile.open(file_name, mode='r')
    data = np.asarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    cella = (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)
    mapcrs = np.asarray([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=np.int64)
    mrc.close()

    ### assert that this is a preprocessed map
    assert(np.all(angle == 90.0))
    assert(np.all(mapcrs == np.asarray([1, 2, 3], dtype=np.int64)))
    try:
        assert(np.all(ncrsstart == 0))
    except:
        origin += ncrsstart * voxel_size
    assert(np.all(np.shape(data) == ncrs[::-1]))
    try:
        assert(np.all(voxel_size == 1.0))
    except AssertionError:
        apix = 1.0
        target_voxel_size = np.asarray([apix for _ in range(3)], dtype=np.float32)
        interp3d.del_mapout()
        interp3d.cubic(data, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, ncrs[2], ncrs[1], ncrs[0])
        data = interp3d.mapout
        ncrs = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
        voxel_size = target_voxel_size

    ### zone box
    coords_shifted = coords - origin # shift coords
    ncrs_min = np.maximum(np.floor((np.min(coords_shifted, axis=0) - 1) / voxel_size), 0).astype(np.int64)
    ncrs_max = np.minimum(np.ceil((np.max(coords_shifted, axis=0) + 1) / voxel_size), ncrs).astype(np.int64)
    assert(np.all(ncrs_min < ncrs_max)) # important...
    origin += np.multiply(ncrs_min, voxel_size)
    ncrs = np.subtract(ncrs_max, ncrs_min)
    data = data[ncrs_min[2] : ncrs_max[2], ncrs_min[1] : ncrs_max[1], ncrs_min[0] : ncrs_max[0]]
    assert(np.all(np.shape(data) == ncrs[::-1])) # impossible

    ### meshgrid
    xarray = np.arange(ncrs[0])
    yarray = np.arange(ncrs[1])
    zarray = np.arange(ncrs[2])
    m = np.ix_(xarray, yarray, zarray)
    p = np.r_ [ 2:0:-1,3:len ( m ) + 1, 0 ]
    grids = np.array(np.meshgrid( *m ), dtype="int32").transpose(p).reshape(-1,len(m))
    del m, p
    tree = BallTree(grids * voxel_size, leaf_size=2)

    coords_shifted = coords - origin # shift again
    indices = tree.query_radius(coords_shifted, r=1.0)
    del coords_shifted, coords, tree

    del_indices = []
    for i, index in enumerate(indices):
        for j in index:
            if data[grids[j, 2], grids[j, 1], grids[j, 0]] < contour:
                del_indices.append(i)
                break

    del_indices = np.unique(del_indices)

    return np.array(del_indices, dtype=np.int64)


def cleanpdb(pdb_path, repdb_path):
    repdb = []
    with open(pdb_path, "r") as f:
        for line in f.readlines():
            if line[:4] == "ATOM" or line[:3] == "TER" or line[:3] == "END":
                repdb.append(line)
    with open(repdb_path, "w") as f:
        for line in repdb:
            f.write(line)
    return 1


def get_SS_type_atoms(pdb_file, targets=None):
    ### execute STRIDE and parse output
    stride = os.popen("tmpdir=$(mktemp -d); cp " + pdb_file + " $tmpdir/tmp.pdb; cd $tmpdir; sed -i 's/HETATM/ATOM  /g' tmp.pdb; stride tmp.pdb; cd -; rm -rf $tmpdir")
    res_name = []
    chain_id = []
    res_id = []
    res_ss = []
    for line in stride.readlines():
        if line[:3] == "ASG":
            res_name.append(line[5:8])
            chain_id.append(line[9:10])
            res_id.append(line[11:15].strip())
            res_ss.append(line[24:25])
    stride.close()

    ### parse PDB file
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser()
    models = parser.get_structure('str', pdb_file)

    ### assign secondary structure type to target atoms
    coords = []
    secstr = []
    resseqs = []
    chainIDs = []
    n_atoms = 0
    for model in models:
        for chain in model.get_chains():
            atoms = chain.get_atoms()
            for atom in atoms:
                if atom.element == 'H':
                    continue
                if targets is not None:
                    if atom.get_name() not in targets:
                        continue
                chainID = chain.id
                residue = atom.get_parent()
                resname = residue.get_resname()
                hetflag, resseq, icode = residue.get_id()
                resseq = (str(resseq) + icode).strip()
                assigned = False

                ### STRIDE assignment
                for i, name in enumerate(res_name):
                    if name == resname and res_id[i] == resseq and chain_id[i] == chainID:
                        if res_ss[i] == 'H' or res_ss[i] == 'G' or res_ss[i] == 'I':
                            secstr.append(0)
                        elif res_ss[i] == 'B' or res_ss[i] == 'b' or res_ss[i] == 'E':
                            secstr.append(1)
                        else:
                            secstr.append(2)

                        assigned = True
                        break

                if not assigned:
                    secstr.append(3)

                chainIDs.append(chainID)
                resseqs.append(resseq)
                coords.append(atom.get_coord())
                n_atoms += 1
        break

    coords = np.array(coords, dtype='float32')
    secstr = np.array(secstr, dtype='int64')
    assert(len(coords) == len(secstr))

    return coords, secstr


def assign_predicted_SS_to_atoms(predicted_grids, predictions, coords, full=0):
    tree = BallTree(predicted_grids, leaf_size=2)

    indices = tree.query_radius(coords, r=3.0)

    secstr = np.full(len(coords), full, dtype=np.int64)
    for i, index in enumerate(indices):
        if len(index) == 0:
            continue
        secstr[i] = np.argmax(np.bincount(predictions[index]))
    return secstr


def Get_path_according_to_type(map_type, mapout_path, name, fold, model_dir):
    map_types = ["SIM6", "SIM10", "EXP25", "EXP"]
    if map_type not in map_types:
        print("[Error] --type, type of cryo-EM density map that does not exist")
        sys.exit()
    mapout_path = os.path.join(mapout_path, map_type)
    if not os.path.exists(mapout_path):
        print(f"# mkdir {mapout_path}")
        os.mkdir(mapout_path)
    else:
        print(f"# {mapout_path} already exists")
    mapout_path = os.path.join(mapout_path, name)
    if not os.path.exists(mapout_path):
        print(f"# mkdir {mapout_path}")
        os.mkdir(mapout_path)
    else:
        print(f"# {mapout_path} already exists")

    npy_all_path = None
    if map_type == "EXP":
        if fold == "none":
            mapout_path = os.path.join(mapout_path, "Folds")
            if not os.path.exists(mapout_path):
                print(f"# mkdir {mapout_path}")
                os.mkdir(mapout_path)
            else:
                print(f"# {mapout_path} already exists")
            model_path = [f"{model_dir}/model_{map_type}_fold{i+1}.pth" for i in range(5)]
            npy_all_path = [f"{mapout_path}/{name}_fold{i+1}.npy" for i in range(5)]
        else:
            mapout_path = os.path.join(mapout_path, f"Fold{fold}")
            if not os.path.exists(mapout_path):
                print(f"# mkdir {mapout_path}")
                os.mkdir(mapout_path)
            else:
                print(f"# {mapout_path} already exists")
            model_path = f"{model_dir}/model_{map_type}_fold{fold}.pth"
    else:
        model_path = f"{model_dir}/model_{map_type}.pth"
    mapout_all_path = f"{mapout_path}/{name}_all.mrc"
    mapout_cx_path = [f"{mapout_path}/{name}_c{i}.mrc" for i in range(4)]
    repdb_path = f"{mapout_path}/{name}.pdb"
    result_path = f"{mapout_path}/{name}_result.txt"
    return model_path, mapout_all_path, mapout_cx_path, npy_all_path, repdb_path, result_path



if __name__ == "__main__":
    parse_map(map_path="./2O49_SIM6_mod.mrc", apix=1.0)
