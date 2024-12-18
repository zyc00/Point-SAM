import os
import sys
import json
import argparse
import h5py
import numpy as np
from progressbar import ProgressBar
from subprocess import call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/home/ubuntu/projects/yuchen/pointcloud-sam")
from pc_sam.commons import check_mkdir, force_mkdir

parser = argparse.ArgumentParser()
parser.add_argument("category", type=str, help="category")
parser.add_argument("level_id", type=int, help="level_id")
parser.add_argument("split", type=str, help="split train/val/test")
parser.add_argument("--num_point", type=int, default=10000, help="num_point")
parser.add_argument("--num_ins", type=int, default=200, help="num_ins")
args = parser.parse_args()

# load meta data files
stat_in_fn = "./stats/after_merging_label_ids/%s-hier.txt" % (args.category)
print("Reading from ", stat_in_fn)
with open(stat_in_fn, "r") as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print("Part Name List: ", part_name_list)

NUM_CLASSES = len(part_name_list)
print("Semantic Labels: ", NUM_CLASSES)
NUM_INS = args.num_ins
print("Number of Instances: ", NUM_INS)
NUM_POINT = args.num_point
print("Number of Points: ", NUM_POINT)


def load_h5(fn):
    with h5py.File(fn, "r") as fin:
        pts = fin["pts"][:]
        colors = fin["rgb"][:]
        label = fin["label"][:]
        return pts, colors, label


def load_json(fn):
    with open(fn, "r") as fin:
        return json.load(fin)


def save_h5(fn, pts, colors, gt_label, gt_mask, gt_valid, gt_other_mask):
    fout = h5py.File(fn, "w")
    fout.create_dataset(
        "pts", data=pts, compression="gzip", compression_opts=4, dtype="float32"
    )
    fout.create_dataset(
        "rgb", data=colors, compression="gzip", compression_opts=4, dtype="uint8"
    )
    fout.create_dataset(
        "gt_label", data=gt_label, compression="gzip", compression_opts=4, dtype="uint8"
    )
    fout.create_dataset(
        "gt_mask", data=gt_mask, compression="gzip", compression_opts=4, dtype="bool"
    )
    fout.create_dataset(
        "gt_valid", data=gt_valid, compression="gzip", compression_opts=4, dtype="bool"
    )
    fout.create_dataset(
        "gt_other_mask",
        data=gt_other_mask,
        compression="gzip",
        compression_opts=4,
        dtype="bool",
    )
    fout.close()


def reformat_data(in_h5_fn, out_h5_fn):
    # save json
    in_json_fn = in_h5_fn.replace(".h5", ".json")
    record = load_json(in_json_fn)

    out_json_fn = out_h5_fn.replace(".h5", ".json")
    cmd = "cp %s %s" % (in_json_fn, out_json_fn)
    print(cmd)
    call(cmd, shell=True)

    # save h5
    pts, colors, label = load_h5(in_h5_fn)
    print("pts: ", pts.shape)
    print("colors: ", colors.shape)
    print("label: ", label.shape)

    # get the first NUM_POINT points
    pts = pts[:, :NUM_POINT, :]
    colors = colors[:, :NUM_POINT, :]
    label = label[:, :NUM_POINT]

    n_shape = label.shape[0]

    gt_label = np.zeros((n_shape, NUM_POINT), dtype=np.uint8)
    gt_mask = np.zeros((n_shape, NUM_INS, NUM_POINT), dtype=bool)
    gt_valid = np.zeros((n_shape, NUM_INS), dtype=bool)
    gt_other_mask = np.zeros((n_shape, NUM_POINT), dtype=bool)

    bar = ProgressBar()
    for i in bar(range(n_shape)):
        cur_label = label[i, :NUM_POINT]
        cur_record = record[i]
        cur_tot = 0
        for item in cur_record["ins_seg"]:
            if item["part_name"] in part_name_list:
                selected = np.isin(cur_label, item["leaf_id_list"])
                gt_label[i, selected] = part_name_list.index(item["part_name"]) + 1
                gt_mask[i, cur_tot, selected] = True
                gt_valid[i, cur_tot] = True
                cur_tot += 1
        gt_other_mask[i, :] = gt_label[i, :] == 0

    save_h5(out_h5_fn, pts, colors, gt_label, gt_mask, gt_valid, gt_other_mask)


# main
data_in_dir = "/data/yuchen/PartNet/ins_seg_h5/%s/" % args.category
data_out_dir = "/data/yuchen/process/ins_seg_h5_for_detection"
force_mkdir(data_out_dir)
data_out_dir = os.path.join(data_out_dir, "%s-%d" % (args.category, args.level_id))
force_mkdir(data_out_dir)

print(args.category, args.level_id, args.split)

h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith(".h5") and item.startswith("%s-" % args.split):
        h5_fn_list.append(item)

for item in h5_fn_list:
    in_h5_fn = os.path.join(data_in_dir, item)
    out_h5_fn = os.path.join(data_out_dir, item)
    reformat_data(in_h5_fn, out_h5_fn)
