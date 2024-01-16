import stat
import os.path
from collections import OrderedDict
from modules import scripts, shared, sd_models, devices
from modules.paths import models_path

SAM_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin"]
scripts_sam_model_dir = os.path.join(scripts.basedir(), "models/sam")
sd_sam_model_dir = os.path.join(models_path, "sam")
sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir
sam_models = OrderedDict()
sam_models_names = {}
sam_device = devices.device

def traverse_all_files(curr_path, model_list):
    f_list = [
        (os.path.join(curr_path, entry.name), entry.stat())
        for entry in os.scandir(curr_path)
        if os.path.isdir(curr_path)
    ]
    print(f_list)
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in SAM_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list


def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[os.path.basename(filename) + f" [{sd_models.model_hash(filename)}]"] = filename

    return res


def update_sam_models():
    sam_models.clear()
    paths = [sam_model_dir]
    
    for path in paths:
        sort_by = shared.opts.data.get(
            "control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        sam_models.update({**found, **sam_models})

    sam_models_names.clear()
    for name_and_hash, filename in sam_models.items():
        if filename is None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        sam_models_names[name] = name_and_hash


def change_sam_device(use_cpu=False):
    sam_device = "cpu" if use_cpu else device
    