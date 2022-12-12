import os
import shutil
import glob

def rename(src_name, dst_name):
    os.rename(src_name, dst_name)

def copyfile(src_name, dst_name):
    shutil.copyfile(src_name, dst_name)


def pair_rename(source1, source2, src1_type, src2_type, save_dir, seq_idx=1):
    src1_files = glob.glob(os.path.join(source1, "*." + src1_type))
    # src2_files = glob.glob(os.path.join(source2, "*." + src2_type))

    src1_files.sort()
    # src2_files.sort()

    for src1 in src1_files:
        split = src1.rindex('/')
        src2 = os.path.join(source2,src1[split + 1:][:- len(src1_type)] + src2_type)

        if os.path.exists(src1) and os.path.exists(src2):
            dst1 = save_dir + f"{seq_idx:06d}." + src1_type
            dst2 = save_dir + f"{seq_idx:06d}." + src2_type

            copyfile(src1, dst1)
            copyfile(src2, dst2)
            seq_idx += 1

pair_rename("/data/lj/datasets/vis/","/data/lj/datasets/vis/","json","jpg", "/data/lj/datasets/new_pandas/")
