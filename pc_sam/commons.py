import os
import sys
import shutil


def printout(flog, data):
    print(data)
    flog.write(data + "\n")


def check_mkdir(x):
    if os.path.exists(x):
        print("ERROR: folder %s exists! Please check and delete it!" % x)
        exit(1)
    else:
        os.mkdir(x)


def force_mkdir(x):
    if not os.path.exists(x):
        os.mkdir(x)


def force_mkdir_new(x):
    if not os.path.exists(x):
        os.mkdir(x)
    else:
        shutil.rmtree(x)
        os.mkdir(x)


def check_exist_dir(x):
    check_dir_exist(x)


def check_dir_exist(x):
    if not os.path.exists(x):
        print("ERROR: folder %s does not exists!" % x)
        exit(1)
