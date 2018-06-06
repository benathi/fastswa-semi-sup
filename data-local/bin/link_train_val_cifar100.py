import os
import sys
import glob
import random

# run this at the current directory to generate 'link_cifar100_train.sh' and 'link_cifar100_val.sh'

# link train+val to train and val
def generate_symlink_script():
    # separate train+val into val (10%) and train (the rest)
    image_folder = '../images/cifar/cifar100/by-image/train+val'
    out_f = {'train':open('link_cifar100_train.sh', 'w'), 'val': open('link_cifar100_val.sh', 'w')}

    out_f['train'].write("mkdir train\n")
    out_f['train'].write("ls train+val | xargs -I {} mkdir 'train/{}'\n")

    out_f['val'].write("mkdir val\n")
    out_f['val'].write("ls train+val | xargs -I {} mkdir 'val/{}'\n")

    val_per_class = 50
    # list all folders
    foldernames = os.walk(image_folder)
    for name in foldernames:
        # name[0] - a name of each class
        # name[2] - a list of all the files
        classname = name[0]
        if classname == image_folder:
            continue
        classname = classname.split("/")[-1]
        listfiles = name[2]
        assert len(listfiles) == 500
        # do random partition
        #print("First 5", listfiles[:5])
        random.shuffle(listfiles)
        #print("First 5 after", listfiles[:5])
        for i, fname in enumerate(listfiles):
            fname = fname.split("/")[-1]
            dirandfname = "{}/{}".format(classname, fname)
            if i < val_per_class:
                f = out_f['val']
                f.write("ln -s ../../train+val/{} val/{}\n".format(dirandfname, dirandfname))
            else:
                f = out_f['train']
                f.write("ln -s ../../train+val/{} train/{}\n".format(dirandfname, dirandfname))

if __name__=="__main__":
    generate_symlink_script()