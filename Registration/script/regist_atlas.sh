#!/bin/bash
# Run affine transformation according to atlas using CMTK.

export CMTK_WRITE_UNCOMPRESSED=1

# Set the red chanel image directory path that is needed to register.
red_path="../../data/R/dual_Crop"

# Set the output red chanel image directory path that is registered.
regist_red_path="../../data/R/template"
mkdir $regist_red_path

# Set the atlas path.
zbbfish="../../data/Atlas/Ref-zbb1.nii"

# Read the red images.
file_name=$(ls $red_path/Red*.nii);
file_name=(${file_name//,/ });
len=$(ls -l $red_path/Red*.nii | grep "^-" | wc -l);

# Set the template step (recommend 100).
step_size=1

for ((i=0;i<$len;i=i+$step_size))
do
    name_num=$(basename -s .nii ${file_name[$i]});

    # Set k to the number of the name.
    k=${name_num:3};

    start_time=$(date +%s)

    # Initialize affine matrix.
    cmtk make_initial_affine --principal-axes $zbbfish ${red_path}/Red${k}.nii ${red_path}/initial${k}.xform
    
    # Generate affine matrix.
    cmtk registration --initial ${red_path}/initial${k}.xform --dofs 12,12 --exploration 8 --accuracy 0.05 --cr -o ${red_path}/affine${k}.xform $zbbfish ${red_path}/Red${k}.nii

    # Apply affine matrix.
    cmtk reformatx -o ${regist_red_path}/template${k}.nii --floating ${red_path}/Red${k}.nii $zbbfish ${red_path}/affine${k}.xform

    end_time=$(date +%s)
    cost_time=$[ $end_time-$start_time ]
    echo "Reg & Warp time is $(($cost_time/60))min $(($cost_time%60))s"
    echo $name_num
 
done
