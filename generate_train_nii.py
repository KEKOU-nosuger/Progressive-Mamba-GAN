
import os
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib
import cv2



path_MRI = "/home/gau/smallDiffusion/dataset/ADNI1_CNMCI_bl_PET"

# x_min =12, x_max =168
# y_min =12, y_max =205
# z_min =12, z_max =181


check = True
while check:



    path = "/home/gau/smallDiffusion/data/ADNI1_CNMCI_bl/CN/PET"    ### CAUTION THIS !!!### 
    files = os.listdir(path)
       
    for fi in files:
        num = 0
        path_son = os.path.join(path, fi)


        fmri = os.path.join(path_son, 'pet_registration_better.nii.gz')

        img_mri = nib.load(fmri).get_fdata()
        img3d_mri = img_mri[10:163, 16:204, 10:163] 

        level = img3d_mri.shape 
        e = np.zeros(shape=(level[0]//2, level[1], level[2]), dtype='float32')
        f = np.zeros(shape=(level[0]//2, level[1]//2, level[2]), dtype='float32')
        g = np.zeros(shape=(level[0]//2, level[1]//2, level[2]//2), dtype='float32')
        
        for x in range(level[0]//2):
            e[x, :, :]= img3d_mri[x*2, :, :]

        for y in range(level[1]//2):
            f[:, y, :]= e[:, y*2, :]

        for z in range(level[2]//2):
            g[:, :, z]= f[:, :, z*2]

        gsize = g.shape 
        # image stretching: 0-255
        flat_gray_g = g.reshape((gsize[0] * gsize[1] * gsize[2], )).tolist()
        N = min(flat_gray_g)
        M = max(flat_gray_g)
        e = 255 / (M - N) * (g - N)

        file_name = os.path.join(path_MRI, '0_' + fi + '.nii') 
        nii = nib.Nifti1Image(e, np.eye(4))
        
        # 保存NIfTI文件
        nib.save(nii, file_name)

    check = False