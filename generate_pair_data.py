
import os
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib
import cv2
import SimpleITK as sitk


path_MRI = "/home/gau/smallDiffusion/dataset/adni1_pair"


path_bl = "/home/gau/smallDiffusion/dataset/ADNI1_CNMCI_bl"
path = "/home/gau/smallDiffusion/dataset/ADNI1_CNMCI_m36"  

files_bl = os.listdir(path_bl)
files = os.listdir(path)
    
for fi in files:

    fmri = os.path.join(path, fi)
    fmri_bl = os.path.join(path_bl, fi)

    if os.path.exists(fmri_bl):
        # print("文件存在")
        
        img_mri = nib.load(fmri).get_fdata()
        img_mri = img_mri/255    

        img_mri_bl = nib.load(fmri_bl).get_fdata()
        img_mri_bl = img_mri_bl/255 

        a = np.expand_dims(img_mri_bl, axis = 0) 
        b = np.expand_dims(img_mri, axis = 0) 
        
        img = np.concatenate((a, b), axis = 0) 


        file_name = os.path.join(path_MRI, fi) 
        nii = nib.Nifti1Image(img, np.eye(4))
        
        # 保存NIfTI文件
        nib.save(nii, file_name)



# path_MRI = "/home/gau/smallDiffusion/dataset/ADNI3_CNMCI_bl"


# path = "/media/gau/gaoxingyu/ADNI3_Z_INPUT/MCI"  
# files = os.listdir(path)
    
# for fi in files:

#     fmri = os.path.join(path, fi)


#     img_mri = nib.load(fmri).get_fdata()
#     img_mri = img_mri/255    



#     file_name = os.path.join(path_MRI, fi[:-3]) 
#     nii = nib.Nifti1Image(img_mri, np.eye(4))
    
#     # 保存NIfTI文件
#     nib.save(nii, file_name)






# path = "/home/gau/DATA/ADNI4_MCI_MRI_bl"
# path_to = "/home/gau/DATA/ADNI4_MCI_MRI_bl_nii"  

# files1 = os.listdir(path)
# for f1 in files1:
#     path1 = os.path.join(path, f1)
#     files2 = os.listdir(path1)
#     for f2 in files2:
#         path2 = os.path.join(path1, f2)
#         files3 = os.listdir(path2)
#         for f3 in files3:
#             path3 = os.path.join(path2, f3)
#             files4 = os.listdir(path3)
#             for f4 in files4:
#                 path4 = os.path.join(path3, f4)
#                 files5 = os.listdir(path4)
#                 for f5 in files5:
#                     path5 = os.path.join(path4, f5)

#                     reader = sitk.ImageSeriesReader()
#                     dicom_names = reader.GetGDCMSeriesFileNames(path5)
#                     reader.SetFileNames(dicom_names)
#                     image2 = reader.Execute()
#                     path_save = os.path.join(path_to, f2)
#                     if not os.path.exists(path_save):
#                         os.makedirs(path_save)
#                     path_file_name = os.path.join(path_save, 'mri.nii')
#                     sitk.WriteImage(image2, path_file_name)
#                     print()

