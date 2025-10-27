import nd2
import numpy as np 
import glob 
import os 
from tqdm import tqdm 
folder = "/media/danielbatora/Dani/Adonis/63 CRE F/Cre staining"
npy_folder = os.path.join(folder, "npy_files")


if not os.path.isdir(npy_folder): 
    os.mkdir(npy_folder)


for file in tqdm(glob.glob(os.path.join(folder, "**/*.nd2"), recursive = True)): 
    if not os.path.isfile(file.replace(".nd2", ".npy")):
        img = nd2.imread(file)
        np.save(os.path.join(npy_folder, file.strip(".nd2").split("/")[-1]), img.data)
    
    
    
    
    