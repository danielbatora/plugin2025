
from skimage.measure import regionprops, label
import skimage.io as io
import os 
from dask import delayed
import numpy as np 
from bg_atlasapi.bg_atlas import BrainGlobeAtlas

import matplotlib.pyplot as plt 
from skimage.transform import resize
from scipy import signal

from tqdm import tqdm
from skimage.filters import threshold_otsu, gaussian

from cellpose import models
from stardist.models import StarDist2D    
from csbdeep.utils import normalize 
import scipy

import psutil
from napari.utils import progress
from shapely.geometry import Point, Polygon



def df_remove_suffixes(data): 
        
    df = data.copy()
    if any(df.columns.str.contains("_y")): 
        
        for i in range(df.shape[0]): 
            if all(df.loc[i, df.columns.str.contains("_x")].values == df.loc[i, df.columns.str.contains("_y")].values):
                df.loc[i, df.columns.str.contains("_y")] = np.nan
                
    try:
        df.loc[df.loc[:, df.columns.str.contains("_x")].isna().iloc[:,0], df.columns.str.contains("_x")] = df.loc[~df.loc[:, df.columns.str.contains("_y")].isna().iloc[:,0], df.columns.str.contains("_y")].values 
        
        df.columns = df.columns.str.replace("_x", "")

        
        return df.loc[:, ~df.columns.str.contains("_y")]
        
    except: 
        for i in range(df.shape[0]): 
            if all(df.loc[i, df.columns.str.contains("_x")].isna().values) and not all(df.loc[i, df.columns.str.contains("_y")].isna().values): 
                df.loc[i, df.columns.str.contains("_x")] = df.loc[i, df.columns.str.contains("_y")].values
        df.columns = df.columns.str.replace("_x", "")

        return df.loc[:, ~df.columns.str.contains("_y")]
    else: 
        return df 
    

def generate_polygon_mask(cropped_image,rectangle,polygon):

    crop_x = np.unique(np.array([i[0] for i in rectangle]))
    crop_y = np.unique(np.array([i[1] for i in rectangle]))    


    poly_mask = np.zeros_like(cropped_image)
    polygon_obj = Polygon(polygon)
    
    count_x = 0
    count_y = 0
    for x in range(crop_x.min(), crop_x.max()):
        for y in range(crop_y.min(), crop_y.max()):
            poly_mask[count_x, count_y] = polygon_obj.contains(Point(x, y))
            count_y += 1
        count_y = 0
        count_x += 1
    return poly_mask


def crop_with_polygon_mask(cropped_image, poly_mask): 
    
    """
    Put pixel values to zero where the rectangle is not in the polygon
    """
    
    return np.where(poly_mask, cropped_image, 0)


def calculate_chunk_size(): 
    
    total_mb_ram = psutil.virtual_memory().total / (1024 * 1024 ) 

    
    chunksize = total_mb_ram // 6.183
    
    return (int(round(chunksize, -3)), int(round(chunksize, -3)))

def calculate_lag(reference, our_image): 
    
    ref_bin = reference > threshold_otsu(reference)
    img_bin = our_image > threshold_otsu(our_image)

    corr_2d = signal.correlate2d(ref_bin.astype(int), img_bin.astype(int), boundary = "fill")
    local_max  = np.unravel_index(np.argmax(corr_2d), corr_2d.shape)
    lag =  np.array(ref_bin.shape) - np.array(local_max) 
    
    
    
    return lag, corr_2d


def calculate_bounding_rectangle(polygon): 
    x_coords = np.unique(np.array([i[0] for i in polygon]))
    y_coords = np.unique(np.array([i[1] for i in polygon]))
    
    rectangle = np.array([[x_coords.min(), y_coords.min()], 
                          [x_coords.min(), y_coords.max()],
                          [x_coords.max(), y_coords.max()], 
                          [x_coords.max(), y_coords.min()]])

    return rectangle.astype(int)


def register_image(reference, our_image, gaussian = True): 
    
    
    ref_bin = reference > threshold_otsu(reference)
    if gaussian: 
        img_bin = our_image > threshold_otsu(gaussian(our_image, 5))
    else: 
        img_bin = our_image > threshold_otsu(our_image, 5)

    corr_2d = signal.correlate2d(ref_bin.astype(int), img_bin.astype(int), boundary = "fill")
    local_max  = np.unravel_index(np.argmax(corr_2d), corr_2d.shape)
    lag =  np.array(ref_bin.shape) - np.array(local_max) 
    
    
    return scipy.ndimage.shift(our_image, -lag)




def return_mask(img, method="stardist"):
    
    
    if method =="stardist": 
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        masks, _ = model.predict_instances(normalize(img), verbose=False)
        return masks
        
    elif method =="cellpose" :
        model = models.Cellpose(gpu=False, model_type='cyto2')
        masks, _, _, _ = model.eval(img, diameter=20, channels=[0,0],flow_threshold=0.4, do_3D=False)
        return masks
    else: 
        return False



def calculate_props(labels: np.array, image: np.array, prop = "intensity_mean"): 
        
    assert labels.shape == image.shape
    props = regionprops(labels, image)
    
    return [i[prop] for i in props]
    
    

def calculate_rate(ref, target, coloc_thr = 0.5): 
    
    if np.unique(target).shape[0] > 2: 
        target = target > threshold_otsu(target)
        
    ref_inds = np.delete(np.unique(ref), 0)
    ref_count = ref_inds.shape[0]
    
    print(f"{ref_count} cells found")
    
    positive_cells = 0
    for i in progress(ref_inds): 
        rate = target[ref == i].sum() / (ref == i).sum()
        if rate > coloc_thr: 
            positive_cells += 1
            
    return positive_cells / ref_count
  

def return_positive_negative_cells(ref, target, coloc_thr = 0.5): 
    
    if np.unique(target).shape[0] > 2: 
        target = target > threshold_otsu(target)
        
    ref_inds = np.delete(np.unique(ref), 0)
    ref_count = ref_inds.shape[0]
    
    print(f"{ref_count} cells found")
    
    positive_cells = ref.copy()
    negative_cells = ref.copy()
    
    
    for i in progress(ref_inds): 
        rate = target[ref == i].sum() / (ref == i).sum()
        if rate < coloc_thr: 
            positive_cells = np.where(positive_cells ==i, 0, positive_cells)
        else: 
            negative_cells = np.where(negative_cells ==i, 0, negative_cells)

    return positive_cells, negative_cells
  
    

        

def fit_dim(polygon, max_shape): 
    
    
    coords_x = [i[0] for i in polygon]
    coords_y = [i[1] for i in polygon]
    
    
    for i in range(len(coords_x)): 
        if coords_x[i] > max_shape[0]: 
            polygon[i][0] = max_shape[0]
        elif coords_y[i] > max_shape[1]:
            polygon[i][1] = max_shape[1]
    
    return polygon



def relabel(mask, chunks): 
    """
    relabel stiched labels from chunks to make them unique
    """
    
    mask = mask.copy()
    
    numchunks_x = np.ceil(mask.shape[0] / chunks[0]).astype(int)
    numchunks_y = np.ceil(mask.shape[1] / chunks[1]).astype(int)

    

    prev_max = 0
    x_start = 0
    y_start = 0
    
    
    for i in range(numchunks_x):
        for j in range(numchunks_y): 
            
            if chunks[0] + x_start <= mask.shape[0]: 
                slice_x = (i * chunks[0],chunks[0] + x_start)
            else: 
                slice_x = (i * chunks[0], mask.shape[0])
                
            if chunks[1] + y_start <= mask.shape[1]: 
                slice_y = (j * chunks[1] ,chunks[1] + y_start )
            else: 
                slice_y = (j* chunks[1], mask.shape[1])
                
            
            mask[slice_y[0]: slice_y[1], slice_x[0]:slice_x[1]] = np.where(mask[slice_y[0]: slice_y[1], slice_x[0]:slice_x[1]] != 0, mask[slice_y[0]: slice_y[1], slice_x[0]:slice_x[1]] + prev_max, mask[slice_y[0]: slice_y[1], slice_x[0]:slice_x[1]])
            prev_max = mask[slice_y[0]: slice_y[1],slice_x[0]:slice_x[1]].max()
            y_start += chunks[0]
        y_start = 0
        x_start += chunks[1]
    return mask
            

def return_ranges(size, chunks): 
    numchunks_x = np.ceil(size[0] / chunks[0]).astype(int)
    numchunks_y = np.ceil(size[1] / chunks[1]).astype(int)

    x_start = 0
    y_start = 0
    
    
    ranges = []
    
    for i in range(numchunks_x):
        for j in range(numchunks_y): 
            
            if chunks[0] + x_start <= size[0]: 
                slice_x = (i * chunks[0],chunks[0] + x_start)
            else: 
                slice_x = (i * chunks[0],size[0])
                
            if chunks[1] + y_start <= size[1]: 
                slice_y = (j * chunks[1] ,chunks[1] + y_start )
            else: 
                slice_y = (j* chunks[1], size[1])            
            ranges.append((slice_x, slice_y))

            y_start += chunks[0]
        y_start = 0
        x_start += chunks[1]
            
    return ranges
            
            
def split_task_to_chunks(img, chunks, filepath):
    ranges = return_ranges(img.shape, chunks)
    files = []
    prev_max = 0
    for i in tqdm(range(len(ranges))):
        x = ranges[i][0]
        y = ranges[i][1]
        mask = delayed(return_mask)(img[x[0]: x[1], y[0]:y[1]], method ="stardist").compute()
        mask = np.where(mask != 0, mask + prev_max, mask)
        prev_max = mask.max()
        files.append(filepath.strip(".tif") + f"_stadist_mask_{i}.npy")
        np.save(filepath.strip(".tif") + f"_stadist_mask_{i}", mask)
    return files
    

def assemble_chunks(size,chunks,  chunkfiles): 
    ranges = return_ranges(size, chunks)
    assemble = np.zeros(size)
    savename = chunkfiles[0].rstrip("_0.npy")
    
    assert len(ranges) == len(chunkfiles)
    for i in range(len(chunkfiles)): 
        x = ranges[i][0]
        y = ranges[i][1]
        
        chunk = np.load(chunkfiles[i])
        assemble[x[0]: x[1], y[0]:y[1]] = chunk 
        os.remove(chunkfiles[i])
    
    np.save(savename, assemble)
    
    
def match_atlas(img, atlas_name, hemisphere, create_mask=False, subrange=False, plot=False): 
    atlas = BrainGlobeAtlas(atlas_name)
    reference = atlas.reference
    hemi = atlas.hemispheres[0]
 
    corrs = []
    
    if not subrange: 
        use_range = (0, reference.shape[0])
    else: 
        use_range = subrange
    
    
    for i in tqdm(range(use_range[0], use_range[1])):
        
        ref = reference[i][hemi==hemisphere]
        ref = ref.reshape((reference[i].shape[0], ref.shape[0] // reference[i].shape[0]))
        our_image = resize(img, ref.shape, preserve_range = True)
        
        if create_mask: 
            our_image = register_image(ref, our_image)
        corr = np.mean(signal.correlate2d(ref, our_image))
        corrs.append(corr)
    
    
    if plot: 
        plt.plot(corrs, color ="blue")
        plt.xlabel("Slice #")
        plt.ylabel("mean 2dcorr")
        plt.show()
        plt.close()

    return np.argmax(corrs) + use_range[0]

