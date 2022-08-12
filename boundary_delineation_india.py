import rasterio
import numpy as np
from dask import delayed
import os
import cv2
import math
from time import sleep
from skimage.morphology import closing,opening,thin
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def main(): ##This is the main function
    os.chdir("E:\Crop Field Delineation\Ground Truth Anand University")
    data=rasterio.open("clipped_input.tif")
    data = rasterio.open("Clipped_large_fields.tif")
    nir_bands = data.read()
    nir_bands = nir_bands[:,0:1536,0:2048]
    shape=nir_bands[0].shape
    std_bands =[]
    for band in nir_bands:
        results = delayed (calculate_std) (band,shape)
        results = results.compute()
        std_bands.append(results)
        print(std_bands)
    std_avg = delayed(avg_std)(std_bands, shape)
    std_avg = std_avg.compute()
    plt.imshow(std_avg)
    #np.save('clipped_large_fields_std_avg.npy',std_avg)
    final_conv_bands=[]
    left_conv_bands,right_conv_bands = directional_filters(std_avg)
    #np.save('left_conv_bands.npy',left_conv_bands)
    #np.save('right_conv_bands.npy',right_conv_bands)
    #np.save('clipped_large_fields_left_conv_bands_liss4.npy',left_conv_bands)
    #np.save('clipped_large_fields_right_conv_bands_liss4.npy',right_conv_bands)
    for i in range(len(left_conv_bands)):
        results=delayed (perform_thresh)(left_conv_bands[i],right_conv_bands[i],shape)
        results=results.compute().astype('uint8')
        final_conv_bands.append(results)
    cca_bands = delayed(compute_cca)(final_conv_bands)
    cca_bands = cca_bands.compute()
    res = result_(cca_bands,shape)
    res = np.array(res)
    np.save('test_large_fields_boundary_delineation.npy',res)
    plt.imshow(res)

    input = res
    area_threshold = 0.8
    result,image = morphological_profile(input,area_threshold)
    color_labels = color.label2rgb(result, image, alpha=0.9, bg_label=0)
    plt.imshow(result)
    plt.show()
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(color_labels)
    ax.set_title('Crop Fields')
    plt.show()
    plt.imshow(res)
    
@delayed
def calculate_std(band,shape):
    sleep(1)
    std_band=np.zeros(shape)
    for i in range(2,(band.shape[0]-2)):
        for j in range(2,(band.shape[1]-2)):
            mean=0 
            row1=band[i][j-2:j+3]
            row2=band[i-1][j-2:j+3]
            row3=band[i-2][j-1:j+2]
            row4=band[i+1][j-2:j+3]
            row5=band[i+2][j-1:j+2]
            mean+=band[i][j-2:j+3].sum()
            mean+=band[i-1][j-2:j+3].sum()
            mean+=band[i-2][j-1:j+2].sum()
            mean+=band[i+1][j-2:j+3].sum()
            mean+=band[i+2][j-1:j+2].sum()
            mean=mean/21
            row1=(row1-mean)**2
            row2=(row2-mean)**2
            row3=(row3-mean)**2
            row4=(row4-mean)**2
            row5=(row5-mean)**2
            var=row1.sum()+row2.sum()+row3.sum()+row4.sum()+row5.sum()
            var=var/21
            std=math.sqrt(var)
            std_band[i-2][j-2]=std
    plt.imshow(std_band)

    return std_band

@delayed
def avg_std(std_bands,shape):
    sleep(1)
    avg_std=np.zeros(shape)
    avg_std = np.mean(std_bands,axis=0)
    return avg_std

#Applying convolution to avg_std image with 16 pairs of consecutive directional operators that are 11.25degrees apart.
def directional_filters(avg_std):
    left_filters=[]
    right_filters=[]

    left_0 = np.zeros((13,13),np.float32)
    left_0[:,6]=1
    left_0[:,5]=-1
    left_filters.append(left_0)
    left_0
    
    left_11=np.array([[ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], np.float32)
    left_filters.append(left_11)

    left_22=np.array([[ 0.,  0.,  0.,  0.,  0., 0.,  0.,  -1.,  1.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  0.,  -1.,  1.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  -1.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  -1.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], np.float32)
    left_filters.append(left_22)
    

    left_33=np.array([[ 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  -1.,  1.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  -1.,  1.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  0.,  -1.,  1.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  0.,  -1.,  1.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., 0.,  -1.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  0.,  -1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  -1.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  0.,  -1.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  -1.,  1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  -1.,  1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], np.float32)
    left_filters.append(left_33)


    left_45 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                    np.float32)
    left_filters.append(left_45)

    left_56 = np.flipud(np.rot90(left_33))
    left_filters.append(left_56)

    left_67 = np.flipud(np.rot90(left_22))
    left_filters.append(left_67)

    left_78 = np.flipud(np.rot90(left_11))
    left_filters.append(left_78)

    left_90 = np.flipud(np.rot90(left_0))
    left_filters.append(left_90)

    left_101=np.fliplr(left_78)
    left_filters.append(left_101)

    left_112=np.fliplr(left_67)
    left_filters.append(left_112)

    left_123=np.fliplr(left_56)
    left_filters.append(left_123)

    left_135=np.fliplr(left_45)
    left_filters.append(left_135)

    left_146=np.fliplr(left_33)
    left_filters.append(left_146)

    left_157=np.fliplr(left_22)
    left_filters.append(left_157)

    left_168=np.fliplr(left_11)
    left_filters.append(left_168)

    for left_filter in left_filters:
        right_filters.append(np.fliplr(np.flipud(left_filter)))
    left_conv_bands=[]
    right_conv_bands=[]

    l0_left=cv2.filter2D(avg_std,-1,left_filters[0])
    l0_right=cv2.filter2D(avg_std,-1,right_filters[0])

    left_conv_bands.append(l0_left)
    right_conv_bands.append(l0_right)

    l1_left=cv2.filter2D(avg_std,-1,left_filters[1])
    l1_right=cv2.filter2D(avg_std,-1,right_filters[1])

    left_conv_bands.append(l1_left)
    right_conv_bands.append(l1_right)

    l2_left=cv2.filter2D(avg_std,-1,left_filters[2])
    l2_right=cv2.filter2D(avg_std,-1,right_filters[2])

    left_conv_bands.append(l2_left)
    right_conv_bands.append(l2_right)

    l3_left=cv2.filter2D(avg_std,-1,left_filters[3])
    l3_right=cv2.filter2D(avg_std,-1,right_filters[3])

    left_conv_bands.append(l3_left)
    right_conv_bands.append(l3_right)
    
    l4_left=cv2.filter2D(avg_std,-1,left_filters[4])
    l4_right=cv2.filter2D(avg_std,-1,right_filters[4])

    left_conv_bands.append(l4_left)
    right_conv_bands.append(l4_right)

    l5_left=cv2.filter2D(avg_std,-1,left_filters[5])
    l5_right=cv2.filter2D(avg_std,-1,right_filters[5])

    left_conv_bands.append(l5_left)
    right_conv_bands.append(l5_right)

    l6_left=cv2.filter2D(avg_std,-1,left_filters[6])
    l6_right=cv2.filter2D(avg_std,-1,right_filters[6])

    left_conv_bands.append(l6_left)
    right_conv_bands.append(l6_right)
    
    l7_left=cv2.filter2D(avg_std,-1,left_filters[7])
    l7_right=cv2.filter2D(avg_std,-1,right_filters[7])

    left_conv_bands.append(l7_left)
    right_conv_bands.append(l7_right)

    l8_left=cv2.filter2D(avg_std,-1,left_filters[8])
    l8_right=cv2.filter2D(avg_std,-1,right_filters[8])

    left_conv_bands.append(l8_left)
    right_conv_bands.append(l8_right)

    l9_left=cv2.filter2D(avg_std,-1,left_filters[9])
    l9_right=cv2.filter2D(avg_std,-1,right_filters[9])

    left_conv_bands.append(l9_left)
    right_conv_bands.append(l9_right)

    l10_left=cv2.filter2D(avg_std,-1,left_filters[10])
    l10_right=cv2.filter2D(avg_std,-1,right_filters[10])

    left_conv_bands.append(l10_left)
    right_conv_bands.append(l10_right)

    l11_left=cv2.filter2D(avg_std,-1,left_filters[11])
    l11_right=cv2.filter2D(avg_std,-1,right_filters[11])

    left_conv_bands.append(l11_left)
    right_conv_bands.append(l11_right)

    l12_left=cv2.filter2D(avg_std,-1,left_filters[12])
    l12_right=cv2.filter2D(avg_std,-1,right_filters[12])

    left_conv_bands.append(l12_left)
    right_conv_bands.append(l12_right)

    l13_left=cv2.filter2D(avg_std,-1,left_filters[13])
    l13_right=cv2.filter2D(avg_std,-1,right_filters[13])

    left_conv_bands.append(l13_left)
    right_conv_bands.append(l13_right)

    l14_left=cv2.filter2D(avg_std,-1,left_filters[14])
    l14_right=cv2.filter2D(avg_std,-1,right_filters[14])

    left_conv_bands.append(l14_left)
    right_conv_bands.append(l14_right)

    l15_left=cv2.filter2D(avg_std,-1,left_filters[15])
    l15_right=cv2.filter2D(avg_std,-1,right_filters[15])

    left_conv_bands.append(l15_left)
    right_conv_bands.append(l15_right)
    
    return left_conv_bands,right_conv_bands
@delayed
def perform_thresh(left_conv_band, right_conv_band,shape):
    sleep(1)
    final_conv_band=np.zeros(shape)        
    for i in range(left_conv_band.shape[0]):
        for j in range(left_conv_band.shape[1]):
            if(left_conv_band[i,j]>0 and right_conv_band[i,j]>0):
                if(left_conv_band[i,j]+right_conv_band[i,j]>3):
                    final_conv_band[i,j]=1
                else:
                    final_conv_band[i,j]=0
            else:
                final_conv_band[i,j]=0
    return final_conv_band

@delayed
def compute_cca(final_conv_bands):
    sleep(1)
    cca_bands=[]
    for band in final_conv_bands:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(band, connectivity=8)
        nb_components-=1
        sizes = stats[1:, -1]
        min_size=30
        img = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img[output == i+1] = 255
        cca_bands.append(img)
    return cca_bands

def result_(cca_bands,shape):
    result=np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(cca_bands[0][i,j]==0 and cca_bands[1][i,j]==0 and cca_bands[2][i,j]==0 and cca_bands[3][i,j]==0 and cca_bands[4][i,j]==0 and cca_bands[5][i,j]==0 and cca_bands[6][i,j]==0 and cca_bands[7][i,j]==0 and cca_bands[8][i,j]==0 and cca_bands[9][i,j]==0 and cca_bands[10][i,j]==0 and cca_bands[11][i,j]==0 and cca_bands[12][i,j]==0 and cca_bands[13][i,j]==0 and cca_bands[14][i,j]==0 and cca_bands[15][i,j]==0):
                result[i,j]=1
            else:
                result[i,j]=0
    result = result.astype('uint16')
    return result

def morphological_profile(input,min_area):
    thinning=thin(input,max_iter=(1))
    c1=closing(thinning)
    c2=opening(c1)
    higher_threshold = 0
    dividing = (c2  > higher_threshold)
    smoother_dividing = filters.rank.mean(util.img_as_ubyte(dividing),
                                          morphology.disk(1))
    binary_smoother_dividing = smoother_dividing > 10

    distance = ndi.distance_transform_edt(binary_smoother_dividing)
    
    local_max_coords = feature.peak_local_max(distance, min_distance=3)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    crop_field_segmentation = segmentation.watershed(-distance, markers, mask=binary_smoother_dividing)

    regions=measure.regionprops(crop_field_segmentation)
    mean_area = np.mean([r.area for r in regions])

    filled_filtered = crop_field_segmentation.copy()
    for r in regions:
        if r.area <min_area*mean_area:
            coords = np.array(r.coords).astype(int)
            filled_filtered[coords[:, 0], coords[:, 1]] = 0
    
    return filled_filtered,binary_smoother_dividing

if __name__ == "__main__":
    main()
