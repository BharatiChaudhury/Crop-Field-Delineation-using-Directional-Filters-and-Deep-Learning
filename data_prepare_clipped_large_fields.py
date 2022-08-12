import rasterio
import numpy as np
import matplotlib.pyplot as plt
from boundary_delineation_india import calculate_std,avg_std,directional_filters
from dask import delayed


def prepare_directional_filters_std(data):
    data = rasterio.open(data)
    nir_bands = data.read()
    #nir_bands = nir_bands[:,0:1536,0:2048] ##for large input
    nir_bands = nir_bands[:,0:256,0:256]
    shape=nir_bands[0].shape
    std_bands =[]
    for band in nir_bands:
        results = delayed (calculate_std) (band,shape)
        results = results.compute()
        std_bands.append(results)
        #print(std_bands)
    std_avg = delayed(avg_std)(std_bands, shape)
    std_avg = std_avg.compute()
    plt.imshow(std_avg)
    final_conv_bands=[]
    left_conv_bands,right_conv_bands = directional_filters(std_avg)
    return std_avg, left_conv_bands, right_conv_bands
