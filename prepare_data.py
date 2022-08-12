import os
import rasterio
import tensorflow as tf
import geopandas as gpd
import numpy as np



def load_data():
    
    os.chdir("E:\Crop Field Delineation\Ground Truth Anand University")
#shape = rasterio.open("")
    fid_id = gpd.read_file("E:\Crop Field Delineation\Ground Truth Anand University\clipped_parcels.shp")

    inp = rasterio.open("E:\Crop Field Delineation\Ground Truth Anand University\clipped_input.tif")
    inp1 = inp.read()
    

    field_data_1 = rasterio.open("E:\Crop Field Delineation\Ground Truth Anand University\\boundary_mask_india.tiff")
    field_label_1 = field_data_1.read()

    field_data_2 = np.load("field_mask.npy")
    field_data_3 = np.load("extent_mask.npy")

    print(field_label_1.shape,field_data_2.shape, field_data_3.shape)
    
    std_avg = np.load("std_avg.npy")
    left_conv_bands = np.load("left_conv_bands.npy")
    right_conv_bands = np.load("right_conv_bands.npy")

    field_label_2 = field_data_2
    field_label_3 = field_data_3

    data_1 = np.concatenate((left_conv_bands[:,0:500,0:1200],right_conv_bands[:,0:500,0:1200]),axis=0)
    label1_1 = field_label_1[:,0:500,0:1200]
    label2_1 = field_label_2[0:500,0:1200]
    label3_1 = field_label_3[0:500,0:1200]
    std_1 = std_avg[0:500,0:1200]

    data_2 = np.concatenate((left_conv_bands[:,500:800,700:1300],right_conv_bands[:,500:800,700:1300]),axis=0)
    label1_2 = field_label_1[:,500:800,700:1300]
    label2_2 = field_label_2[500:800,700:1300]
    label3_2 = field_label_3[500:800,700:1300]
    std_2 = std_avg[500:800,700:1300]

    data_3 = np.concatenate((left_conv_bands[:,750:1200,1700:2100],right_conv_bands[:,750:1200,1700:2100]),axis=0)
    label1_3 = field_label_1[:,750:1200,1700:2100]
    label2_3 = field_label_2[750:1200, 1700:2100]
    label3_3 = field_label_3[750:1200, 1700:2100]
    std_3 = std_avg[750:1200, 1700:2100]

    data_4 = np.concatenate((left_conv_bands[:,350:606,1967:2223],right_conv_bands[:,350:606,1967:2223]),axis=0)
    label1_4 = field_label_1[:,350:606,1967:2223]
    label2_4 = field_label_2[350:606,1967:2223]
    label3_4 = field_label_3[350:606,1967:2223]
    std_4 = std_avg[350:606, 1967:2223]

    data_5 = np.concatenate((left_conv_bands[:,1200:1600,0:700],right_conv_bands[:,1200:1600,0:700]),axis=0)
    label1_5 = field_label_1[:,1200:1600,0:700]
    label2_5 = field_label_2[1200:1600,0:700]
    label3_5 = field_label_3[1200:1600,0:700]
    std_5 = std_avg[1200:1600,0:700]

# Divide left_conv_filters and right_conv_filters into smaller patches
    inp_data =[]
    label1 = []
    label2 = []
    label3 = []
    std = []
    for i in range(0,data_2.shape[1],50):
        for j in range(0,data_2.shape[2],50):
            if(i + 256 < data_2.shape[1] and j+256 < data_2.shape[2]):
                inp_data.append(data_2[:,i:i+256,j:j+256])
                label1.append(label1_2[:,i:i+256,j:j+256])
                label2.append(label2_2[i:i+256,j:j+256])
                label3.append(label3_2[i:i+256,j:j+256])
                std.append(std_2[i:i+256,j:j+256])
    for i in range(0,data_1.shape[1],50):
        for j in range(0,data_1.shape[2],50):
            if(i + 256 < data_1.shape[1] and j+256 < data_1.shape[2]):
                inp_data.append(data_1[:,i:i+256,j:j+256])
                label1.append(label1_1[:,i:i+256,j:j+256])
                label2.append(label2_1[i:i+256,j:j+256])
                label3.append(label3_1[i:i+256,j:j+256])
                std.append(std_1[i:i+256,j:j+256])
    for i in range(0,data_3.shape[1],50):
        for j in range(0,data_3.shape[2],50):
            if(i + 256 < data_3.shape[1] and j+256 < data_3.shape[2]):
                inp_data.append(data_3[:,i:i+256,j:j+256])
                label1.append(label1_3[:,i:i+256,j:j+256])
                label2.append(label2_3[i:i+256,j:j+256])
                label3.append(label3_3[i:i+256,j:j+256])
                std.append(std_3[i:i+256,j:j+256])
                
    for i in range(0,data_5.shape[1],50):
       for j in range(0,data_5.shape[2],50):
           if(i + 256 < data_5.shape[1] and j+256 < data_5.shape[2]):
               inp_data.append(data_5[:,i:i+256,j:j+256])
               label1.append(label1_5[:,i:i+256,j:j+256])
               label2.append(label2_5[i:i+256,j:j+256])
               label3.append(label3_5[i:i+256,j:j+256])
               std.append(std_5[i:i+256,j:j+256])
               

    inp_data.append(data_4)
    label1.append(label1_4)
    label2.append(label2_4)
    label3.append(label3_4)
    std.append(std_4)

    inp_data = np.array(inp_data)
    label1 = np.array(label1)
    label2 = np.array(label2)
    label3 = np.array(label3)
    std = np.array(std)

    std_ = np.expand_dims(std,axis=3)
    std_.shape
    print(inp_data.shape, label1.shape, label2.shape, label3.shape, std.shape)

    return label1,label2,label3,std_,inp_data
