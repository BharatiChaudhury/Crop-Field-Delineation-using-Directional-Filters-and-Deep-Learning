# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:48:31 2021

@author: bhara
"""
import os
import geopandas as gpd
import rasterio
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from rasterio.features import rasterize
import matplotlib.pyplot as plt

os.chdir("E:\Crop Field Delineation\Ground Truth Anand University")
train=gpd.read_file("clipped_parcels.shp")
raster=rasterio.open("E:\Crop Field Delineation\Ground Truth Anand University\clipped_input.tif")
raster_meta=raster.meta

def poly_from_utm(polygon, transform):
    poly_pts=[]
    ##make a polygon from multipolygon
    poly=cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform*tuple(i))
    #make a shapely polygon object
    new_poly=Polygon(poly_pts)
    return new_poly

def get_distance(label):
    tlabel=label.astype(np.uint8)
    dist=cv2.distanceTransform(tlabel,cv2.DIST_L2,0)
    dist=cv2.normalize(dist,dist,0,1.0,cv2.NORM_MINMAX)
    return dist
def get_boundary(label,kernel_size=(5,5)):
    tlabel=label.astype(np.uint8)
    temp=cv2.Canny(tlabel,0,1)
    tlabel=cv2.dilate(temp,cv2.getStructuringElement(cv2.MORPH_CROSS,kernel_size),iterations=8)
    tlabel=tlabel.astype(np.float32)
    tlabel/=255
    return tlabel
poly_shp=[]
im_size=(raster.meta['height'],raster.meta['width'])
for num,row in train.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        poly = poly_from_utm(row['geometry'], raster.meta['transform'])
        poly_shp.append(poly)
    else:
        for p in row['geometry']:
            poly=poly_from_utm(p, raster.meta['transform'])
            poly_shp.append(poly)
#prepare field segmentation mask
field_mask=rasterize(shapes=poly_shp,out_shape=im_size,fill=0,all_touched=(True))

##Prepare boundary mask
boundary=gpd.GeoSeries(poly_shp).boundary
boundary_mask=rasterize(shapes=boundary,out_shape=im_size)

transform=raster.transform
dtype=raster.dtypes[0]
boundary_mask=boundary_mask.astype('uint16')
extent_mask=get_distance(field_mask)

print(boundary_mask.shape,extent_mask.shape,field_mask.shape)
plt.imshow(boundary_mask)
plt.show()
plt.imshow(extent_mask)
plt.show()
plt.imshow(field_mask)
plt.show()
np.save("field_mask",field_mask)
np.save("extent_mask", extent_mask)

var=rasterio.open("boundary_mask_india_dilation.tiff",'w',driver='Gtiff',
                 width=boundary_mask.shape[1],
                 height=boundary_mask.shape[0],
                 count=1,
                 transform=transform,
                 dtype=dtype
                 )
var.write(boundary_mask,1)
var.close()
boundary=get_boundary(field_mask)
boundary=boundary.astype('uint16')
boundary_m=rasterio.open("boundary_mask_india_dilation_fn.tiff",'w',driver='Gtiff',
                 width=field_mask.shape[1],
                 height=field_mask.shape[0],
                 count=1,
                 transform=transform,
                 dtype=dtype
                 )
boundary_m.write(boundary,1)
boundary_m.close()


extent_mask=extent_mask.astype('uint16')
extent_m=rasterio.open("extent_mask_india_dilation.tiff",'w',driver='Gtiff',
                 width=extent_mask.shape[1],
                 height=extent_mask.shape[0],
                 count=1,
                 transform=transform,
                 dtype=extent_mask.dtype
                 )
extent_m.write(extent_mask,1)
extent_m.close()
