import os
import numpy as np
from osgeo import gdal
from numpy import ma
hw_times_path=r"E:\研究生毕业论文\数据和处理（原始资料）\热浪指标计算数据\危险性\热浪次数"
hw_times_coefficient=0.1593
hw_times_minmax = [0, 7.86609983444213]
hw_max_HI_path = r"E:\研究生毕业论文\数据和处理（原始资料）\热浪指标计算数据\危险性\最大热浪单日指数"
hw_max_HI_coefficient=0.5889
hw_max_HI_minmax = [0, 40.63876343]


hw_max_duration_path=r"E:\研究生毕业论文\数据和处理（原始资料）\热浪指标计算数据\危险性\最长热浪持续天数"
hw_max_duration_coefficient = 0.2519
hw_max_duration_minmax =[0, 45]

hw_times=[os.path.join(hw_times_path,i) for i in os.listdir(hw_times_path) if i.endswith("tif")]
hw_max_HI=[os.path.join(hw_max_HI_path,i) for i in os.listdir(hw_max_HI_path) if i.endswith("tif")]
hw_max_duration=[os.path.join(hw_max_duration_path,i) for i in os.listdir(hw_max_duration_path) if i.endswith("tif")]

min_dan=0
max_dan=0

hw_dans=[]
ndv=0
for i in range(20):

    dataset = gdal.Open(hw_times[i])
    hw_times_ndv = dataset.GetRasterBand(1).GetNoDataValue()
    hw_times_data = dataset.ReadAsArray(0, 0, dataset.RasterXSize,
                                      dataset.RasterYSize)
    del dataset

    dataset = gdal.Open(hw_max_HI[i])
    hw_max_HI_ndv = dataset.GetRasterBand(1).GetNoDataValue()
    hw_max_HI_data = dataset.ReadAsArray(0, 0, dataset.RasterXSize,
                                         dataset.RasterYSize)
    del dataset

    dataset = gdal.Open(hw_max_duration[i])
    hw_max_duration_ndv = dataset.GetRasterBand(1).GetNoDataValue()
    im_width = dataset.RasterXSize
    im_height=dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    hw_max_duration_data = dataset.ReadAsArray(0, 0, im_width,
                                               )
    del dataset

    hw_times_data = ma.masked_where(hw_times_data == hw_times_ndv,
                                    hw_times_data).filled(np.nan)
    hw_times_data=(hw_times_data-hw_times_minmax[0])/(hw_times_minmax[1]-hw_times_minmax[0])
    hw_max_HI_data = ma.masked_where(hw_max_HI_data == hw_max_HI_ndv,
                                     hw_max_HI_data).filled(np.nan)
    hw_max_HI_data=(hw_max_HI_data-hw_max_HI_minmax[0])/(hw_max_HI_minmax[1]-hw_max_HI_minmax[0])
    hw_max_duration_data = ma.masked_where(
        hw_max_duration_data == hw_max_duration_ndv,
        hw_max_duration_data).filled(np.nan)
    hw_max_duration_data=(hw_max_duration_data-hw_max_duration_minmax[0])/(hw_max_duration_minmax[1]-hw_max_duration_minmax[0])
    hw_dangerous = hw_times_coefficient * hw_times_data + hw_max_HI_coefficient * hw_max_HI_data + hw_max_duration_coefficient * hw_max_duration_data

    if np.nanmin(hw_dangerous)<min_dan:
        min_dan = np.nanmin(hw_dangerous)
    if np.nanmax(hw_dangerous)>max_dan:
        max_dan = np.nanmax(hw_dangerous)
    ndv = hw_max_HI_ndv
    hw_dans.append(hw_dangerous)

for i,hw_dangerous in enumerate(hw_dans):
    year = 2000 + i
    hw_dangerous = (hw_dangerous-min_dan)/(max_dan-min_dan)
    hw_dangerous = ma.masked_where(np.isnan(hw_dangerous),
                                   hw_dangerous).filled(ndv)
    datatype = gdal.GDT_Float32
    filepath = os.path.join(r"E:\研究生毕业论文\数据和处理（原始资料）\热浪指标计算数据\危险性\危险性",str(year)+"_危险性.tif")
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filepath, im_width, im_height, 1, datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        dataset.GetRasterBand(1).SetNoDataValue(ndv)  # 设置nodata值
        dataset.GetRasterBand(1).WriteArray(hw_dangerous)
    del dataset
