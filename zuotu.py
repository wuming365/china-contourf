from osgeo import gdal
import numpy as np
import os
from tqdm import tqdm
import numpy.ma as ma
import shapefile
import pandas as pd


def openSingleImage(filename: str):
    """打开影像

    Args:
        filename (str): 文件路径

    Returns:
        im_data: array,数据数组
        im_proj: str,坐标系
        im_geotrans: tuple,仿射矩阵
        im_height,im_width: float,图像高和宽
        ndv: float,NoDataValue
    """

    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_band = dataset.GetRasterBand(1)
    ndv = im_band.GetNoDataValue()
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_data, im_proj, im_geotrans, im_height, im_width, ndv


def openImages(filenames):
    """
    打开工作文件夹中的所有影像
    dirpath:数据读取文件夹
    return:Igs,im_geotrans.im_proj,ndv
    """
    Igs = []
    idate = 0
    with tqdm(filenames) as t:
        for filename in t:
            if filename[-4:] == ".tif":
                Image, im_proj, im_geotrans, _, _, ndv = openSingleImage(
                    filename)
                Igs.append(Image)
                idate = idate + 1
                t.set_description(filename + " is already open……")
    return np.array(Igs), im_geotrans, im_proj, ndv


def write_img(im_data, filename, im_proj, im_geotrans, dirpath, ndv):
    """写影像"""
    # im_data被写的影像
    # im_proj, im_geotrans均为被写影像参数
    # filename创建新影像的名字，dirpath影像写入文件夹
    # 判断栅格数据类型
    datatype = gdal.GDT_Float32
    fullpath = dirpath + "\\" + filename
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_bands = 1  # 均为单波段影像
        im_height, im_width = im_data.shape
        im_data = [im_data]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fullpath, im_width, im_height, im_bands, datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(ndv)  # 设置nodata值
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    else:
        os.remove(fullpath)
        with open(r"I:/dem_chazhi/log/error.txt", "a") as o:
            o.write(fullpath + "\r")
    del dataset
    del im_data


# # #求平均
# # index = [
# #     "热浪次数(mean)_1", "热浪次数(mean)_2", "热浪次数(mean)_3", "热浪次数(mean)", "热浪日数(3)_2",
# #     "热浪日数(3)_1",
# #     "热浪日数(3)_3", "热浪日数(3)", "最长热浪持续日数", "最大单日热浪指数(3)", "热浪开始日期(3)",
# #     "热浪结束日期(3)", "最大热浪HI指数极差(3)"
# # ]
# # for i in tqdm(index):
# #     filenames = [
# #         fr"G:\dem_chazhi\result\热浪不同指标结果\{year}_{i}.tif"
# #         for year in range(2000, 2020)
# #     ]
# #     Igs, im_geotrans, im_proj, ndv = openImages(filenames)
# #     Igs = ma.masked_equal(Igs, ndv)
# #     meanIgs = np.mean(Igs, axis=0)
# #     meanIgs = meanIgs.filled(ndv)
# #     write_img(meanIgs, f"{i}.tif", im_proj, im_geotrans,
# #               r"G:\dem_chazhi\result\mean", ndv)


def mkdir(path):
    """
    创建文件夹
    """
    folder = os.path.exists(path)
    foldername = path.split("\\")[-1]

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("The folder " + foldername + " is created")


def clipTiff(path_inputraster, path_outputraster, path_clipshp, NDV):
    """
    裁剪影像
    path_inputraster:str
    path_outputraster:str
    path_clipshp:str
    """
    input_raster = gdal.Open(path_inputraster)

    # 两个投影一样
    r = shapefile.Reader(path_clipshp)
    ds = gdal.Warp(path_outputraster,
                   input_raster,
                   format='GTiff',
                   outputBounds=r.bbox,
                   cutlineDSName=path_clipshp,
                   dstNodata=NDV)
    ds = None


# # #裁剪
# # shp_path = r"E:\研究生毕业论文\数据和处理（原始资料）\审图号 GS（2019）1719 号地图的底图数据\中国省级地图GS（2019）1719号\十五城市"
# # raster_path = r"G:\dem_chazhi\result\热浪不同指标结果"
# # index = [
# #     "热浪次数(mean)_1", "热浪次数(mean)_2", "热浪次数(mean)_3", "热浪次数(mean)", "热浪日数(3)_2",
# #     "热浪日数(3)_1", "热浪日数(3)_3", "热浪日数(3)", "最长热浪持续日数", "最大单日热浪指数(3)",
# #     "热浪开始日期(3)", "热浪结束日期(3)", "最大热浪HI指数极差(3)"
# # ]
# # out_path = r"G:\dem_chazhi\result\不同城市热浪指标"
# # shps = [
# #     os.path.join(shp_path, i) for i in os.listdir(shp_path)
# #     if i.endswith("shp")
# # ]

# # for shp in tqdm(shps):
# #     region = os.path.basename(shp[:-4])
# #     mkdir(os.path.join(out_path, region))
# #     for i in index:
# #         ndv = openSingleImage(os.path.join(raster_path, f"2000_{i}.tif"))[5]
# #         for year in range(2000, 2020):
# #             raster = os.path.join(raster_path, f"{year}_{i}.tif")
# #             out_raster = os.path.join(os.path.join(out_path, region),
# #                                       f"{region}_{year}_{i}.tif")
# #             if not os.path.exists(out_raster):
# #                 clipTiff(raster, out_raster, shp, ndv)

# # #统计
# # indexs = {
# #     "热浪次数(mean)_1": "HWF_1",
# #     "热浪次数(mean)_2": "HWF_2",
# #     "热浪次数(mean)_3": "HWF_3",
# #     "热浪次数(mean)": "HWF",
# #     "热浪日数(3)_1": "HWTD_1",
# #     "热浪日数(3)_2": "HWTD_2",
# #     "热浪日数(3)_3": "HWTD_3",
# #     "热浪日数(3)": "HWTD",
# #     "最长热浪持续日数": "HWMD",
# #     "最大单日热浪指数(3)": "HWMHI",
# #     "热浪开始日期(3)": "HWSD",
# #     "热浪结束日期(3)": "HWED",
# #     "最大热浪HI指数极差(3)": "HWMRHI",
# # }
# # in_path = r"G:\dem_chazhi\result\不同城市热浪指标"
# # df = pd.DataFrame(columns=("region", "year", "index_ch", "index_en", "data",
# #                            "max", "min"))
# # for dirpath, dirnames, filenames in os.walk(in_path):
# #     if len(filenames) != 0:
# #         for filename in filenames:
# #             file_path = os.path.join(dirpath, filename)
# #             region = filename.split("_")[0]
# #             year = filename.split("_")[1]
# #             list_name = filename.split("_")
# #             if len(list_name) == 3:
# #                 index_ch = list_name[-1].split(".")[0]
# #             else:
# #                 index_ch = "_".join(list_name[-2:]).split(".")[0]
# #             index_en = indexs[index_ch]
# #             im_data, _, _, _, _, ndv = openSingleImage(file_path)
# #             if np.max(im_data) == ndv and np.min(im_data) == ndv:
# #                 df = df.append(
# #                     {
# #                         "region": region,
# #                         "year": year,
# #                         "index_ch": index_ch,
# #                         "index_en": index_en,
# #                     },
# #                     ignore_index=True)
# #             else:
# #                 im_data_1 = ma.masked_equal(im_data, ndv)
# #                 data = np.nanmean(im_data_1)
# #                 max_data = np.nanmax(im_data_1)
# #                 min_data = np.nanmin(im_data_1)
# #                 df = df.append(
# #                     {
# #                         "region": region,
# #                         "year": year,
# #                         "index_ch": index_ch,
# #                         "index_en": index_en,
# #                         "data": data,
# #                         "max": max_data,
# #                         "min": min_data,
# #                     },
# #                     ignore_index=True)
# # df.to_csv(r"E:\高温热浪危险性论文\表格\统计_城市.csv",
# #           na_rep='NA',
# #           index=False,
# #           encoding="gbk")

# #检查并统计
import scipy.stats as st
from tqdm import tqdm, trange
# # in_path = r"E:\高温热浪危险性论文\表格\统计_城市.csv"
# # df = pd.read_csv(in_path, encoding="gbk")
# # df1 = pd.DataFrame()
# # for index_en in df["index_en"].unique():
# #     for region in df["region"].unique():
# #         df2 = df[(df["index_en"] == index_en) & (df["region"] == region)]
# #         data = np.array(df2["data"])
# #         if index_en != "HWSD":
# #             data = np.delete(data, np.where(data == 0))
# #         else:
# #             data = np.delete(data, np.where(data == "NA"))
# #         if len(data) == 0:
# #             slope = "NA"
# #             r_value = "NA"
# #             p_value = "NA"
# #         else:
# #             slope, intercept, r_value, p_value, std_err = st.linregress(
# #                 list(range(len(data))), data)
# #         df1 = df1.append(
# #             {
# #                 "region": region,
# #                 "index_en": index_en,
# #                 "slope": slope,
# #                 "r_value": r_value,
# #                 "p_value": p_value,
# #             },
# #             ignore_index=True)
# # df1.to_csv(r"E:\高温热浪危险性论文\表格\城市变化.csv", encoding="gbk", index=False)

# # # #热浪危险性的slope
# in_path = r"G:\dem_chazhi\result\hazard"
# # # indexs = ["热浪次数(mean)", "最大单日热浪指数(3)", "最长热浪持续日数"]
# year_datas = []
# ndv = -32768.0
# for year in range(1990, 2020):
#     im_datas = []
#     im_data, im_proj, im_geotrans, im_height, im_width, ndv1 = openSingleImage(
#         os.path.join(in_path,
#                      str(year) + "_hazard.tif"))
#     im_data = ma.masked_equal(im_data, ndv1).filled(ndv)
#     year_datas.append(im_data)
# year_datas = np.transpose(year_datas)
# (width, height, length) = year_datas.shape
# slope_data = np.empty((height, width), dtype=np.float32)
# p_data = np.zeros((height, width), dtype=np.int32)
# for i in trange(width):
#     for j in range(height):
#         y = year_datas[i][j]
#         if ndv not in y:
#             x = list(range(length))
#             slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
#             slope_data[j][i] = slope
#             if p_value < 0.05:
#                 p_data[j][i] = 1
# slope_data = ma.masked_where(year_datas.transpose()[0] == ndv,
#                              slope_data).filled(ndv)
# write_img(slope_data, "slope_hazard.tif", im_proj, im_geotrans, in_path, ndv)
# write_img(p_data, "p_value_hazard.tif", im_proj, im_geotrans, in_path, 0)

# import operator

# path = r"G:\dem_chazhi\result\危险性"
# hazard = os.path.join(path, "slope_危险性.tif")
# p = os.path.join(path, "p_value_危险性.tif")
# im_data, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(p)
# hazard1, _, _, _, _, ndv1 = openSingleImage(hazard)
# im_data = im_data.astype(np.bool8)
# im_data = ~im_data
# im_data = ma.masked_where(hazard1 == ndv1,
#                           im_data.astype(np.int32)).filled(ndv)
# write_img(im_data, "p_value_危险性1.tif", im_proj, im_geotrans, path, ndv)


