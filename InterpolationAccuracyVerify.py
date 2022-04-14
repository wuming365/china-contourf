from osgeo import gdal
import os
import numpy as np
import pandas as pd
from icecream import ic
from pandas.core.dtypes.missing import notnull
from tqdm import tqdm
dateAndMethod = {}
veriationStationValues = {}


def getGlobalConstant():
    global dateAndMethod
    global veriationStationValues
    a = []
    veriationPath = "../文本_验证点提取/验证点/"
    with open("../文本_验证点提取/log.txt", "r") as r:
        dateAndMethod = pd.DataFrame(np.array(
            [i.split() for i in np.array(r.readlines())]),
                                     columns=["date", "rhu", "tem", "method"])
    filenames = [i for i in os.listdir(veriationPath) if i.endswith("txt")]
    with tqdm(filenames) as t:
        for filename in t:
            t.set_description_str("获取验证点信息：")
            date = filename.split(".")[0].split("_")[-1]
            method = filename.split(".")[0].split("_")[1]
            series = filename.split(".")[0].split("_")[0]
            filePath = os.path.join(veriationPath, filename)
            with open(filePath, "r") as r:
                rd = r.readlines()
                listDate = [date] * len(rd)
                listMethod = [method] * len(rd)
                listSeries = [series] * len(rd)
                b = np.array([i.split() for i in np.array(rd)])
                b = np.c_[b, listDate, listMethod, listSeries]

                if (len(a) != 0):
                    a = np.r_[a, b]
                else:
                    a = b
    veriationStationValues = pd.DataFrame(
        a,
        columns=[
            "id", "longitude", "latitude", "elevation", "tValue", "date",
            "method", "series"
        ],
    )
    veriationStationValues["longitude"] = veriationStationValues[
        "longitude"].astype("float")
    veriationStationValues["latitude"] = veriationStationValues[
        "latitude"].astype("float")
    veriationStationValues["elevation"] = veriationStationValues[
        "elevation"].astype("float")
    veriationStationValues["tValue"] = veriationStationValues["tValue"].astype(
        "float")


def openSingleImage(filename: str):
    """打开影像

    Args:
        filename (str): 文件路径

    Returns:
        im_data: array,数据数组
        im_proj: str,坐标系
        im_geotrans: tuple,仿射矩阵
        im_height,im_width: float,图像高和宽 
    """

    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_band = dataset.GetRasterBand(1)
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)
    return im_data, im_proj, im_geotrans, im_height, im_width


def getPointsData(filename, stations):
    img, _, geotrans, height, width = openSingleImage(filename)
    xOrigin = geotrans[0]
    yOrigin = geotrans[3]
    pixelWidth = geotrans[1]
    stations["tifX"] = (stations["longitude"] - xOrigin) / pixelWidth
    stations["tifY"] = (yOrigin - stations["latitude"]) / pixelWidth
    stations["tifX"] = stations["tifX"].astype("int")
    stations["tifY"] = stations["tifY"].astype("int")
    stations["vValue"] = -9999
    for i in stations.index:
        stations.loc[i, "vValue"] = img[stations.loc[i, "tifY"]][stations.loc[
            i, "tifX"]]
    return stations

def main():
    imgPath = r"I:\dem_chazhi\output"
    getGlobalConstant()
    series = ["RHU-13003", "TEM-12001"]
    alldata = []
    for i in tqdm(series):
        for j in tqdm(dateAndMethod.index):
            if j != 0:
                stations = veriationStationValues[
                    (veriationStationValues["date"] == dateAndMethod.loc[
                        j, "date"]) & (veriationStationValues["method"] ==
                                       dateAndMethod.loc[j, "method"]) &
                    (veriationStationValues["series"] == i)]
                stations = getPointsData(
                    os.path.join(imgPath,
                                 f"{i}_{dateAndMethod.loc[j, 'date']}.tif"),
                    stations)
                if len(alldata) == 0:
                    alldata = stations
                else:
                    alldata = pd.concat([alldata, stations],
                                        axis=0,
                                        ignore_index=True)
    alldata.to_csv("../文本_验证点提取/updatePoint.csv", encoding='utf-8')
def veri_kriging():
    imgPath = r"E:\aaa\kriging_shp"
    os.chdir(imgPath)
    df=pd.read_csv(r"E:\aaa\kriging.csv")
    series=["RHU-13003","TEM-12001"]
    for ser in series:
        dates=df['date'][df['series']==ser].unique()
        for date in dates:
            mask = (df['series'] == ser) & (df['date'] == date)
            list_tifX = np.array(df[mask]['tifX'])
            list_tifY = np.array(df[mask]['tifY'])
            vValue = np.zeros_like(list_tifX,dtype=np.float32)
            img = openSingleImage("kriging" + str(date) + ".tif")[0]
            for i in range(len(list_tifX)):
                vValue[i]=img[list_tifY[i]][list_tifX[i]]
            df.loc[mask,'vValue']=vValue
    df.to_csv(r"E:\aaa\kriging.csv", index=False)

import os
import pandas as pd
def renamefile():
    csv_path=r"E:\aaa\very_shp"
    files=[i for i in os.listdir(csv_path) if i.endswith("csv")]
    for file in files:
        old_name=os.path.join(csv_path,file)
        df = pd.read_csv(old_name)
        method=df['method'][1]
        new_name=os.path.join(csv_path,file.replace("aus",method,1))
        del df
        os.rename(old_name,new_name)
if __name__ == "__main__":
    renamefile()
