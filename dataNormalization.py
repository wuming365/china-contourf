"""
在获取完全部的气象站点后，重新遍历所有下载的数据，需要确定年月以便确定日，如果是月数据，那需要根据文件名更改部分代码，并且只需要提供年，月信息可以注释掉（略微复杂）。如果插值的数据类型有好多，可以在指标列表中逐一添加
"""

from tqdm import tqdm
from tqdm import trange
import os
from icecream import ic
from decimal import Decimal
from osgeo import osr
import csv
from multiprocessing import Process


def proj(
    x,
    y,
):
    geosrs = osr.SpatialReference()
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(
        'PROJCS["Asia_North_Albers_Equal_Area_Conic",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",0],PARAMETER["longitude_of_center",105],PARAMETER["standard_parallel_1",25],PARAMETER["standard_parallel_2",47],PARAMETER["false_easting",4000000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    )
    #这个是为了将地理坐标系投影，使用python读取投影文件的投影信息然后复制到这里
    geosrs.SetWellKnownGeogCS("WGS84")
    ct = osr.CoordinateTransformation(geosrs, prosrs)

    coords = ct.TransformPoint(x, y, 0)
    return [coords[0], coords[1]]


def getTheNumberOfDaysInTheMonth(year, month):
    """
    根据年份，月份信息显示此月份天数
    :param year: 年份：
    :param month: 月份（1～12）：
    :return: 当月天数
    """
    if month > 12 or month <= 0:
        return -1
    if month == 2:
        return 29 if year % 4 == 0 and year % 100 != 0 or year % 400 == 0 else 28

    if month in (4, 6, 9, 11):
        return 30
    else:
        return 31


def createOutputFile(outputDirectory, Indexs, timeIntervalYear,
                     timeIntervalMonth):
    for index in Indexs:
        for i in range(timeIntervalYear[1] - timeIntervalYear[0] + 1):
            year = timeIntervalYear[0] + i
            for j in range(timeIntervalMonth[1] - timeIntervalMonth[0] + 1):
                month = timeIntervalMonth[0] + j
                days = getTheNumberOfDaysInTheMonth(year, month)
                for day in range(days):
                    filePath = f"{outputDirectory}\\{index}_{year}{str(month).zfill(2)}{str(day+1).zfill(2)}.txt"
                    open(filePath, 'w')


def main(inputDirectory, outputDirectory, station, stationInfo):
    for dirpath, _, filenames in os.walk(outputDirectory):  #先循环打开一个输出文件
        try:
            with tqdm(filenames) as t:
                for filename in t:
                    size = os.path.getsize(os.path.join(dirpath, filename))
                    with open(os.path.join(dirpath, filename), "w") as oo:
                        basename = filename.split('.')[0]  #没有后缀的名字
                        index = basename.split('_')[0]  #指标
                        month = basename.split('_')[1][:-2]  #月
                        day = basename.split('_')[1][-2:]  #日，根据文件名特点提取的
                        inputFilePath = f"{inputDirectory}\\SURF_CLI_CHN_MUL_DAY-{index}-{month}.txt"  #根据文件名特点自行修改

                        with open(inputFilePath, "r") as oi:
                            for line in oi.readlines():
                                lineStr = line.split()
                                dayinput = lineStr[6]
                                if lineStr[0] in station:
                                    index = station.index(lineStr[0])
                                    latitude = format(
                                        float(stationInfo[index][0]),
                                        '15.6f')  #纬度
                                    longitude = format(
                                        float(stationInfo[index][1]),
                                        '15.6f')  #经度
                                    elevation = format(
                                        float(stationInfo[index][3]),
                                        '10.2f')  #高程
                                    value=32766
                                    if "TEM-12001" in filename:
                                        if lineStr[8] != "32766":
                                            value = format(
                                                float(lineStr[8]) * 0.1,
                                                '10.2f'
                                            )  #第8列是平均值 但温度该用日最高温度即第九列
                                    elif "RHU-13003" in filename:
                                        if lineStr[7] != "32766":
                                            value = format(
                                                float(lineStr[7]), '10.2f'
                                            )  #第8列是平均值，如果需要别的列，可根据网页上元数据列数进行修改
                                    
                                    #如果有其他数据，可在与指标列表信息保持一致的情况下自行添加elif，并取值为value
                                    if value!=32766:
                                        string = f"{lineStr[0]}{longitude}{latitude}{elevation}{value}\n"
                                        if int(day) == int(dayinput):
                                            # ic(lineStr)
                                            oo.write(string)
        except:
            continue


def test(testTxt):

    with open(testTxt, "r") as t:
        for line in t.readlines():
            lineStr = line.split()
            print(lineStr[4])


if __name__ == '__main__':
    testTxt = "test.txt"
    inputDirectory = r"E:\研究生毕业论文\数据和处理（原始资料）\中国气象站点数据_from国家气象数据中心\温度和相对湿度_待处理"  #
    outputDirectory = r"E:\研究生毕业论文\实验部分\1990-2020文本数据"
    Indexs = ["RHU-13003", "TEM-12001"]  #将指标逐一添加到这个列表中
    timeIntervalYear = [1990, 2019]
    timeIntervalMonth = [5, 9]
    year = 1990
    process_list = []
    station = []
    stationInfo = []
    with open("./站点_使用.txt", "r") as ooooo:  #经arcmap dem提取后剩余的点存到txt里
        for line in ooooo.readlines():
            lineStr = line.split()
            station.append(lineStr[0])
            stationInfo.append(lineStr[1:])

    # for i in range(5):
    #     year+=i*6
    #     p=Process(target=main,args=(inputDirectory,outputDirectory,year,station,stationInfo))
    #     p.start()
    #     process_list.append(p)

    # for p in process_list:
    #     p.join()

    createOutputFile(
        outputDirectory, Indexs, timeIntervalYear,
        timeIntervalMonth)  #先创建所需要的文件，如果是月度数据，根据需要修改函数，将timeIntervalMonth注释掉
    main(inputDirectory, outputDirectory, station, stationInfo)
    # test(testTxt)