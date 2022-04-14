import os

### Py10.2 kriging_interpolate
import arcpy
from arcpy.sa import *
import arcgisscripting
import time
import pandas as pd

def CsvToShp(dirPath2, projection, field1):
    arcpy.env.workspace = dirPath2
    filenames=[i for i in os.listdir(dirPath2) if i.endswith("csv")]

    for filename in filenames:
        filePath1 = os.path.join(dirPath2, filename)
        filePath2 = filePath1.replace(".csv", ".shp", 1)
        # dirPath3 = dirPath2.replace("kriging", "kriging_shp", 1)
        if os.path.exists(filePath2):
            print(filename[:-4] + " has been processed...")
            pass
        else:
            print("Processing " + filename[:-4] + "...")
            gp = arcgisscripting.create()
            gp.MakeXYEventLayer_management(filePath1, "longitude",
                                            "latitude", filename[:-4],
                                            projection, field1)
            gp.FeatureClassToShapefile_conversion(filename[:-4], r"E:\aaa\very_shp\shp")

def PointToRaster(dirPath2, field1, extent, name):
    dirPath2 = dirPath2 + "\\" + name
    arcpy.env.workspace = dirPath2
    for dirPath, dirname, filenames in os.walk(dirPath2):
        for filename in filenames:
            if filename[-4:] == ".shp" and "2014" in filename:
                start=time.time()
                filePath1 = os.path.join(dirPath2, filename)
                filePath2 = filePath1.replace("2_shp", name,
                                              1).replace(".shp", ".tif", 1)
                filePath3 = "\\".join(filePath2.split(
                    "\\")[:-2]) + "\\" + name + filePath2.split("\\")[-1]
                if os.path.exists(filePath3):
                    print(filename[0:8] + " has been processed...")
                    pass
                else:
                    print("Processing " + name + filename[0:8] + "...")
                    arcpy.env.extent = extent
                    raster_kriging = Kriging(
                        filePath1, field1,
                        KrigingModelOrdinary("SPHERICAL", 0.371455),
                        0.01)
                    # elevation = os.path.join("E:\\interpolate\\elevation\\",
                                            #  name + ".tif")
                    # raster_kriging.save(filePath3)
                    path_clipshp = extent
                    # kriging = ExtractByMask(raster_kriging, path_clipshp)
                    extent1 = arcpy.Describe(
                        path_clipshp
                    ).extent  # 得到8个 后面又4个NaN 重名会导致运行一个后下一个的默认extent变小导致无法计算
                    arcpy.env.extent = extent1
                    # raster_correct = raster_kriging - 0.0065 * Raster(
                    #     elevation)
                    raster_kriging.save(filePath3)
                end=time.time()
                with open("kriging.txt","a") as o:
                    o.write(str(end-start)+"\n")
def main():
    # df = pd.read_csv(r"E:\aaa\updatePoint.csv")
    # series=["RHU-13003","TEM-12001"]
    # os.chdir(r"E:\aaa\very_shp")
    # for ser in series:
    #     dates=df['date'][df['series']==ser].unique()
    #     for date in dates:
    #         mask = (df['series'] == ser) & (df['date'] == date)
    #         df[mask].to_csv("aus_"+ser[:3]+"_"+str(date)+".csv",index_col=False)
    # df = pd.read_csv(r"E:\aaa\kriging.csv")
    # for ser in series:
    #     # df['date']=df['date'].astype(int)
    #     dates=df['date'][df['series']==ser].unique()
    #     for date in dates:
    #         mask = (df['series'] == ser) & (df['date'] == date)
    #         df[mask].to_csv("kri_"+ser[:3]+"_"+str(date)+".csv",index_col=False)
    CsvToShp(r"E:\aaa\very_shp",r"E:\interpolate\data\0_base\GCS_WGS_1984.prj","dValue")
if __name__ == "__main__":
    # CsvToShp(r"E:\aaa\kriging",r"E:\interpolate\data\0_base\GCS_WGS_1984.prj","HTEMPX")
    # PointToRaster(r"E:\aaa\kriging_shp","HTEMPX",r"E:\aaa\aaa\researchregion.shp","kriging")
    main()
