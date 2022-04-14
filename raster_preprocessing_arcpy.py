import arcpy
import os
import time
arcpy.env.workspace = "I:/dem_chazhi/output"
rasters = arcpy.ListRasters("*","tif")
# print(rasters)
mask = "I:/dem_chazhi/output/clip/clip.shp"
i=0
start_time=time.time()
for raster in rasters:
    out = "I:/dem_chazhi/turn/"+"clip_"+raster#裁剪后数据位置和数据名称设定
    if not os.path.exists(out) and not os.path.exists("I:/dem_chazhi/result/"+raster):
        arcpy.gp.ExtractByMask_sa(raster,mask,out)
        if "RHU" in raster:
            arcpy.gp.RasterCalculator_sa('Con("'+out+'" > 100,100,"'+out+'")', "I:/dem_chazhi/result/"+raster)
        else:
            os.rename(out,"I:/dem_chazhi/result/"+raster)
    i+=1
    end_time=time.time()
    min=(end_time-start_time)/60
    print("Done:{}/{}, Time:{:.2f}min, Average:{:.2f}s".format(i,len(rasters),min,min*60/i))
