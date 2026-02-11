import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Polygon

# path to the classified dataset
ClassifiedData = r"C:\Users\allcl\OneDrive\Desktop\desktop\grad school\0. Research\SyCLoPS\dataset\SyCLoPS_classified_ERA5_1940_2024.parquet"

# open the parquet format file (PyArrow package required)
dfc = pd.read_parquet(ClassifiedData)

# select TC and TD LPS nodes and filter QS out of Track_Info
dfc_sub = dfc[((dfc.Short_Label=='TC') | (dfc.Short_Label=='TD')) & ~(dfc['Track_Info'].str.contains('QS', case=False, na=False))]

# make a new column with YEAR only from ISOTIME
dfc_sub["YEAR"] = pd.to_datetime(dfc_sub["ISOTIME"]).dt.year

# pivot by year and count of TCs or TDs
count = dfc_sub.groupby("YEAR").size()

# load the basin shapefile
polygons = []
with open('tc_basins.dat') as f:
    for line_no, line in enumerate(f, start = 1):
        if line.startswith("#"):
            continue
        line = line.strip()
        if not line:
            continue # skip empty lines

        parts = line.split(",")

        basin = parts[0]
        n_vertices = int(parts[1])

        coords_raw = parts[2:]

        # remove empty strings/whitespace
        coords_raw = [x.strip() for x in coords_raw if x.strip()]

        # make sure number of coordinates matches n_vertices
        if len(coords_raw) != n_vertices * 2:
            raise ValueError(
                f"Line {line_no}: {basin} has {len(coords_raw)} coords, "
                f"expected {n_vertices*2}"
            )
        
        # build (lon, lat) tuples for shapely and convert to float safely
        coords = []
        for j in range(0, len(coords_raw), 2):
            try:
                lat = float(coords_raw[j])
                lon = float(coords_raw[j + 1])
                coords.append((lon, lat))
            except IndexError:
                raise ValueError(
                    f"Line {line_no}: {basin} coordinate index out of range at position {j}"
                )
            except ValueError:
                raise ValueError(
                    f"Line {line_no}: {basin} invalid number {coords_raw[j]} or {coords_raw[j+1]}"
                )

        polygons.append({
            "basin name": basin,
            "geometry": Polygon(coords)
        })



# create a GeoDataFrame
basins = gpd.GeoDataFrame(polygons, crs = "EPSG:4326")

# fix invalide polygons
basins["geometry"] = basins["geometry"].buffer(0)

# remove empy geometries
basins = basins[~basins.is_empty]

# project to a CRS with meters for accurate area calc
basins = basins.to_crs(epsg=3857)  # Web Mercator; units in meters

# remove zero area polygons
basins = basins[basins.geometry.area > 0]

# optional: convert back to geographic CRS if needed for plotting/spatial joins
basins = basins.to_crs(epsg=4326)

# filter points
points = gpd.GeoDataFrame(
    dfc_sub, 
    geometry = gpd.points_from_xy(dfc_sub.LON, dfc_sub.LAT),
    crs = "EPSG:4326"
)

filtered = gpd.sjoin(
    points,
    basins[basins["basin name"] == "NE Pacific"],
    how = "inner",
    predicate = "within"
)

# make the scatter plot for all TCs/TDs
plt.figure(figsize = (12,6))
plt.scatter(x = count.index, y = count.values)

# set tick marks to 10 yr intervals
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.6, alpha = 0.5)

# axis labels
plt.xlabel('Year')
plt.ylabel('Number of TCs/TDs')
plt.title('Number of TCs/TDs in NE Pacific from 1940-2024')

plt.show()