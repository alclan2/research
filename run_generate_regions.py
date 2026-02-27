import sys 
sys.path.insert(0, "./")
import matplotlib.pyplot as plt
from region_funcs import generate_regions
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker

fpaths = [
    "annual_tc_counts_anomaly_NAtlantic_4deg.nc"
]

da_region, reconstructed = generate_regions(fpaths, nRegions = 6, nIter = 50)

# create map with projection of continents
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

#da_region.plot()
da_region.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),   # tells cartopy data are lat/lon
    cmap="viridis",
    add_colorbar=True
)

# add axis labels
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.8,
    color='gray',
    alpha=0,
)

# keep labels only on left and bottom
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.MultipleLocator(20)
gl.ylocator = mticker.MultipleLocator(10)

# add continents and coastlines
ax.coastlines()

# format and save
plt.title("TC Count Anomaly in North Atlantic 1940-2024 (3 regions, 4deg grid)")
#plt.savefig("./images/region_generation/region_annual_TC_anomaly_NAtlantic_6regions_16deg.png")
plt.show()