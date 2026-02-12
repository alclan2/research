import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# path to the classified dataset
ClassifiedData = r"C:\Users\allcl\OneDrive\Desktop\desktop\grad school\0. Research\SyCLoPS\dataset\SyCLoPS_classified_ERA5_1940_2024.parquet"

# open the parquet format file (PyArrow package required)
dfc = pd.read_parquet(ClassifiedData)

# select TC and TD LPS nodes and filter QS out of Track_Info
dfc_sub = dfc[((dfc.Short_Label=='TC') | (dfc.Short_Label=='TD')) & ~(dfc['Track_Info'].str.contains('QS', case=False, na=False))]

# set index to ISOTIME column by year
dfc_sub = dfc_sub.set_index("ISOTIME")
dfc_sub['YEAR'] = dfc_sub.index.year

# create 4deg bins
dfc_sub['lat_bin'] = 4 * np.floor(dfc_sub["LAT"] / 4)
dfc_sub['lon_bin'] = 4 * np.floor(dfc_sub["LON"] / 4)

# calc number TCs per bin
annual_TC_counts = dfc_sub.groupby(["YEAR", "lat_bin", "lon_bin"]).size().reset_index(name="n_cyclones")

# give each line an ID
annual_TC_counts["cell_id"] = list(zip(annual_TC_counts["lat_bin"], annual_TC_counts["lon_bin"]))

# pivot to get format: rows=year, columns=grid cells
wide_counts = annual_TC_counts.pivot(index="YEAR", columns="cell_id", values="n_cyclones").fillna(0)

# convert to array
arr = wide_counts.values
#print(arr.shape)







# histogram check
#plt.hist(arr.flatten(), bins=range(int(arr.max())+2), edgecolor='k')
#plt.xlabel("Number of cyclones per year per cell")
#plt.ylabel("Frequency")
#plt.title("Distribution of cyclone counts across cells and years")
#plt.show()

max_cyclones = arr.max()
print("Maximum number of cyclones in any year-cell:", max_cyclones)

# get indices of max values
row_idx, col_idx = np.where(arr == max_cyclones)

# print the results
for r, c in zip(row_idx, col_idx):
    print(f"Year: {wide_counts.index[r]}, Cell: {wide_counts.columns[c]}, Cyclones: {arr[r, c]}")

N = 5  # top 5
flat_indices = np.argsort(arr.flatten())[::-1][:N]  # indices of top N values
rows, cols = np.unravel_index(flat_indices, arr.shape)

for r, c in zip(rows, cols):
    print(f"Year: {wide_counts.index[r]}, Cell: {wide_counts.columns[c]}, Cyclones: {arr[r, c]}")