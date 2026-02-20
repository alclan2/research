import sys 
sys.path.insert(0, "./")
import matplotlib.pyplot as plt
from region_funcs import generate_regions

fpaths = [
    "annual_tc_counts.nc"
]

da_region, reconstructed = generate_regions(fpaths, nRegions = 10, nIter = 5)

da_region.plot()
plt.savefig("./region_annual_TC.png")
