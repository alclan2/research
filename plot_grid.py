import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# plots to use
image_paths = [
    r"images\region_generation\region_annual_TC_NAtlantic_3regions_4deg.png", 
    r"images\region_generation\region_annual_TC_NAtlantic_4regions_4deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_5regions_4deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_6regions_4deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_3regions_8deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_4regions_8deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_5regions_8deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_6regions_8deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_3regions_12deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_4regions_12deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_5regions_12deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_6regions_12deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_3regions_16deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_4regions_16deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_5regions_16deg.png",
    r"images\region_generation\region_annual_TC_NAtlantic_6regions_16deg.png"
]

# create 4x4 grid of plots
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for ax, img_path in zip(axes.flat, image_paths):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
#plt.savefig("combined_4x4.png", dpi=300)
plt.show()