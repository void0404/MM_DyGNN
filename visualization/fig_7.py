import fiona
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import mapclassify as mc

# ----------------------- 1. Read data & project -----------------------
file_path = './cross_map_235.geojson' # the path to your GeoJSON file

with fiona.open(file_path) as src:
    transformer = Transformer.from_crs(src.crs, 'EPSG:3857', always_xy=True)
    geoms, top_vals = [], []
    for feat in src:
        geom = shape(feat['geometry'])
        geoms.append(transform(transformer.transform, geom))
        top_vals.append(feat['properties']['235col_m_msf'])

values = np.array(top_vals).reshape(-1, 1)

# ----------------------- 2. Jenks Natural Breaks ---------------------
k = 6
classifier = mc.NaturalBreaks(values.flatten(), k=k)
cluster_ranks = classifier.yb            # 0‥k‑1
breaks = classifier.bins                 # k‑1 upper bounds

# ----------------------- 3. Generate labels (first class lower bound = 0) ------------
full_breaks = [values.min()] + breaks.tolist()   # left-closed, right-open

labels = []
for idx, (lo, hi) in enumerate(zip(full_breaks[:-1], full_breaks[1:])):
    if idx == 0:                       # First class
        labels.append(f"0.0000–{hi:.4f}")
    else:                              # Other classes
        labels.append(f"{lo:.4f}–{hi:.4f}")

# ----------------------- 4. Colors -----------------------------------
cmap = plt.cm.get_cmap('Reds', k)
colors = [cmap(i) for i in range(k)]

# ----------------------- 5. Bounds -----------------------------------
bounds = np.array([g.bounds for g in geoms])
minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()

# ----------------------- 6. Plotting ---------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

for geom, rank_idx in zip(geoms, cluster_ranks):
    fc = colors[int(rank_idx)]
    if geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            ax.fill(*poly.exterior.xy, facecolor=fc,
                    edgecolor='white', linewidth=0.4)
    else:
        ax.fill(*geom.exterior.xy, facecolor=fc,
                edgecolor='white', linewidth=0.4)

# ----------------------- 7. Legend -----------------------------------
handles = [
    Patch(facecolor=colors[i], edgecolor='white', label=labels[i])
    for i in range(k)
]
ax.legend(
    handles=handles,
    title='Attention Weight',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=3, fontsize=9, title_fontsize=10
)

# ----------------------- 8. Scale bar -------------------------------
scalebar = AnchoredSizeBar(ax.transData, 10000, '10 km',
                           loc='lower center', pad=0.5,
                           color='black', frameon=False, size_vertical=1)
ax.add_artist(scalebar)

# ----------------------- 9. Final adjustments -----------------------
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_aspect('equal')
ax.margins(0)
plt.tight_layout(pad=0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_axis_off()

# ----------------------- 10. Save & show ----------------------------
plt.savefig('./{outputpath}',
            format='svg', bbox_inches='tight', pad_inches=0)
plt.show()
