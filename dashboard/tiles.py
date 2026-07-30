import io
import math
import os
import pygame
import urllib.request

# Creates static satellite backdrop for the dashboard.
# Fetches Esri World Imagery tiles (standard XYZ web mercator, 256px, no API Key),
# stitches the tiles covering the coverage footprint, and crops to the exact pixel window for
# a 'cov_w_m' x 'cov_h_m' box centered on the fix given by the GPS sensor. Touches the network only at
# initial startup time and then works fully offline.

ESRI_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")
TILE = 256
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tilecache")

# Web Mercator ground resolution, meters per pixel, at the given latitude/zoom
def _mpp(lat, z):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)

# Convert latitude/longitude to pixel format
def _latlon_to_global_px(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * TILE
    lat_r = math.radians(lat)
    y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n * TILE
    return x, y

# Return a pygame.Surface of the satellite imagery cropped to the exact coverage footprint centered
# at (lat, lon)
def footprint_backdrop(lat, lon, cov_w_m, cov_h_m, z=18):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{lat:.6f}_{lon:.6f}_{cov_w_m:.0f}x{cov_h_m:.0f}_z{z}.png"
    cache_path = os.path.join(CACHE_DIR, key)
    if os.path.exists(cache_path):
        return pygame.image.load(cache_path)

    # Exact crop window in global pixel space
    m = _mpp(lat, z)
    crop_w = int(round(cov_w_m / m))
    crop_h = int(round(cov_h_m / m))
    cx, cy = _latlon_to_global_px(lat, lon, z)
    left = cx - crop_w / 2
    top = cy - crop_h / 2
    right = left + crop_w
    bottom = top + crop_h

    # Which tiles cover that window
    tx0 = int(math.floor(left / TILE)) - 1
    ty0 = int(math.floor(top / TILE)) - 1
    tx1 = int(math.floor((right - 1e-6) / TILE)) + 1
    ty1 = int(math.floor((bottom - 1e-6) / TILE)) + 1
    stitched = pygame.Surface(((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            url = ESRI_URL.format(z=z, x=tx, y=ty)
            req = urllib.request.Request(url, headers={"User-Agent" : "drone-dashboard/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            tile_img = pygame.image.load(io.BytesIO(data))
            stitched.blit(tile_img, ((tx - tx0) * TILE, (ty - ty0) * TILE))

    # Crop stitched image to the exact footprint window
    off_x = int(round(left - tx0 * TILE))
    off_y = int(round(top - ty0 * TILE))
    crop = pygame.Surface((crop_w, crop_h))
    crop.blit(stitched, (0, 0), (off_x, off_y, crop_w, crop_h))

    pygame.image.save(crop, cache_path)
    return crop

# Offline sanity check of the geometry (no pygame, no network)
if __name__ == "__main__":
    lat, lon = 33.499177, -117.706882
    z = 18
    m = _mpp(lat, z)
    cw, ch = round(285 / m), round (174 / m)
    cx, cy = _latlon_to_global_px(lat, lon, z)
    print(f"z={z} m/px={m:.4f} crop={cw}x{ch}px center_global=({cx:.1f},{cy:.1f})")
    left, top = cx - cw / 2, cy - ch / 2
    tx0, ty0 = int(left // TILE), int(top // TILE)
    tx1, ty1 = int((left + cw) // TILE), int((top + ch) // TILE)
    print(f"tilex x[{tx0}..{tx1}] y[{ty0}..{ty1}] = {(tx1-tx0+1)*(ty1-ty0+1)} tiles")
    print(f"crop offset in stitched = ({left - tx0*TILE:.1f},{top - ty0*TILE:.1f})")
