import rasterio
from rasterio.warp import reproject, Resampling

input_mask  = r'C:\Users\Nikola\Desktop\projekat_is\is_park_segmentation\raw_novi\frankfurt_maska.tif'
output_mask = r'C:\Users\Nikola\Desktop\projekat_is\is_park_segmentation\raw_novi\frankfurt_mask.tif'
reference   = r'C:\Users\Nikola\Desktop\projekat_is\is_park_segmentation\raw_novi\frankfurt.tiff'  # satelitska slika

with rasterio.open(reference) as ref:
    target_crs       = ref.crs
    target_transform = ref.transform
    target_width     = ref.width
    target_height    = ref.height

with rasterio.open(input_mask) as src:
    kwargs = src.meta.copy()
    kwargs.update({
        'crs':       target_crs,
        'transform': target_transform,
        'width':     target_width,
        'height':    target_height
    })

    with rasterio.open(output_mask, 'w', **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_crs=src.crs,
            dst_crs=target_crs,
            dst_transform=target_transform,
            resampling=Resampling.nearest
        )

print("Gotovo!")