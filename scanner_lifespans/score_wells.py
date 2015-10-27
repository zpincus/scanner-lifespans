import numpy
from scipy import ndimage
import pathlib
import functools
from zplib.image import fast_fft

MICRONS_PER_INCH = 25400

## Top-level function
def score_wells(well_images, well_mask, image_dpi, min_feature, max_feature, high_thresh, low_thresh, erode_iters):
    diff_images = difference_image_sets(well_images, min_feature, max_feature, image_dpi)
    scores = score_image_sets(diff_images, well_mask, high_thresh, low_thresh, erode_iters, image_dpi)
    return numpy.array(scores)

## Helper functions
def difference_image_sets(well_images, min_feature, max_feature, image_dpi):
    microns_per_pixel = MICRONS_PER_INCH / image_dpi
    for images in well_images:
        yield difference_image(images, min_feature, max_feature, microns_per_pixel)

def score_image_sets(diff_images, well_mask, high_thresh, low_thresh, erode_iters, image_dpi):
    microns_per_pixel = MICRONS_PER_INCH / image_dpi
    enlarged_mask = ndimage.binary_dilation(well_mask, iterations=int(round(80/microns_per_pixel)))
    for diff_image in diff_images:
        yield score_diff_image(diff_image, enlarged_mask, high_thresh, low_thresh, erode_iters)

@functools.lru_cache(maxsize=32)
def get_fft_filter(image_shape, min_feature, max_feature, microns_per_pixel):
    fftw_hints = pathlib.Path(__file__).parent / 'fftw_hints'
    if fftw_hints.exists():
        fast_fft.load_plan_hints(str(fftw_hints))
    fft_filter = fast_fft.SpatialFilter(image_shape, period_range=(min_feature, max_feature),
        spacing=microns_per_pixel, order=2, keep_dc=False, threads=1, better_plan=True)
    fast_fft.store_plan_hints(str(fftw_hints))
    return fft_filter

def difference_image(images, min_feature, max_feature, microns_per_pixel):
    fft_filter = get_fft_filter(images[0].shape, min_feature, max_feature, microns_per_pixel)
    differences = []
    for image in images[1:]:
        diff = image.astype(numpy.float32) - images[0]
        filtered_diff = numpy.abs(fft_filter.filter(diff).copy())
        differences.append(filtered_diff)
    diff_image = numpy.mean(differences, axis=0)
    return diff_image

def score_diff_image(diff_image, well_mask, high_thresh, low_thresh, erode_iters):
    mask = diff_image > high_thresh
    if erode_iters:
        eroded = ndimage.binary_erosion(mask, iterations=erode_iters)
    else:
        eroded = mask
    loose_mask = diff_image > low_thresh
    mask = ndimage.binary_dilation(eroded, mask=loose_mask, iterations=-1)
    mask = mask & well_mask
    score = diff_image[mask].mean() if numpy.any(mask) else low_thresh
    return score
