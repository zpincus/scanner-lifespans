import numpy
from scipy import ndimage
import pathlib

from zplib.image import fast_fft

MAX_WORM_WIDTH_MICRONS = 75
MICRONS_PER_INCH = 25400

## Top-level function
def score_wells(well_images, well_mask, image_dpi, min_feature, max_feature, local_max_percentile,
        high_thresh, low_thresh, erode_iters, return_images):
    microns_per_pixel = MICRONS_PER_INCH / image_dpi
    image_shape = well_images[0][0].shape
    enlarged_mask = ndimage.binary_dilation(well_mask, iterations=int(round(80/microns_per_pixel)))
    fft_filter = get_fft_filter(image_shape, min_feature, max_feature, microns_per_pixel)
    cleaned_image_sets = []
    diff_images = []
    well_scores = []
    for i, images in enumerate(well_images):
        recentered, shading_corrected = clean_images(images, microns_per_pixel, local_max_percentile)
        diff_image = difference_image(shading_corrected, fft_filter)
        if return_images:
            diff_images.append(diff_image)
        score = score_diff_image(diff_image, enlarged_mask, high_thresh, low_thresh, erode_iters)
        well_scores.append(score)
    return well_scores, diff_images

## Helper functions
def get_fft_filter(image_shape, min_feature, max_feature, microns_per_pixel):
    fftw_hints = pathlib.Path(__file__).parent / 'fftw_hints'
    if fftw_hints.exists():
        fast_fft.load_plan_hints(str(fftw_hints))
    fft_filter = fast_fft.SpatialFilter(image_shape, period_range=(min_feature, max_feature),
        spacing=microns_per_pixel, order=2, keep_dc=False, better_plan=True)
    fast_fft.store_plan_hints(str(fftw_hints))
    return fft_filter

def clean_images(images, microns_per_pixel, local_max_percentile):
    image_shape = images[0].shape
    denoised = [ndimage.median_filter(image, 3) for image in images]
    medians = [numpy.median(d) for d in denoised]
    recentered = [denoised[0]]
    recentered += [(d.astype(numpy.float32) - m + medians[0]).clip(0, numpy.inf).astype(images[0].dtype) for d, m in zip(denoised[1:], medians[1:])]
    window = int(round(2*MAX_WORM_WIDTH_MICRONS/microns_per_pixel))
    small_size = numpy.round((numpy.array(image_shape) / 2)).astype(int)
    shrink_factor = small_size / image_shape
    small_images = [ndimage.zoom(image, shrink_factor, order=1) for image in recentered]
    local_max = [ndimage.percentile_filter(image, local_max_percentile, window) for image in small_images]
    max_max = numpy.max(local_max, axis=0)
    big_max = ndimage.zoom(max_max, 1/shrink_factor, order=1)
    shading_corrected = [r.astype(numpy.float32) / big_max for r in recentered]
    return recentered, shading_corrected

def difference_image(images, fft_filter):
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
