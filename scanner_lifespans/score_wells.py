import numpy
from scipy import ndimage
import pathlib
import functools
from zplib.image import fast_fft

MICRONS_PER_INCH = 25400

## Top-level function
def score_wells(well_images, well_mask, image_dpi, min_feature, max_feature, high_thresh, low_thresh, erode_iters, rescale=False):
    if rescale:
        if rescale == 'each':
            well_images = rescale_each_image(well_images, well_mask)
        else:
            well_images = rescale_images(well_images, well_mask)
    diff_images = difference_image_sets(well_images, min_feature, max_feature, image_dpi)
    scores = score_image_sets(diff_images, well_mask, high_thresh, low_thresh, erode_iters)
    return numpy.array(list(scores))

## Helper functions
def rescale_images(well_images, well_mask):
    means = []
    for images in well_images:
        for image in images:
            means.append(image[well_mask].mean())
    factor = 128 / numpy.median(means)
    return [[image.astype(numpy.float32)*factor for image in images] for images in well_images]

def rescale_each_image(well_images, well_mask):
    well_images_out = []
    for images in well_images:
        images_out = []
        for image in images:
            factor = 128 / image[well_mask].mean()
            images_out.append(image.astype(numpy.float32)*factor)
        well_images_out.append(images_out)
    return well_images_out

def difference_image_sets(well_images, min_feature, max_feature, image_dpi):
    microns_per_pixel = MICRONS_PER_INCH / image_dpi
    for images in well_images:
        yield difference_image(images, min_feature, max_feature, microns_per_pixel)

def score_image_sets(diff_images, well_mask, high_thresh, low_thresh, erode_iters):
    for diff_image in diff_images:
        yield score_diff_image(diff_image, well_mask, high_thresh, low_thresh, erode_iters)

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
    if low_thresh is not None:
        loose_mask = diff_image > low_thresh
        mask = ndimage.binary_dilation(eroded, mask=loose_mask, iterations=-1)
    else:
        mask = eroded
    mask = mask & well_mask
    score = diff_image[mask].mean() if numpy.any(mask) else 0
    return score
