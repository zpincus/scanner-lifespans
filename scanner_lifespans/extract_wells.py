import warnings
import numpy
from scipy import ndimage
from scipy import cluster
from skimage import feature
from zplib.scalar_stats import mcd
from zplib.image import mask
from zplib.image import maxima

import registration

## Top-level functions
def get_well_mask(image):
    small_image, zoom_factor = downsample_image(image)
    rough_mask = get_rough_mask(small_image)
    well_mask, labeled_image, num_regions = get_labeled_mask(rough_mask)
    sample_coords, centroids = find_object_regions(labeled_image, num_regions)
    average = numpy.zeros(sample_coords.shape[1:], numpy.float32)
    for centroid in centroids:
        average += ndimage.map_coordinates(well_mask, sample_coords+numpy.reshape(centroid, (2,1,1)), order=1, output=numpy.float32)
    average /= len(centroids)
    large_average = zoom(average, zoom_factor, order=1)
    well_mask = large_average > 0.5
    return well_mask

def extract_wells(images, well_mask, x_names, y_names, exclude_names, x_names_first=True):
    images = list(images) # make sure images is a list and not an iterator
    x_names = list(map(str, x_names)) # make sure names are all strings
    y_names = list(map(str, y_names)) # make sure names are all strings
    x, y = ndimage.find_objects(well_mask)[0]
    x_size = x.stop-x.start
    y_size = y.stop-y.start
    well_size = max(x_size, y_size)
    match_scores, potential_centroids = find_potential_well_centroids(images[0], well_mask, well_size)
    well_names, centroids = find_well_names(match_scores, potential_centroids, well_size, x_names, y_names, exclude_names, x_names_first)
    initial_shifts = register_images(images, well_mask.shape, numpy.random.permutation(centroids)[:40])
    image_centroids = register_wells(images, centroids, well_mask.shape, initial_shifts)
    well_images = get_well_images(images, image_centroids, well_mask.shape)
    return image_centroids, well_names, well_images

## Helper functions
def zoom(*args, **kwargs):
    with warnings.catch_warnings():
        # suppress warning about image size changing from previous versions of scipy
        warnings.simplefilter('ignore', lineno=549)
        return ndimage.zoom(*args, **kwargs)

def find_potential_well_centroids(image, well_mask, well_size):
    small_image, zoom_factor = downsample_image(image)
    small_mask = zoom(well_mask, 1/zoom_factor, order=1, output=numpy.float32)
    small_mask = small_mask > 0.5
    small_well_size = well_size / zoom_factor
    peaks = feature.match_template(small_image, small_mask, pad_input=True, mode='edge')
    centroids, match_scores = maxima.find_local_maxima(peaks, min_distance=0.9*small_well_size)
    good_matches = match_scores > 0.1 * match_scores.max()
    return match_scores[good_matches], numpy.array(centroids)[good_matches] * zoom_factor

def find_well_names(match_scores, potential_centroids, well_size, x_names, y_names, exclude_names, x_names_first):
    centroids = numpy.asarray(potential_centroids)
    if exclude_names is not None:
        exclude_names = set(exclude_names)
    else:
        exclude_names = set()
    x_grid_positions = find_grid_positions(match_scores, centroids[:,0], well_size, num_positions=len(x_names))
    y_grid_positions = find_grid_positions(match_scores, centroids[:,1], well_size, num_positions=len(y_names))
    well_names = []
    centroid_indices = []
    for x_pos, x_name in zip(x_grid_positions, x_names):
        for y_pos, y_name in zip(y_grid_positions, y_names):
            if x_names_first:
                well_name = x_name + y_name
            else:
                well_name = y_name + x_name
            if well_name not in exclude_names:
                well_names.append(well_name)
                well_position = (x_pos, y_pos)
                squared_distances = ((centroids - well_position)**2).sum(axis=1)
                centroid_indices.append(squared_distances.argmin())
    # assert all indices are unique. Otherwise, two well names were assigned to a single position
    assert len(centroid_indices) == len(set(centroid_indices))
    return well_names, centroids[centroid_indices]

def find_grid_positions(match_scores, match_locations, well_size, num_positions):
    """Find the positions of one axis of a grid, given a set of scores for possible wells
    and the positions on that axis where those scores were found."""
    # match scores and locations are the positions (along just the x or y axis)
    # of possible wells. If there are 12 rows of wells along that given axis,
    # then these locations should group into 12 clusters, plus some background
    # false positives. These false positives likely have lower scores, though.
    # So: construct a 1-d array that has the well scores at each position.
    # Then, smooth the array, so that clusters of points will merge together,
    # and find the highest local maxima of the smoothed array.
    match_locations = match_locations.round().astype(int)
    min_loc, max_loc = match_locations.min(), match_locations.max()
    accumulator = numpy.zeros((max_loc - min_loc + 1), dtype=match_scores.dtype)
    match_locations -= min_loc
    accumulator[match_locations] = match_scores
    smoothed = ndimage.gaussian_filter1d(accumulator, well_size/5, mode='constant')
    positions, scores = maxima.find_local_maxima(smoothed, min_distance=well_size/3)
    positions = positions[:,0]
    best = scores.argsort()[-num_positions:]
    return numpy.sort(positions[best]) + min_loc

def register_images(images, aperture_shape, centroids):
    shifts = []
    for image in images[1:]:
        shift = registration.register_images(images[0], image, centroids, aperture_shape, xtol=0.5, eps=0.1)
        shifts.append(shift)
    return shifts

def register_wells(images, centroids, aperture_shape, shifts):
    centroids_out = [centroids]
    for image, shift in zip(images[1:], shifts):
        image_centroids = registration.register_wells(images[0], image, centroids,
        aperture_shape, initial_shift=shift, xtol=0.1, eps=0.02)
        centroids_out.append(image_centroids)
    return centroids_out

def get_well_images(images, image_centroids, well_size):
    well_images = []
    for image, centroids in zip(images, image_centroids):
        well_images.append(registration.get_wells(image, centroids, well_size))
    well_images = list(zip(*well_images)) # instead of n lists of 384 images, make 384 lists of n images...
    return well_images

def downsample_image(image, max_dim=2500):
    orig_shape = image.shape
    largest_dim = max(orig_shape)
    if largest_dim > max_dim:
        # need to make the image smaller
        shrink_factor = max_dim / largest_dim
        zoom_factor = largest_dim / max_dim
        image = zoom(image, shrink_factor, order=1)
    else:
        zoom_factor = 1
    return image, zoom_factor

def get_labeled_mask(well_mask):
    # make an image where each separate masked region has a specific numeric label
    labeled_image, num_regions = ndimage.label(well_mask, output=numpy.uint16)
    region_indices = numpy.arange(1, num_regions + 1) # region '0' is the non-masked background area
    ones = numpy.ones_like(labeled_image)
    # sum up an image full of 1s separately for each labeled region (except the background)
    # this then gives the area in pixels of each region
    areas = ndimage.sum(ones, labels=labeled_image, index=region_indices)
    # now erase any labels where the area isn't pretty close to the median
    median_area = numpy.median(areas)
    bad_areas = (areas < median_area*0.7) | (areas > median_area*1.3)
    bad_indices = region_indices[bad_areas]
    if len(bad_indices) > 0:
        for i in bad_indices:
            # set the mask image to zero everywhere that the label image matches this particular bad index
            well_mask[labeled_image == i] = 0
        labeled_image, num_regions = ndimage.label(well_mask, output=numpy.uint16)
    return well_mask, labeled_image, num_regions

def find_object_regions(labeled_image, num_regions, size_fudge_factor=1.2):
    slices = ndimage.find_objects(labeled_image)
    x_max = max(x.stop-x.start for x, y in slices)
    y_max = max(y.stop-y.start for x, y in slices)
    well_size = int(round(max(x_max, y_max) * size_fudge_factor))
    well_shape = (well_size,well_size)
    sample_coords = numpy.indices(well_shape) - (well_size-1)/2 # recenter indices
    region_indices = numpy.arange(1, num_regions + 1)
    # calculate the centroid of each well in the labeled image
    centroids = ndimage.center_of_mass(numpy.ones_like(labeled_image), labels=labeled_image, index=region_indices)
    return sample_coords, centroids

def get_rough_mask(image):
    """Take an image with light wells and dark inter-well regions and construct
    a mask with 'True' values in all pixels belonging to a well and 'False' values
    elsewhere."""
    # take every 3rd pixel in each direction and then flatten to a 1D array.
    image_sample = image[::3, ::3].flatten()
    # use that subsample to estimate the mean and variance of the background pixels
    mean, std = mcd.robust_mean_std(image_sample, subset_fraction=0.5)
    # find the non-background pixels
    well_mask = image > (mean + 12*std)
    well_mask = ndimage.binary_fill_holes(well_mask)
    well_mask = mask.remove_small_radius_objects(well_mask, max_radius=10)
    well_mask = mask.remove_edge_objects(well_mask)
    return well_mask

