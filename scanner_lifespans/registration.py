import numpy
import scipy.ndimage as ndimage
import scipy.optimize as optimize

## Image comparison metrics
def abs_diff(i1, i2):
    return numpy.abs(i1 - i2)

def square_diff(i1, i2):
    return (i1 - i2)**2

## Top-level functions
def register_images(fixed_image, moving_image, centroids, sample_shape, initial_shift=(0,0), search_bounds=5, diff_function=abs_diff, xtol=0.01, eps=0.01):
    sample_coords = prepare_sample_positions(centroids, sample_shape)
    fixed_image_sample = ndimage.map_coordinates(fixed_image, sample_coords, output=numpy.float32, cval=numpy.nan, order=1)
    args = fixed_image_sample, moving_image, sample_coords, diff_function, numpy.nan, {}
    bounds = numpy.array([-search_bounds, search_bounds]) + initial_shift
    bounds = [bounds, bounds] # one for x and one for y
    result = optimize.minimize(compare_images, initial_shift, args=args, method='TNC', options={'xtol':xtol, 'eps':eps}, bounds=bounds)
    return result.x

def register_wells(fixed_image, moving_image, centroids, sample_shape, initial_shift=(0,0), search_bounds=2, diff_function=abs_diff, xtol=0.01, eps=0.01):
    sample_coords = prepare_sample_positions(centroids, sample_shape)
    fixed_image_sample = ndimage.map_coordinates(fixed_image, sample_coords, output=numpy.float32, cval=numpy.nan, order=1)
    centroids_out = []
    search_range = numpy.array([-search_bounds, search_bounds])
    bounds = [search_range + initial_shift[0], search_range + initial_shift[1]] # one for x and one for y
    for i in range(len(centroids)):
        fixed_image_i = fixed_image_sample[i]
        sample_coords_i = sample_coords[:,i]
        args_i = fixed_image_i, moving_image, sample_coords_i, diff_function, numpy.nan, {}
        result = optimize.minimize(compare_images, initial_shift, args=args_i, method='TNC', options={'xtol':xtol, 'eps':eps}, bounds=bounds)
        centroids_out.append(centroids[i] + result.x)
    return centroids_out

def get_wells(image, centroids, sample_shape):
    sample_coords = prepare_sample_positions(centroids, sample_shape)
    return ndimage.map_coordinates(image, sample_coords, order=1)

## Helper functions
def compare_images(shift, fixed_image_sample, moving_image, sample_coords, diff_function, cval, cache):
    hash_shift = tuple(shift)
    if hash_shift in cache:
        return cache[hash_shift]
    shifted_coords = sample_coords + numpy.reshape(shift, (2,)+(1,)*(len(sample_coords.shape)-1))
    moving_image_sample = ndimage.map_coordinates(moving_image, shifted_coords, output=numpy.float32, cval=cval, order=1)
    diff = diff_function(fixed_image_sample, moving_image_sample)
    result = diff[numpy.isfinite(diff)].sum()
    cache[hash_shift] = result
    return result

def centered_indices(shape):
    indices = numpy.indices(shape) # indices.shape == (2, shape[0], shape[1])
    centered = indices - (numpy.reshape(shape, (2,1,1))-1)/2 # recenter indices
    return centered

def prepare_sample_positions(centroids, sample_shape):
    centroids = numpy.asarray(centroids)
    x_ind, y_ind = centered_indices(sample_shape)
    x_coords = numpy.add.outer(centroids[:,0], x_ind)
    y_coords = numpy.add.outer(centroids[:,1], y_ind)
    sample_coords = numpy.array([x_coords, y_coords]) # shape = (2, len(centroids), sample_shape[0], sample_shape[1])
    return sample_coords
