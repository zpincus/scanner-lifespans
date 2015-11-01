import numpy
from scipy import ndimage
from scipy import optimize
import collections

# Top-level functions

def align_edges(images, approx_edge_locations=None, corner_fraction=0.3, edge_search_fraction=1, origin=None, target_shape=None, target_bbox=None):
    '''Align well edges from a set of images to one another, and optionally force
    aligned edges to be an a given absolute position.

    Parameters:
        images: list of images of square-ish white-on-black well images
        approx_edge_locations: if not None, (left, right, top, bottom) tuple of
            positions in image where the four edges will approximately be.
        corner_fraction: if wells are rounded squares, this gives the fraction
            of the well width that is the rounded corner, and should be ignored
            when calculating the image gradient that is used to find the edge
            position.
        edge_search_fraction: fraction of the total image width away from the
            given approx_edge_location that the actual edge will be sought.
        origin: if not None, gives the position of the well to be aligned in
            the image.
        target_shape: if not None, gives the shape of the output aligned images.
            Must be specified if origin is specified.
        target_bbox: (left, right, top, bottom) tuple of locations where the
            aligned edges will be placed in the output images.

    Returns: list of images resampled such that well edges are in the same positions.
    '''
    if target_shape is None:
        if origin is not None:
            raise ValueError('If an origin is specified, a target_shape must be as well.')
        shape = images[0].shape
    else:
        shape = target_shape

    if origin is None:
        origin = numpy.array([0, 0], dtype=int)
        sliced_images = images
    else:
        sx, sy = shape
        ox, oy = origin
        sliced_images = [image[ox:ox+sx, oy:oy+sy] for image in images]

    fixed_edges = find_edges(sliced_images[0], approx_edge_locations, corner_fraction, edge_search_fraction)
    if target_bbox is None:
        aligned = [images[0]]
        target_bbox = fixed_edges[:4]
    else:
        aligned = [_remap(images[0], origin, target_shape, fixed_edges[:4], target_bbox)]

    for sliced, image in zip(sliced_images[1:], images[1:]):
        moving_edges = find_edges(sliced, approx_edge_locations, corner_fraction, edge_search_fraction)
        left = _match_edges(fixed_edges, moving_edges, 'left', window_width=10)
        right = _match_edges(fixed_edges, moving_edges, 'right', window_width=10)
        top = _match_edges(fixed_edges, moving_edges, 'top', window_width=10)
        bottom = _match_edges(fixed_edges, moving_edges, 'bottom', window_width=10)
        aligned.append(_remap(image, origin, shape, (left, right, top, bottom), target_bbox))
    return aligned

WellEdges = collections.namedtuple('WellEdges', ['left', 'right', 'top', 'bottom', 'lr_profile', 'lr_gradient', 'lr_indices', 'tb_profile', 'tb_gradient', 'tb_indices'])

def find_edges(image, approx_edge_locations=None, corner_fraction=0.3, search_fraction=1):
    if approx_edge_locations is None:
        right, bottom = image.shape
        left = top = 0
    else:
        left, right, top, bottom = approx_edge_locations
    lr_strip = _get_image_slice(image, left, right, corner_fraction, axis=0)
    left, right, lr_profile, lr_gradient, lr_indices = _find_edges_1d(lr_strip, left, right, search_fraction)
    tb_strip = _get_image_slice(image, top, bottom, corner_fraction, axis=1)
    top, bottom, tb_profile, tb_gradient, tb_indices = _find_edges_1d(tb_strip, top, bottom, search_fraction)
    return WellEdges(left, right, top, bottom, lr_profile, lr_gradient, lr_indices, tb_profile, tb_gradient, tb_indices)

# Helper functions

def _remap(image, origin, shape, bbox_in, bbox_out):
    shape = numpy.array(shape)
    coords = numpy.empty([2]+list(shape), numpy.float32)
    l_in, r_in, t_in, b_in = bbox_in
    l_out, r_out, t_out, b_out = bbox_out
    x_ind = _get_indices(shape[0], l_in, r_in, l_out, r_out)
    coords[0] = x_ind[:, numpy.newaxis]
    y_ind = _get_indices(shape[1], t_in, b_in, t_out, b_out)
    coords[1] = y_ind[numpy.newaxis, :]
    coords += origin[:, numpy.newaxis, numpy.newaxis]
    return ndimage.map_coordinates(image, coords, order=1)

def _get_indices(size, low_in, high_in, low_out, high_out):
    indices_out = numpy.arange(size)
    # find line through (low_out, low_in), (high_out, high_in)
    # low_in = a * low_out + b
    # high_in = a * high_out + b
    a = (low_in - high_in) / (low_out - high_out)
    b = high_in - a * high_out
    indices_in = a * indices_out + b
    return indices_in

_edges = dict(left='lr_', right='lr_', top='tb_', bottom='tb_')
def _match_edges(fixed_edges, moving_edges, edge, window_width):
    assert edge in _edges
    fixed_edges = fixed_edges._asdict()
    moving_edges = moving_edges._asdict()
    prefix = _edges[edge]
    fixed = fixed_edges[prefix+'profile']
    moving = moving_edges[prefix+'profile']
    d_moving = moving_edges[prefix+'gradient']
    indices = moving_edges[prefix+'indices']
    fixed_center = fixed_edges[edge]
    moving_center_est = moving_edges[edge]
    window = numpy.arange(window_width) + int(round(fixed_center - window_width/2))
    offset_est = moving_center_est - fixed_center
    offset = optimize.minimize(_score, offset_est, args=(fixed, moving, d_moving, window, indices), jac=True, method='TNC').x[0]
    return fixed_center + offset

def _get_image_slice(image, low, high, corner_fraction, axis):
    if axis != 0:
        image = image.T
    full_width = high-low
    exclude = int(round(full_width * corner_fraction))
    image_strip = image[:, low+exclude:high-exclude]
    return image_strip

def _find_edges_1d(image, low_est, high_est, search_fraction):
    profile = image.mean(axis=1)
    indices = numpy.arange(len(profile))
    gradient = numpy.gradient(profile)
    search_radius = int(round(search_fraction * len(profile)))
    low_start = max(0, low_est - search_radius)
    low_stop = min(len(profile), low_est + search_radius)
    low = _weighted_position(gradient, indices, gradient[low_start:low_stop].argmax()+low_start, 5)
    high_start = max(0, high_est - search_radius)
    high_stop = min(len(profile), high_est + search_radius)
    high = _weighted_position(-gradient, indices, gradient[high_start:high_stop].argmin()+high_start, 5)
    return low, high, profile, gradient, indices

def _weighted_position(values, indices, position, width):
    near = values[position-width:position+width+1]
    near_i = indices[position-width:position+width+1]
    near = near / near.sum()
    return (near * near_i).sum()

def _score(offset, fixed, moving, d_moving, window, indices):
    '''score function is sum((fixed[i] - moving[i+offset])**2). Return score and
    d_score/d_offset for a given offset and window (which defines the set of
    i that the score is computed over).
    '''
    diff = (fixed[window] - numpy.interp(window + offset, indices, moving))
    score = diff**2
    der = -2 * diff * numpy.interp(window + offset, indices, d_moving)
    return score.sum(), der.sum()