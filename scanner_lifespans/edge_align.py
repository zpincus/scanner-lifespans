import numpy
from scipy import ndimage
from scipy import optimize
import collections

# Top-level functions

def align_edges(images, target_shape=None, target_bbox=None):
    fixed_edges = find_edges(images[0])
    if target_shape is None:
        shape = images[0].shape
    else:
        shape = target_shape
    if target_bbox is None:
        aligned = [images[0]]
        target_bbox = fixed_edges[:4]
    else:
        aligned = [_remap(images[0], target_shape, fixed_edges[:4], target_bbox)]
    for image in images[1:]:
        moving_edges = find_edges(image)
        left = _match_edges(fixed_edges, moving_edges, 'left', window_width=10)
        right = _match_edges(fixed_edges, moving_edges, 'right', window_width=10)
        top = _match_edges(fixed_edges, moving_edges, 'top', window_width=10)
        bottom = _match_edges(fixed_edges, moving_edges, 'bottom', window_width=10)
        aligned.append(_remap(image, shape, (left, right, top, bottom), target_bbox))
    return aligned

WellEdges = collections.namedtuple('WellEdges', ['left', 'right', 'top', 'bottom', 'lr_profile', 'lr_gradient', 'lr_indices', 'tb_profile', 'tb_gradient', 'tb_indices'])

def find_edges(image):
    left, right, lr_profile, lr_gradient, lr_indices = _find_edges_1d(image, axis=0)
    top, bottom, tb_profile, tb_gradient, tb_indices = _find_edges_1d(image, axis=1)
    return WellEdges(left, right, top, bottom, lr_profile, lr_gradient, lr_indices, tb_profile, tb_gradient, tb_indices)


# Helper functions

def _remap(image, shape, bbox_in, bbox_out):
    shape = numpy.array(shape)
    coords = numpy.empty([2]+list(shape), numpy.float32)
    l_in, r_in, t_in, b_in = bbox_in
    l_out, r_out, t_out, b_out = bbox_out
    x_ind = _get_indices(shape[0], l_in, r_in, l_out, r_out)
    coords[0] = x_ind[:, numpy.newaxis]
    y_ind = _get_indices(shape[1], t_in, b_in, t_out, b_out)
    coords[1] = y_ind[numpy.newaxis, :]
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

def _find_edges_1d(image, axis):
    if axis != 0:
        image = image.T
    edge_width = int(image.shape[1] / 2.5)
    image_strip = image[:, edge_width:-edge_width]
    profile = image_strip.mean(axis=1)
    indices = numpy.arange(len(profile))
    gradient = numpy.gradient(profile)
    left = _weighted_position(gradient, indices, gradient.argmax(), 5)
    right = _weighted_position(-gradient, indices, gradient.argmin(), 5)
    return left, right, profile, gradient, indices

def _weighted_position(values, indices, position, width):
    near = values[position-width:position+width+1]
    near_i = indices[position-width:position+width+1]
    near = near / near.sum()
    return (near * near_i).sum()

def _score(offset, fixed, moving, d_moving, window, indices):
    diff = (fixed[window] - numpy.interp(window + offset, indices, moving))
    score = diff**2
    der = -2 * diff * numpy.interp(window + offset, indices, d_moving)
    return score.sum(), der.sum()