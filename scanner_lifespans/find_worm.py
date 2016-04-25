import numpy
from scipy import ndimage
from skimage import restoration

from zplib.image import active_contour
from zplib.image import canny
from zplib.image import mask
from zplib.image import polyfit
from zplib.image import pyramid
from zplib.scalar_stats import mcd

S = numpy.ones((3, 3), dtype=bool)

def find_worm_from_fluorescence(image, low_pct=99.2, high_pct=99.99, max_hole_radius=12):
    low_thresh, high_thresh = numpy.percentile(image, [low_pct, high_pct])
    worm_mask = mask.hysteresis_threshold(image, low_thresh, high_thresh)
    worm_mask = mask.fill_small_radius_holes(worm_mask, max_hole_radius)
    worm_mask = mask.get_largest_object(worm_mask)
    return worm_mask

def find_worm_from_brightfield(image):
    small_image, well_mask = get_well_mask(image)
    if well_mask is None:
        no_mask = numpy.zeros(image.shape, dtype=bool)
        return no_mask, no_mask, no_mask
    initial_edges, initial_area = find_initial_worm(small_image, well_mask)
    slicex, slicey = ndimage.find_objects(initial_area, max_label=1)[0]
    slices = (slice(slicex.start*4, slicex.stop*4), slice(slicey.start*4, slicey.stop*4))
    well_mask = pyramid.pyr_up(well_mask, 4)>0.5
    initial_edges = pyramid.pyr_up(initial_edges, 4)>0.5
    initial_area = pyramid.pyr_up(initial_area, 4)>0.5
    sliced_edges, sliced_worm = refine_worm(image[slices], initial_area[slices], initial_edges[slices])
    worm_mask = numpy.zeros_like(well_mask)
    worm_mask[slices] = sliced_worm
    edges = numpy.zeros_like(well_mask)
    edges[slices] = sliced_edges
    return well_mask, edges, worm_mask

def get_well_mask(image):
    small_image = pyramid.pyr_down(image, 4)
    smoothed, gradient, sobel = canny.prepare_canny(small_image, 2)
    local_maxima = canny.canny_local_maxima(gradient, sobel)
    # well outline has ~6000 px full-size = 1500 px at 4x downsampled
    # So find the intensity value of the 2000th-brightest pixel, via percentile:
    highp = 100 * (1-2000/local_maxima.sum())
    highp = max(highp, 90)
    low_edge, high_edge = numpy.percentile(gradient[local_maxima], [90, highp])
    # Do canny edge-finding starting with gradient pixels as bright or brighter
    # than the 2000th-brightest pixel, and spread out to the 90th percentile
    # intensity:
    well_edge = canny.canny_hysteresis(local_maxima, gradient, low_edge, high_edge)
    # connect nearby edges and remove small unconnected bits
    well_edge = ndimage.binary_closing(well_edge, structure=S)
    well_edge = mask.remove_small_area_objects(well_edge, 300, structure=S)
    # Get map of distances and directions to nearest edge to use for contour-fitting
    distances, nearest_edge = active_contour.edge_direction(well_edge)
    # initial curve is the whole image less one pixel on the outside
    initial = numpy.ones(well_edge.shape, dtype=bool)
    initial[:,[0,-1]] = 0
    initial[[0,-1],:] = 0
    # Now evolve the curve inward until it contacts the canny well edges.
    gac = active_contour.GAC(initial, nearest_edge, advection_mask=(distances < 10), balloon_direction=-1)
    stopper = active_contour.StoppingCondition(gac, max_iterations=200)
    while stopper.should_continue():
        # otherwise evolve the curve by shrinking, advecting toward edges, and smoothing
        gac.balloon_force(iters=3)
        gac.advect(iters=2)
        gac.smooth()
    gac.smooth(depth=3)
    # now erode everywhere the contour edge is right on a canny edge:
    gac.move_to_outside(well_edge[tuple(gac.inside_border_indices.T)])
    gac.smooth()
    well_mask = gac.mask
    if well_mask.sum() / well_mask.size < 0.25:
        # if the well mask is too small, something went very wrong
        well_mask = None
    return small_image, well_mask


def find_initial_worm(small_image, well_mask):
    # plan here is to find known good worm edges with Canny using a stringent threshold, then
    # relax the threshold in the vicinity of the good edges.
    # back off another pixel from the well edge to avoid gradient from the edge
    shrunk_mask = ndimage.binary_erosion(well_mask, structure=S)
    smoothed, gradient, sobel = canny.prepare_canny(small_image, 2, shrunk_mask)
    local_maxima = canny.canny_local_maxima(gradient, sobel)
    # Calculate stringent and medium-stringent thresholds. The stringent threshold
    # is the 200th-brightest edge pixel, and the medium is the 450th-brightest pixel
    highp = 100 * (1-200/local_maxima.sum())
    highp = max(highp, 94)
    mediump = 100 * (1-450/local_maxima.sum())
    mediump = max(mediump, 94)
    low_worm, medium_worm, high_worm = numpy.percentile(gradient[local_maxima], [94, mediump, highp])
    stringent_worm = canny.canny_hysteresis(local_maxima, gradient, low_worm, high_worm)
    # Expand out 20 pixels from the stringent worm edges to make our search space
    stringent_area = ndimage.binary_dilation(stringent_worm, mask=well_mask, iterations=20)
    # now use the relaxed threshold but only in the stringent area
    relaxed_worm = canny.canny_hysteresis(local_maxima, gradient, low_worm, medium_worm) & stringent_area
    # join very close-by objects, and remove remaining small objects
    candidate_worm = ndimage.binary_dilation(relaxed_worm, structure=S)
    candidate_worm = ndimage.binary_erosion(candidate_worm)
    candidate_worm = mask.remove_small_area_objects(candidate_worm, 30, structure=S)
    # Now figure out the biggest blob of nearby edges, and call that the worm region
    glommed_candidate = ndimage.binary_dilation(candidate_worm, structure=S, iterations=2)
    glommed_candidate = ndimage.binary_erosion(glommed_candidate, iterations=2)
    # get just outline, not any regions filled-in due to closing
    glommed_candidate ^= ndimage.binary_erosion(glommed_candidate)
    glommed_candidate = mask.get_largest_object(glommed_candidate, structure=S)
    worm_area = ndimage.binary_dilation(glommed_candidate, mask=well_mask, structure=S, iterations=12)
    worm_area = mask.fill_small_radius_holes(worm_area, max_radius=15)
    candidate_edges = relaxed_worm & candidate_worm & worm_area
    return candidate_edges, worm_area


def refine_worm(image, initial_area, candidate_edges):
    # find strong worm edges (roughly equivalent to the edges found by find_initial_worm,
    # which are in candidate_edges): smooth the image, do canny edge-finding, and
    # then keep only those edges near candidate_edges
    smooth_image = restoration.denoise_tv_bregman(image, 140).astype(numpy.float32)
    smoothed, gradient, sobel = canny.prepare_canny(smooth_image, 8, initial_area)
    local_maxima = canny.canny_local_maxima(gradient, sobel)
    candidate_edge_region = ndimage.binary_dilation(candidate_edges, iterations=4)
    strong_edges = local_maxima & candidate_edge_region

    # Now threshold the image to find dark blobs as our initial worm region
    # First, find areas in the initial region unlikely to be worm pixels
    mean, std = mcd.robust_mean_std(smooth_image[initial_area][::4], 0.85)
    non_worm = (smooth_image > mean - std) & initial_area
    # now fit a smoothly varying polynomial to the non-worm pixels in the initial
    # region of interest, and subtract that from the actual image to generate
    # an image with a flat illumination field
    background = polyfit.fit_polynomial(smooth_image, mask=non_worm, degree=2)
    minus_bg = smooth_image - background
    # now recalculate a threshold from the background-subtracted pixels
    mean, std = mcd.robust_mean_std(minus_bg[initial_area][::4], 0.85)
    initial_worm = (minus_bg < mean - std) & initial_area
    # Add any pixels near the strong edges to our candidate worm position
    initial_worm |= ndimage.binary_dilation(strong_edges, iterations=3)
    initial_worm = mask.fill_small_radius_holes(initial_worm, 5)

    # Now grow/shrink the initial_worm region so that as many of the strong
    # edges from the canny filter are in contact with the region edges as possible.
    ac = active_contour.EdgeClaimingAdvection(initial_worm, strong_edges,
        max_region_mask=initial_area)
    stopper = active_contour.StoppingCondition(ac, max_iterations=100)
    while stopper.should_continue():
        ac.advect(iters=1)
        ac.smooth(iters=1, depth=2)
    worm_mask = mask.fill_small_radius_holes(ac.mask, 7)

    # Now, get edges from the image at a finer scale
    smoothed, gradient, sobel = canny.prepare_canny(smooth_image, 0.3, initial_area)
    local_maxima = canny.canny_local_maxima(gradient, sobel)
    strong_sum = strong_edges.sum()
    highp = 100 * (1 - 1.5*strong_sum/local_maxima.sum())
    lowp = max(100 * (1 - 3*strong_sum/local_maxima.sum()), 0)
    low_worm, high_worm = numpy.percentile(gradient[local_maxima], [lowp, highp])
    fine_edges = canny.canny_hysteresis(local_maxima, gradient, low_worm, high_worm)

    # Expand out the identified worm area to include any of these finer edges
    closed_edges = ndimage.binary_closing(fine_edges, structure=S)
    worm = ndimage.binary_propagation(worm_mask, mask=worm_mask|closed_edges, structure=S)
    worm = ndimage.binary_closing(worm, structure=S, iterations=2)
    worm = mask.fill_small_radius_holes(worm, 5)
    worm = ndimage.binary_opening(worm)
    worm = mask.get_largest_object(worm)
    # Last, smooth the shape a bit to reduce sharp corners, but not too much to
    # sand off the tail
    ac = active_contour.CurvatureMorphology(worm, max_region_mask=initial_area)
    ac.smooth(depth=2, iters=2)
    return strong_edges, ac.mask