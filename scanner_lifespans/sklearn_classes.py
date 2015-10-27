## sklearn interface to well-scoring, for parameter fitting etc.
import numpy
from scipy import stats
import pathlib
from scipy import stats
import itertools
import pandas as pd

from sklearn import grid_search
import joblib
import freeimage
from zplib.scalar_stats import compare_distributions

from . import score_wells
from . import estimate_lifespans
from . import run_analysis

def load_images(well_names, base_dir):
    base_dir = pathlib.Path(base_dir)
    well_mask = freeimage.read(str(base_dir / 'well_mask.png')) > 0
    well_images = []
    for well_name in well_names:
        all_images = sorted(base_dir.glob('*/well_images/'+well_name+'-*.png'))
        date_images = []
        well_images.append(date_images)
        for date_dir, images_for_date in itertools.groupby(all_images, lambda path: path.parent.parent):
            date_images.append([freeimage.read(str(image)) for image in images_for_date])
    return numpy.array(well_images), well_mask

def score_images(i, n, well_images, well_mask, states, ages, image_dpi, percentile, min_feature, max_feature, high_thresh, low_thresh, erode_iters):
        scores = []
        for date_images in well_images:
            if percentile is not None:
                scaled_images = []
                for images in date_images:
                    scale = numpy.percentile(images[0][well_mask], percentile)
                    scaled_images.append([i.astype(numpy.float32)/scale for i in images])
                date_images = scaled_images
            diff_images = score_wells.difference_image_sets(date_images, min_feature, max_feature, image_dpi)
            score_iter = score_wells.score_image_sets(diff_images, well_mask, high_thresh, low_thresh, erode_iters, image_dpi)
            scores.append(list(score_iter))
        scores = numpy.array(scores)
        live_scores = scores[states==1]
        dead_scores = scores[states==0]
        t = stats.ttest_ind(live_scores, dead_scores, equal_var=False).statistic
        ks = compare_distributions.ks_statistic(live_scores, dead_scores)
        try:
            states_out = estimate_lifespans.estimate_lifespans(scores, ages, states, scores, ages)
            lifespans = estimate_lifespans.states_to_lifespans(states, ages)
            lifespans_out = estimate_lifespans.states_to_lifespans(states_out, ages)
            lifespans_out[numpy.isnan(lifespans_out)] = numpy.nanmax(lifespans_out)+1
            r = stats.pearsonr(lifespans, lifespans_out)[0]
        except:
            r = 0
        if not numpy.isfinite(r):
            r = 0
        print('scoring: ', i, n, min_feature, max_feature, percentile, high_thresh, low_thresh, erode_iters, t, ks, r)
        return r, t, ks, live_scores, dead_scores

def prep_inputs(base_dir, total_wells):
    data = run_analysis.load_data(base_dir)
    well_mask = numpy.array([i != None for i in data.eval_last_alive_indices])
    wells_included = well_mask.sum()
    if total_wells < wells_included:
        mask_mask = numpy.zeros(wells_included, dtype=bool)
        mask_mask[:total_wells] = True
        well_mask[well_mask] = numpy.random.permutation(mask_mask)
    well_names = list(numpy.array(data.well_names)[well_mask])
    states = estimate_lifespans.last_alive_indices_to_states(data.eval_last_alive_indices, len(data.dates))[well_mask]
    return well_names, states, data.ages

def test_grid(workers, base_dir, well_names, states, ages, image_dpi, grid):
    grid = grid_search.ParameterGrid(grid)
    n = len(grid)
    well_images, well_mask = load_images(well_names, base_dir)
    good_params = [param_set for param_set in grid if param_set['high_thresh'] > param_set['low_thresh']]
    scores = joblib.Parallel(n_jobs=workers, pre_dispatch='all', batch_size=1)(joblib.delayed(score_images)(i, n, well_images, well_mask, states, ages, image_dpi, **param_set)
        for i, param_set in enumerate(good_params))
    return parse_scores(scores, good_params)

def parse_scores(scores, good_params):
    stats = []
    params = []
    master_keys = None
    for (r, t, ks, live_scores, dead_scores), param_set in zip(scores, good_params):
        stats.append((r, t, ks))
        keys = sorted(param_set.keys())
        if master_keys is None:
            master_keys = keys
        else:
            assert master_keys == keys
        params.append([param_set[key] for key in keys])
    stats = pd.DataFrame(stats)
    stats.columns = ['r', 't', 'ks']
    params = pd.DataFrame(params)
    params.columns = master_keys
    order = stats.r.argsort()
    return stats.iloc[order], params.iloc[order]

