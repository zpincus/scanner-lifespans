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
        for date_dir, images_for_date in itertools.groupby(all_images, lambda path: path.parent.parent):
            date_images.append([freeimage.read(str(image)) for image in images_for_date])
        date_images = score_wells.rescale_images(date_images, well_mask)
        well_images.append(date_images)
    return numpy.array(well_images), well_mask

def score_images(i, n, well_data, ref_lifespans, image_dpi, min_feature, max_feature, high_thresh, low_thresh, erode_iters):
        hmm_data = []
        all_ages = []
        all_scores = []
        all_states = []
        for well_images, well_mask, states, ages, lifespans in well_data:
            scores = []
            for date_images in well_images:
                date_scores = score_wells.score_wells(date_images, well_mask, image_dpi, min_feature, max_feature, high_thresh, low_thresh, erode_iters, rescale=False)
                scores.append(date_scores)
            scores = numpy.array(scores)
            hmm_data.append((scores, ages, lifespans))
            num_worms, num_timepoints = scores.shape
            assert len(ages) == num_timepoints
            all_ages.extend(numpy.tile(ages, num_worms)) # repeat ages for each worm. all_ages.shape == flat_scores.shape
            all_scores.extend(scores.flatten())
            all_states.extend(states.flatten())
        all_ages = numpy.array(all_ages)
        all_scores = numpy.array(all_scores)
        all_states = numpy.array(all_states)

        dead_scores = all_scores[all_states == 0]
        if numpy.all(dead_scores == dead_scores[0]):
            # problem if all dead scores are the same. Add some artificial jitter
            live_scores = live_scores[all_states == 0]
            if numpy.all(live_scores == live_scores[0]):
                print('perfect separation: ', i, n, min_feature, max_feature, high_thresh, low_thresh, erode_iters)
                return 1, [1]*len(well_data), [], []
            print('all dead scores identical: ', i, n, min_feature, max_feature, high_thresh, low_thresh, erode_iters)
            min_live = live_scores[live_scores > dead_scores[0]].min()
            dead_mask = all_states == 0
            all_scores[dead_mask] += numpy.random.normal(scale=min_live/100, size=sum(dead_mask))

        all_lifespans = []
        all_lifespans_est = []
        r2s = []
        for scores, ages, lifespans in hmm_data:
            states_est = estimate_lifespans.simple_hmm(scores, ages,
                ref_lifespans, all_ages, all_scores, all_states, lifespan_sigma=3)[0]
            lifespans_est = estimate_lifespans.cleanup_lifespans(
                estimate_lifespans.states_to_lifespans(states_est, ages), ages)
            all_lifespans.append(lifespans)
            all_lifespans_est.append(lifespans_est)
            r2s.append(_safe_r2(lifespans, lifespans_est))
        r2s = numpy.array(r2s)
        r2 = _safe_r2(numpy.concatenate(all_lifespans), numpy.concatenate(all_lifespans_est))
        print('scoring: ', i, n, min_feature, max_feature, high_thresh, low_thresh, erode_iters, r2, r2s.min(), r2s.max())
        return r2, r2s, all_lifespans, all_lifespans_est

def _safe_r2(x, y):
    try:
        r = stats.pearsonr(x, y)[0]**2
    except:
        r = 0
    if not numpy.isfinite(r):
        r = 0
    return r

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

def test_grid(workers, base_dirs, total_wells, image_dpi, grid, ref_lifespans):
    base_dirs = list(base_dirs)
    grid = grid_search.ParameterGrid(grid)
    good_params = [param_set for param_set in grid if
        (param_set['low_thresh'] is None or param_set['high_thresh'] >= param_set['low_thresh']) and
        (param_set['min_feature'] is None or param_set['max_feature'] > param_set['min_feature'])]
    n = len(good_params)
    all_lifespans = []
    well_data = []
    for base_dir in base_dirs:
        well_names, states, ages = prep_inputs(base_dir, total_wells//len(base_dirs))
        well_images, well_mask = load_images(well_names, base_dir)
        lifespans = estimate_lifespans.cleanup_lifespans(
            estimate_lifespans.states_to_lifespans(states, ages), ages)
        well_data.append((well_images, well_mask, states, ages, lifespans))
    evaluator = joblib.Parallel(n_jobs=workers, pre_dispatch='all', batch_size=1)
    results = evaluator(joblib.delayed(score_images)(i, n, well_data, ref_lifespans, image_dpi, **param_set)
        for i, param_set in enumerate(good_params))
    return parse_scores(results, good_params)

def parse_scores(results, good_params):
    data = []
    master_keys = None
    for res, param_set in zip(results, good_params):
        keys = sorted(param_set.keys())
        if master_keys is None:
            master_keys = keys
        else:
            assert master_keys == keys
        row = [param_set[key] for key in keys]
        r2 = res[0]
        data.append(row + [r2])
    data = pd.DataFrame(data)
    data.columns = master_keys + ['r2']
    order = data['r2'].argsort()
    data = data.iloc[order]
    data.index = list(range(len(good_params)))
    results = [results[i] for i in order]
    pooled_r2s, single_r2s, all_lifespans, all_lifespans_est = zip(*results)
    return single_r2s, all_lifespans, all_lifespans_est, data



