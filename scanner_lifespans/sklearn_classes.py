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

def score_images(i, n, well_images, well_mask, states, ages, ref_lifespans, image_dpi, min_feature, max_feature, high_thresh, low_thresh, erode_iters):
        scores = []
        for date_images in well_images:
            diff_images = score_wells.difference_image_sets(date_images, min_feature, max_feature, image_dpi)
            score_iter = score_wells.score_image_sets(diff_images, well_mask, high_thresh, low_thresh, erode_iters)
            scores.append(list(score_iter))
        scores = numpy.array(scores)
        lifespans = estimate_lifespans.states_to_lifespans(states, ages)
        num_worms, num_timepoints = scores.shape
        assert len(ages) == num_timepoints
        ref_ages = numpy.tile(ages, num_worms) # repeat ages for each worm. all_ages.shape == flat_scores.shape
        ref_scores = scores.flatten()
        ref_states = states.flatten()
        try:
            states_out = estimate_lifespans.simple_hmm(scores, ages,
                ref_lifespans, ref_ages, ref_scores, ref_states, lifespan_sigma)[0]
            lifespans_out = estimate_lifespans.cleanup_lifespans(
                estimate_lifespans.states_to_lifespans(states_out, ages), ages)
            r = stats.pearsonr(lifespans, lifespans_out)[0]
        except:
            r = 0
        if not numpy.isfinite(r):
            r = 0
        live_scores = scores[states==1]
        dead_scores = scores[states==0]
        t = stats.ttest_ind(live_scores, dead_scores, equal_var=False).statistic
        ks = compare_distributions.ks_statistic(live_scores, dead_scores)
        print('scoring: ', i, n, min_feature, max_feature, high_thresh, low_thresh, erode_iters, t, ks)
        return live_scores, dead_scores

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

def test_grid(workers, base_dirs, total_wells, image_dpi, grid, order_by='t'):
    base_dirs = list(base_dirs)
    grid = grid_search.ParameterGrid(grid)
    good_params = [param_set for param_set in grid if param_set['high_thresh'] >= param_set['low_thresh'] and
        (param_set['min_feature'] is None or param_set['max_feature'] > param_set['min_feature'])]
    n = len(good_params)
    all_scores = []
    for base_dir in base_dirs:
        well_names, states, ages = prep_inputs(base_dir, total_wells//len(base_dirs))
        well_images, well_mask = load_images(well_names, base_dir)
        evaluator = joblib.Parallel(n_jobs=workers, pre_dispatch='all', batch_size=1)
        scores = evaluator(joblib.delayed(score_images)(i, n, well_images, well_mask, states, ages, image_dpi, **param_set)
            for i, param_set in enumerate(good_params))
        all_scores.append(scores)
    # all-scores is a list of len(base_dirs) containing a list of (live_scores, dead_scores) pairs for each grid position
    # want to concatenate these scores across the different directories...
    score_lists = zip(*all_scores)
    concatenated_scores = []
    for scores in score_lists:
        live_scores, dead_scores = zip(*scores)
        concatenated_scores.append((numpy.concatenate(live_scores), numpy.concatenate(dead_scores)))
    assert len(concatenated_scores) == len(good_params)
    live_len = len(concatenated_scores[0][0])
    dead_len = len(concatenated_scores[0][1])
    for live_scores, dead_scores in concatenated_scores[1:]:
        assert len(live_scores) == live_len
        assert len(dead_scores) == dead_len
    return parse_scores(concatenated_scores, good_params, order_by)

def parse_scores(scores, good_params, order_by='t'):
    stats_out = []
    params = []
    master_keys = None
    for (live_scores, dead_scores), param_set in zip(scores, good_params):
        t = stats.ttest_ind(live_scores, dead_scores, equal_var=False).statistic
        ks = compare_distributions.ks_statistic(live_scores, dead_scores)
        stats_out.append((t, ks))
        keys = sorted(param_set.keys())
        if master_keys is None:
            master_keys = keys
        else:
            assert master_keys == keys
        params.append([param_set[key] for key in keys])
    stats_out = pd.DataFrame(stats_out)
    stats_out.columns = ['t', 'ks']
    params = pd.DataFrame(params)
    params.columns = master_keys
    order = stats_out[order_by].argsort()
    stats_out, params = stats_out.iloc[order], params.iloc[order]
    index = list(range(len(scores)))
    stats_out.index = index
    params.index = index
    scores_out  = [scores[i] for i in order]
    return scores_out, stats_out, params



