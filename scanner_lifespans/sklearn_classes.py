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

def load_images(well_names, base_dir, rescale):
    base_dir = pathlib.Path(base_dir)
    well_mask = freeimage.read(str(base_dir / 'well_mask.png')) > 0
    well_images = []
    for well_name in well_names:
        all_images = sorted(base_dir.glob('*/well_images/'+well_name+'-*.png'))
        date_images = []
        for date_dir, images_for_date in itertools.groupby(all_images, lambda path: path.parent.parent):
            date_images.append([freeimage.read(str(image)) for image in images_for_date])
        if rescale:
            date_images = score_wells.rescale_images(date_images, well_mask)
        well_images.append(date_images)
    return numpy.array(well_images), well_mask

def score_images(well_images, well_mask, image_dpi, min_feature, max_feature, high_thresh, low_thresh, erode_iters):
    scores = []
    for date_images in well_images:
        date_scores = score_wells.score_wells(date_images, well_mask, image_dpi, min_feature, max_feature, high_thresh, low_thresh, erode_iters, rescale=False)
        scores.append(date_scores)
    return numpy.array(scores)

def grid_score_image_dir(base_dir, total_wells, image_dpi, grid, rescale, workers, seed):
    print('num_tasks', len(grid))
    numpy.random.seed(seed)
    well_names, states, ages = prep_inputs(base_dir, total_wells)
    well_images, well_mask = load_images(well_names, base_dir, rescale)
    evaluator = joblib.Parallel(n_jobs=workers, pre_dispatch='all', batch_size=1, verbose=10)
    grid_scores = evaluator(joblib.delayed(score_images)(well_images, well_mask, image_dpi, **param_set)
        for param_set in grid)
    return grid_scores, states, ages

def compute_lifespans(ref_lifespans, image_dir_results, workers):
    lifespan_estimator = estimate_lifespans.SimpleHMMEstimator(ref_lifespans, lifespan_sigma=3)
    dir_grid_scores, dir_states, dir_ages = zip(*image_dir_results)
    dir_lifespans = [estimate_lifespans.cleanup_lifespans(estimate_lifespans.states_to_lifespans(states, ages), ages)
        for states, ages in zip(dir_states, dir_ages)]
    evaluator = joblib.Parallel(n_jobs=workers, pre_dispatch='all', batch_size=1, verbose=10 if workers > 1 else 0)
    grid_lifespans_est = evaluator(joblib.delayed(compute_lifespans_for_dirs)(dir_scores, dir_states, dir_ages, lifespan_estimator)
        for dir_scores in zip(*dir_grid_scores))
    # dir_lifespans: list of len(image_dir_results) containing list of true lifespans for each animal in the image_dir
    # grid_lifespans_est: list of len(grid) containing lists of len(image_dir_results) containing list of estimated lifespans for each animal
    return dir_lifespans, grid_lifespans_est

def compute_lifespans_for_dirs(dir_scores, dir_states, dir_ages, lifespan_estimator):
    all_ages = []
    all_scores = []
    all_states = []
    for scores, states, ages in zip(dir_scores, dir_states, dir_ages):
        num_worms, num_timepoints = scores.shape
        assert len(ages) == num_timepoints
        all_ages.extend(numpy.tile(ages, num_worms)) # repeat ages for each worm. all_ages.shape == flat_scores.shape
        all_scores.extend(scores.flatten())
        all_states.extend(states.flatten())
    all_ages = numpy.array(all_ages)
    all_scores = numpy.array(all_scores)
    all_states = numpy.array(all_states)
    scores_live = all_scores[all_states==1]
    scores_dead = all_scores[all_states==0]
    could_clean = cleanup_scores(scores_live, scores_dead)
    if not could_clean:
        # achieved perfect separation... no need for HMM
        dir_lifespans_est = dir_lifespans
    else:
        obs_estimator = estimate_lifespans.hmm.ObservationProbabilityEstimator([scores_dead, scores_live])
        dir_lifespans_est = []
        for scores, ages in zip(dir_scores, dir_ages):
            p_obses = [obs_estimator(score_series) for score_series in scores]
            p_initial = lifespan_estimator.p_initial(ages[0])
            p_transition = lifespan_estimator.p_transition(ages)
            states = numpy.array([estimate_lifespans.hmm.viterbi(p_obs, p_transition, p_initial) for p_obs in p_obses])
            lifespans_est = estimate_lifespans.cleanup_lifespans(estimate_lifespans.states_to_lifespans(states, ages), ages)
            dir_lifespans_est.append(lifespans_est)
    return dir_lifespans_est


def cleanup_scores(scores_live, scores_dead):
    if numpy.all(scores_dead == scores_dead[0]):
        # problem if all dead scores are the same. Add some artificial jitter
        live_scores = scores_live[all_states == 0]
        if numpy.all(scores_live == scores_live[0]):
            return False
        min_live = scores_live[scores_live > scores_dead[0]].min()
        scores_dead += numpy.random.normal(scale=min_live/100, size=len(scores_dead))
    return True

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

def test_grid(workers, base_dirs, total_wells, image_dpi, grid, ref_lifespans, rescale):
    base_dirs = list(base_dirs)
    wells_per_dir = total_wells // len(base_dirs)
    grid = grid_search.ParameterGrid(grid)
    if len(base_dirs) < workers:
        dir_workers = 1
        grid_workers = workers
    else:
        dir_workers = workers
        grid_workers = 1
    good_params = [param_set for param_set in grid if
        (param_set['low_thresh'] is None or param_set['high_thresh'] >= param_set['low_thresh']) and
        (param_set['min_feature'] is None or param_set['max_feature'] > param_set['min_feature'])]
    evaluator = joblib.Parallel(n_jobs=dir_workers, pre_dispatch='all', batch_size=1, verbose=10 if dir_workers > 1 else 0)
    image_dir_results = evaluator(joblib.delayed(grid_score_image_dir)(base_dir, wells_per_dir, image_dpi, grid, rescale, grid_workers, seed=i)
        for i, base_dir in enumerate(base_dirs))
    dir_lifespans, grid_lifespans_est = compute_lifespans(ref_lifespans, image_dir_results, workers)
    return parse_scores(dir_lifespans, grid_lifespans_est, good_params)

def parse_scores(dir_lifespans, grid_lifespans_est, good_params):
    data = []
    master_keys = None
    grid_single_r2s = []
    all_lifespans = numpy.concatenate(dir_lifespans)
    for dir_lifespans_est, param_set in zip(grid_lifespans_est, good_params):
        all_lifespans_est = numpy.concatenate(dir_lifespans_est)
        r2 = _safe_r2(all_lifespans, all_lifespans_est)
        dir_r2s = []
        for lifespans, lifespans_est in zip(dir_lifespans, dir_lifespans_est):
            dir_r2s.append(_safe_r2(lifespans, lifespans_est))
        grid_single_r2s.append(dir_r2s)
        keys = sorted(param_set.keys())
        if master_keys is None:
            master_keys = keys
        else:
            assert master_keys == keys
        row = [param_set[key] for key in keys]
        data.append(row + [r2])
    data = pd.DataFrame(data)
    data.columns = master_keys + ['r2']
    order = data['r2'].argsort()
    data = data.iloc[order]
    data.index = list(range(len(good_params)))
    grid_lifespans_est = [grid_lifespans_est[i] for i in order]
    grid_single_r2s = [grid_single_r2s[i] for i in order]
    return data, grid_single_r2s, grid_lifespans_est, dir_lifespans



