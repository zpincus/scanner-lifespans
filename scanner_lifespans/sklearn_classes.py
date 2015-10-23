## sklearn interface to well-scoring, for parameter fitting etc.
import numpy
from sklearn import base
from sklearn.externals import joblib
import pathlib

from zplib.scalar_stats import kde

from . import score_wells
from . import estimate_lifespans
from . import run_analysis

class BaseTransform(base.BaseEstimator, base.TransformerMixin):
    def fit(X, y=None):
        pass

class LoadImages(BaseTransform):
    def transform(self, X):
        well_names, base_dir = X
        base_dir = pathlib.Path(base_dir)
        well_mask = freeimage.read(str(base_dir / 'well_mask.png')) > 0
        well_images = []
        for well_name in well_names:
            all_images = sorted(base_dir.glob('*/well_images/'+well_name+'-*.png'))
            for date_dir, images_for_date in itertools.group_by(all_images, lambda path: path.parent.parent):
                images = [freeimage.read(str(image)) for image in images_for_date]
                well_images.append(images)
    return well_images, well_mask

class CleanImages(BaseTransform):
    def __init__(self, cache_dir, image_dpi, local_max_percentile):
        self.cache_dir = cache_dir
        self.local_max_percentile = local_max_percentile
        self.image_dpi = image_dpi

    def transform(self, X):
        well_images, well_mask = X
        memory = joblib.Memory(self.cache_dir)
        image_iter = memory.cache(score_wells.clean_image_sets)(well_images, self.local_max_percentile, self.image_dpi)
        return list(image_iter), well_mask

class DiffImages(BaseTransform):
    def __init__(self, cache_dir, min_feature, max_feature, image_dpi):
        self.cache_dir = cache_dir
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.image_dpi = image_dpi

    def transform(self, X):
        cleaned_images, well_mask = X
        memory = joblib.Memory(self.cache_dir)
        image_iter = memory.cache(score_wells.difference_image_sets)(cleaned_images, self.min_feature, self.max_feature, self.image_dpi)
        return list(image_iter), well_mask

class ScoreImages(BaseTransform):
    def __init__(self, cache_dir, high_thresh, low_thresh, erode_iters, image_dpi):
        self.cache_dir = cache_dir
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.erode_iters = erode_iters
        self.image_dpi = image_dpi

    def transform(self, X):
        diff_images, well_mask = X
        memory = joblib.Memory(self.cache_dir)
        score_iter = memory.cache(score_wells.score_image_sets)(diff_images, well_mask, self.high_thresh, self.low_thresh, self.erode_iters, self.image_dpi)
        return list(score_iter)

class EvaluateScores(base.BaseEstimator):
    def score(X, y):
        live_scores = X[y==1]
        dead_scores = X[y==0]
        return kde.ks_statistic(live_scores, dead_scores)

def prep_inputs(base_dir, total_wells):
    data = run_analysis.load_data(base_dir)
    well_mask = data.eval_last_alive_indices != None
    wells_included = well_mask.sum()
    if total_wells < wells_included:
        mask_mask = numpy.zeros(wells_included, dtype=bool)
        mask_mask[:total_wells] = True
        well_mask[well_mask] = numpy.random.permutation(mask_mask)
    well_names = data.well_names[well_mask]
    states = estimate_lifespans.last_alive_indices_to_states(data.data.eval_last_alive_indices, len(data.dates))[well_mask]
    #states.shape = num_wells, num_timepoints
    # flatten so all timepoints for each well are contiguous
    states = states.ravel(order='C')
    X = well_names, base_dir
    y = states
    return X, y
