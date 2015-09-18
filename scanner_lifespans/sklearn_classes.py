## sklearn interface to well-scoring, for parameter fitting etc.
from sklearn import base
from sklearn.externals import joblib
import pathlib

from . import score_wells
from . import estimate_lifespans

class ScoreWells(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, cache_dir, well_names, well_mask, image_dpi, min_feature, max_feature, local_max_percentile, high_thresh, low_thresh, erode_iters):
        self.cache_dir = cache_dir
        self.well_names = well_names
        self.well_mask = well_mask
        self.image_dpi = image_dpi
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.local_max_percentile = local_max_percentile
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.erode_iters = erode_iters

    @staticmethod
    def score_wells(image_dir, well_names, well_mask, image_dpi, min_feature, max_feature, local_max_percentile, high_thresh, low_thresh, erode_iters):
        image_dir = pathlib.Path(image_dir)
        for well_name in well_names:
            images = [freeimage.read(str(image)) for image in sorted(image_dir.glob(well_name+'-*.png'))]
            well_images.append(images)

        # just return well scores, not diff images.
        # This is to simplify caching b/c don't have to cache output images
        return score_wells.score_wells(well_images, well_mask, image_dpi, min_feature, max_feature, local_max_percentile, high_thresh, low_thresh, erode_iters)[1]

    def transform(image_dirs):
        memory = joblib.Memory(self.cache_dir)
        score_wells = memory.cache(self.score_wells)
        scores = []
        for image_dir in X:
             scores.append(self.score_wells(image_dir, self.well_names, self.well_mask, self.image_dpi, self.min_feature, self.max_feature, self.local_max_percentile, self.high_thresh, self.low_thresh, self.erode_iters))
