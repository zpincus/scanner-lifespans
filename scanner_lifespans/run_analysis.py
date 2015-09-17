import numpy
import pickle
import pathlib
import re
import datetime
import collections
import concurrent.futures as futures
import freeimage

import extract_wells
import score_wells
import estimate_lifespans
#import evaluate_lifespans

ROW_NAMES_384 = 'ABCDEFGHIJKLMNOP'
COL_NAMES_384 = ['{:02d}'.format(i) for i in range(1, 25)]

HOLLY_NAME_PARAMS = dict(
    image_glob='*.tif',
    date_regex=r'^\d{4}-\d{2}-\d{2}',
    date_format='%Y-%m-%d'
)

HOLLY_OLD_NAME_PARAMS = dict(
    image_glob='*.tif',
    date_regex=r'^[A-Z][a-z]{2}\d{1,2}',
    date_format='%b%d'
)

HOLLY_PLATE_PARAMS = dict(
    x_names=ROW_NAMES_384,
    y_names=COL_NAMES_384,
    x_names_first=True,
    exclude_names=('A23', 'A24', 'B23', 'B24')
)

HOLLY_OLD_PLATE_PARAMS = dict(
    x_names=ROW_NAMES_384,
    y_names=COL_NAMES_384,
    x_names_first=True,
    exclude_names=None
)


HOLLY_IMAGE_SCORE_PARAMS = dict(
    image_dpi=2400,
    min_feature=40, # in microns
    max_feature=400, # in microns
    high_thresh=0.12,
    low_thresh=0.08,
    erode_iters=0,
    local_max_percentile=80
)

## top-level functions
def run_analysis(in_dir, out_dir, age_at_first_scan, name_params, plate_params, score_params,
      training_data, max_workers=None):
    """Convenience function for running complete analysis. See process_image_dir()
    and estimate_lifespans() for description of the parameters."""
    process_image_dir(in_dir, out_dir, age_at_first_scan, name_params, plate_params, score_params, max_workers)
    estimate_lifespans(out_dir, training_data)

def process_image_dir(in_dir, out_dir, age_at_first_scan, name_params, plate_params, score_params, max_workers=None):
    """Estimate lifespans from scanned plate images.

    Parameters:
    in_dir: path to directory of scanned images.
    out_dir: path to write out the extracted images and metadata.
    age_at_first_scan: age in days of the first image scan in in_dir
    name_params: information for finding image dates from file names: a dict with
        keys 'image_glob', 'date_regex', and 'date_format', as per extract_well_images()
    plate_params: configuration information for extracting wells from the plates.
        This must be a parameter dictionary suitable to pass to extract_wells.extract_wells()
    score_params: configuration information for scoring wells for movement.
        This must be a parameter dictionary suitable to pass to score_wells.score_wells()
    max_workers: maximum number of image-extraction jobs to run in parallel. If None,
        then use all CPUs that the machine has. For debugging, use 1.
    """
    out_dir = _get_dir(out_dir)
    image_sets = parse_inputs(in_dir, **name_params)
    dates = sorted(image_sets.keys())
    make_well_mask(out_dir, image_sets[dates[0]][0]) # maks mask from first image on first day -- least junk-filled
    runner = BackgroundRunner(max_workers)
    for date in dates:
        out_dir_for_date = out_dir / date.isoformat()
        age = (date - dates[0]).days + age_at_first_scan
        image_files = image_sets[date]
        runner.submit(process_image_set, image_files, out_dir_for_date, date, age,
            plate_params, score_params)
    errors = runner.wait()
    for i, error in enumerate(errors):
        if error is not None:
            print("Error processing images for date {}:".format(dates[i]))
            print(error)

def estimate_lifespans(scored_dir, training_data):
    """Once well images have been scored, estimate worm lifespans.

    Parameters:
    scored_dir: corresponds to out_dir parameter to process_image_dir() --
        the parent directory of all of the extracted and scored images.
    training_data: path to training_data.pickle file with calibration information.
    """
    scored_dir = pathlib.Path(scored_dir)
    dates, ages, well_names, scores = aggregate_scores(scored_dir)
    training = _load(training_data)

    states = estimate_lifespans.estimate_states(scores, ages, training.states, training.scores, training.ages)
    lifespans, last_alive_i = estimate_lifespans.states_to_lifespans(states, ages)
    #TODO: only store last alive indices, and have helper functions to generate lifespans
    last_alive_dates = [dates[i] for i in last_alive_i]
    lifespans_out = [(well_name, str(lifespan)) for well_name, lifespan in zip(well_names, lifespans)]
    _dump_csv(lifespans_out, out_dir/'lifespans.csv')
    last_out = [(well_name, date.isoformat() if ld else '') for well_name, date in zip(well_names, last_alive_dates)]
    _dump_csv(last_out, out_dir/'last_alive.csv')
    _dump(out_dir/'lifespans.pickle', well_names=well_names, ages=ages, states=states, lifespans=lifespans, last_alive_dates=last_alive_dates)

def evaluate_lifespans(scored_dir, ris_widget):
    """Once well images have been scored, manually evaluate lifespans.

    If the lifespans have been computationally esitmated, that data will be
    loaded; if not, then all wells will be set to "no lifespan" (i.e. "dead on
     arrival"), but that can of course be manually changed for each well of
     interest.

    Parameters:
    scored_dir: corresponds to out_dir parameter to process_image_dir() --
        the parent directory of all of the extracted and scored images.
    training_data: path to training_data.pickle file with calibration information.
    """

    data = load_data(scored_dir)
    if not hasattr(data, 'last_alive_dates'):
        # Lifespans have not been estimated. This will allow manual annotation.
        data.last_alive_dates = [None] * len(data.well_names)
    evaluator = evaluate.DeathDayEvaluator(ris_widget, out_dir, data.dates, data.ages, data.last_alive_dates, data.well_names)
    return evaluator

def make_training_data(scored_dir, training_data, manual_annotation_csv):
    """Given a scored directory and a file containing manually annotated lifespans,
    write out the training_data file."""
    data = load_data(scored_dir)
    csv_well_names, csv_lifespans = read_lifespan_annotation_csv(manual_annotation_csv)
    states = estimate_lifespans.lifespan_to_states(csv_lifespans, data.ages)

    # it could be that the wells in the CSV are only a subset of the wells in the data,
    # so find the indices in the data for just these wells.
    indices_of_data_wells = {well:i for i, well in enumerate(data.well_names)}
    well_indices = [indices_of_data_wells[well] for well in csv_well_names]
    scores = data.scores[well_indices]
    _dump(training_data, states=states, ages=ages, scores=scores)


## Helper functions

def parse_inputs(in_dir, image_glob, date_regex, date_format):
    """
    Find wells in a directory of scanner images, group into sets of images by date.

    Parameters:
    in_dir: path to directory of scanned images.
    image_glob: glob-pattern to match the desired images in in_dir
    date_regex: regular expression to recognize the date from the image name
    date_format: datetime.strptime format expression to parse date from the part
        of the filename that matches date_regex.

    Returns: dictionary mapping date objects to lists of images from that date.
    """
    in_dir = pathlib.Path(in_dir)
    all_images = in_dir.glob(image_glob)
    image_sets = collections.defaultdict(list)
    for image_file in all_images:
        if image_file.name.startswith('.'): # ignore mac resource-files
            continue
        dates = re.findall(date_regex, image_file.name)
        if len(dates) != 1:
            print(dates, image_file.name)
            print("could not properly parse filename: {}".format(image_file))
            continue
        date = datetime.datetime.strptime(dates[0], date_format).date()
        if date.year == 1900:
            date = date.replace(year=datetime.date.today().year)
        image_sets[date].append(image_file)
    for image_files in image_sets.values():
        image_files.sort()
    return image_sets

def make_well_mask(out_dir, image_file):
    """Calculate and store well mask if necessary.

    Parameters:
    out_dir: directory where well_mask.png should exist or be created.
    image_file: path to an image to create the mask from, if it doesn't exist.
    """
    out_dir = pathlib.Path(out_dir)
    well_mask_f = out_dir / 'well_mask.png'
    if not well_mask_f.exists():
        image = freeimage.read(image_file)
        if image.dtype == numpy.uint16:
            image = (image >> 8).astype(numpy.uint8)
        well_mask = extract_wells.get_well_mask(image)
        freeimage.write((well_mask * 255).astype(numpy.uint8), str(well_mask_f))

class BackgroundRunner:
    """Class for running jobs in background processes. Does not collect
    the return value of the jobs! Use submit() to add jobs, and then
    wait() to get a list of the error states for each submitted job:
    either None for successful completion, or an exception object.

    Wait() will only wait until the first exception, and after that will
    attempt to cancel all pending jobs.

    If max_workers is 1, then just run the job in this process. Useful
    for debugging, where a proper traceback from a foreground exception
    can be helpful.
    """
    def __init__(self, max_workers):
        if max_workers == 1:
            self.executor = None
        else:
            self.executor = futures.ProcessPoolExecutor(max_workers)
        self.futures = []

    def submit(self, fn, *args, **kwargs):
        if self.executor:
            self.futures.append(self.executor.submit(fn, *args, **kwargs))
        else:
            fn(*args, **kwargs)

    def wait(self):
        if not self.executor:
            return []
        futures.wait(self.futures, return_when=futures.FIRST_EXCEPTION)
        # If there was an exception, cancel all the rest of the jobs.
        # If there was no exception, can "cancel" the jobs anyway, because canceling does
        # nothing if the job is done.
        for future in self.futures:
            future.cancel()
        errors = []
        for future in self.futures:
            if future.cancelled():
                errors.append(None)
            else:
                errors.append(future.exception())
        self.futures = []
        return errors

def process_image_set(image_files, out_dir, date, age, plate_params, score_params):
    """Do all processing for a given date's images: extract the wells to
    separate image files, and then score each well's images.

    See extract_image_set() and score_image_set() for a description of the parameters.
    """
    extract_image_set(image_files, out_dir, date, age, plate_params)
    score_image_set(out_dir, score_params, write_difference_images=False)

def extract_image_set(image_files, out_dir, date, age, plate_params, ignore_previous=False):
    """Find wells in a set of scanner images and extract each well into a separate image
    for further processing.

    Parameters:
    image_files: list of paths to a set of images.
    out_dir: path to write out the extracted images and metadata.
    date: date object referring to image scan date
    age: age in days of the worms in these images
    plate_params: configuration information for extracting wells from the plates.
        This must be a parameter dictionary suitable to pass to extract_wells.extract_wells()
    ignore_previous: if False, and stored results already exist, skip processing
    """
    out_dir = pathlib.Path(out_dir)
    metadata = out_dir / 'metadata.pickle'
    if not ignore_previous and metadata.exists():
        return
    images = []
    print('extracting images for {}'.format(date))
    well_mask = freeimage.read(str(out_dir.parent / 'well_mask.png')) > 0
    for image_file in image_files:
        image = freeimage.read(image_file)
        if image.dtype == numpy.uint16:
            image = (image >> 8).astype(numpy.uint8)
        images.append(image)
    image_centroids, well_names, well_images = extract_wells.extract_wells(images, well_mask, **plate_params)
    well_dir = _get_dir(out_dir / 'well_images')
    for well_name, well_image_set in zip(well_names, well_images):
        for i, image in enumerate(well_image_set):
            freeimage.write(image, str(well_dir/well_name)+'-{}.png'.format(i))
    _dump(metadata, date=date, age=age, well_names=well_names, image_centroids=image_centroids)

def score_image_set(out_dir, score_params, write_difference_images, ignore_previous=False):
    """Score wells for a single day's scanned images.

    Parameters:
    out_dir: directory in which well_images directory is found, and to which score
        data will be written.
    score_params: configuration information for scoring wells for movement.
        This must be a parameter dictionary suitable to pass to score_wells.score_wells()
    write_difference_images: if True, output the (large) motion difference images
    ignore_previous: if False, and stored results already exist, skip processing
    """
    out_dir = pathlib.Path(out_dir)
    score_file = out_dir / 'scores.pickle'
    if not ignore_previous and score_file.exists():
        return
    print('scoring images for {}'.format(str(out_dir)))
    well_names = _load(out_dir / 'metadata.pickle').well_names
    well_mask = freeimage.read(str(out_dir.parent / 'well_mask.png')) > 0
    well_dir = out_dir / 'well_images'
    well_images = []
    for well_name in well_names:
        images = [freeimage.read(str(image)) for image in sorted(well_dir.glob(well_name+'-*.png'))]
        well_images.append(images)
    diff_images, well_scores = score_wells.score_wells(well_images, well_mask,
        return_difference_images=write_difference_images, **score_params)
    if write_difference_images:
        diff_dir = _get_dir(out_dir / 'abs_diff_images')
        for well_name, diff_image_set in zip(well_names, diff_images):
            for i, image in enumerate(diff_image_set):
                freeimage.write(i, str(diff_dir/well_name)+'-{}.tif'.format(i))
    _dump(score_file, well_names=well_names, well_scores=well_scores)
    scores_out = [[name, str(score)] for name, score in zip(well_names, well_scores)]
    _dump_csv(scores_out, out_dir / 'scores.csv')

def rescore_images(extracted_dir, score_params, max_workers=None):
    """Calculate movement scores from previously-extracted well images.

    Only necessary to call if it is desired to change the scoring parameters
    from a previous call to process_image_dir, without re-extracting all of
    the images.

    Parameters:
    extracted_dir: corresponds to out_dir parameter to extract_well_images() --
        the parent directory of all of the extracted images.
    score_params: configuration information for scoring wells for movement.
        This must be a parameter dictionary suitable to pass to score_wells.score_wells()
    max_workers: maximum number of well-scoring jobs to run in parallel. If
        None, use all CPUs. Use 1 for debugging.
    """
    extracted_dir = pathlib.Path(extracted_dir)
    out_dirs = [metadata.parent for metadata in extracted_dir.glob('*/metadata.pickle')]
    runner = BackgroundRunner(max_workers)
    for out_dir in out_dirs:
        runner.submit(score_image_set, out_dir, score_params, ignore_previous=True, write_difference_images=False)
    errors = runner.wait()
    for i, error in enumerate(errors):
        if error is not None:
            print("Error scoring images for directory {}:".format(str(out_dirs[i])))
            print(error)

def aggregate_scores(out_dir):
    """Once all images have been scored, aggregate the per-image-set (i.e. per-day)
    score data to a single file for the whole experiment.
    """
    out_dir = pathlib.Path(out_dir)
    well_names = None
    all_scores = {}
    for scorefile in out_dir.glob('*/scores.pickle'): # find all such files below outdir
        scores = _load(scorefile)
        data = _load(scorefile.parent / 'metadata.pickle')
        assert data.well_names == scores.well_names
        if well_names is None:
            well_names = data.well_names
        else:
            assert well_names == data.well_names
        all_scores[data.date] = data.age, scores.well_scores
    assert len(all_scores) > 0 # makes sure there are files to score!
    dates, ages_and_scores = zip(*sorted(all_scores.items()))
    ages, scores = zip(*ages_and_scores)
    ages = numpy.array(ages)
    scores = numpy.array(scores).T
    data_out = [[''] + [d.isoformat() for d in dates]]
    data_out += [[''] + [str(a) for a in ages]]
    for well_name, score in zip(well_names, scores):
        data_out += [[well_name] + [str(s) for s in score]]
    _dump_csv(data_out, out_dir/'scores.csv')
    _dump(out_dir / 'scores.pickle', dates=dates, ages=ages, well_names=well_names, scores=scores)
    return dates, ages, well_names, scores

def load_data(scored_dir):
    """Load score data, and if available, lifespan data, from a processed directory.

    Returns a Data object with attributes:
        dates
        ages
        well_names
        scores
    and if lifespan data are found, also:
        states
        lifespans
        last_alive_dates
    """
    scored_dir = pathlib.Path(scored_dir)
    data = Data()
    scores = _load(scored_dir / 'scores.pickle')
    for name in ('dates', 'ages', 'well_names', 'scores'):
        setattr(data, name, getattr(scores, name))
    lifespan_data = scored_dir / 'lifespans.pickle'
    if lifespan_data.exists():
        lifespans = _load(lifespan_data)
        assert scores.ages == lifespans.ages and scores.well_names == lifespans.well_names
        for name in ('states', 'lifespans', 'last_alive_dates'):
            setattr(data, name, getattr(lifespans, name))
    return data

class Data:
    def __init__(self, **kwargs):
        """Add all keyword arguments to self.__dict__, which is to say, to
        the namespace of the class. I.e.:

        d = Data(foo=5, bar=6)
        d.foo == 5 # True
        d.bar > d.foo # True
        d.baz # AttributeError
        """
        self.__dict__.update(kwargs)

def _get_dir(path):
    """Create a directory at path if it does not already exist. Return a
    pathlib.Path object for that directory."""
    path = pathlib.Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    return path

def _dump(path, **data_dict):
    """Dump the keyword arguments into a dictionary in a pickle file."""
    path = pathlib.Path(path)
    with path.open('wb') as f:
        pickle.dump(data_dict, f)

def _load(path):
    """Load a dictionary from a pickle file into a Data object for
    attribute-style value lookup."""
    path = pathlib.Path(path)
    with path.open('rb') as f:
        return Data(**pickle.load(f))

def _dump_csv(data, path):
    """Write a list of lists to a csv file."""
    path = pathlib.Path(path)
    with path.open('w') as f:
        f.write('\n'.join(','.join(row) for row in data))

def _load_csv(csv):
    """Load a csv file to a list of lists."""
    path = pathlib.Path(path)
    data = []
    with path.open('w') as f:
        for line in f:
            data.append(line.split(','))
    return data

def read_lifespan_annotation_csv(csv):
    """ Read a csv of lifespans to lists of well names and lifespans."""
    well_names, lifespans = [], []
    csv = _load_csv(csv)
    for line in csv[1:]: # skip line zero as header
        well_names.append(line[0])
        lifespans.append(float(line[1]))
    return well_names, lifespans