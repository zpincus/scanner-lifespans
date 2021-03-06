import numpy
import pathlib
import re
import datetime
import collections

import freeimage
from zplib import util

from . import extract_wells
from . import score_wells
from . import estimate_lifespans
from .util import ROW_NAMES_384, COL_NAMES_384, BackgroundRunner

HOLLY_NAME_PARAMS = dict(
    image_glob='*.tif',
    date_regex=r'^\d{4}-\d{2}-\d{2}',
    date_format='%Y-%m-%d'
)

HOLLY_PLATE_PARAMS = dict(
    x_names=ROW_NAMES_384,
    y_names=COL_NAMES_384,
    x_names_first=True,
    exclude_names=('A23', 'A24', 'B23', 'B24')
)

HOLLY_IMAGE_SCORE_PARAMS = dict(
    image_dpi=2400,
    min_feature=50, # in microns
    max_feature=250, # in microns
    high_thresh=5,
    low_thresh=2.5,
    erode_iters=1
)

HOLLY_IMAGE_SCORE_PARAMS_new = dict(
    image_dpi=2400,
    min_feature=40, # in microns
    max_feature=125, # in microns
    high_thresh=2.75,
    low_thresh=None,
    erode_iters=2
)


## top-level functions
def run_analysis(in_dir, out_dir, age_at_first_scan, name_params, plate_params, score_params,
      training_data, re_extract=False, re_score=False, max_workers=None):
    """Convenience function for running complete analysis. See process_image_dir()
    and calculate_lifespans() for description of the parameters."""
    was_error = process_image_dir(in_dir, out_dir, age_at_first_scan, name_params, plate_params, score_params, re_extract, re_score, max_workers)
    if not was_error:
        calculate_lifespans(out_dir, training_data)
        data = load_data(out_dir)
        still_alive = numpy.isnan(data.lifespans)
        live_count = sum(still_alive)
        if live_count > 0:
            print('{} individual{} estimated to still be alive.'.format(live_count, 's are' if live_count > 1 else ' is'))
            if live_count < 10:
                print('Wells with live worms:')
                for alive, well in zip(still_alive, data.well_names):
                    if alive:
                        print('  '+well)
        else:
            print('The last survivor was estimated to have died {} days ago.'.format(data.ages[-1] - data.lifespans.max()))

def process_image_dir(in_dir, out_dir, age_at_first_scan, name_params, plate_params, score_params, re_extract=False, re_score=False, max_workers=None):
    """Extract well images from scanned plate images and score worm movement.

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
    re_extract: if True, don't skip well-image extraction even if there are existing images.
    re_score: if True, don't skip scoring well images even if there are existing scores.
        re_extract implies re_score. If the former is True, scoring will be re-done
        even if re_score is False.
    max_workers: maximum number of image-extraction jobs to run in parallel. If None,
        then use all CPUs that the machine has. For debugging, use 1.

    Returns: whether any of the jobs caused an error.
    """
    out_dir = util.get_dir(out_dir)
    image_sets = parse_inputs(in_dir, **name_params)
    dates = sorted(image_sets.keys())
    make_well_mask(out_dir, image_sets[dates[0]][0], ignore_previous=re_extract) # maks mask from first image on first day -- least junk-filled
    runner = BackgroundRunner(max_workers)
    if re_extract:
        re_score = True
    for date in dates:
        out_dir_for_date = out_dir / date.isoformat()
        age = (date - dates[0]).days + age_at_first_scan
        image_files = image_sets[date]
        runner.submit(process_image_set, image_files, out_dir_for_date, date, age,
            plate_params, score_params, re_extract, re_score)
    results, was_error, error_indices, cancelled_indices = runner.wait()
    for i in error_indices:
        print("Error processing images for date {}:".format(dates[i]))
        print(results[i])
    if not was_error:
        aggregate_scores(out_dir)
    return was_error

def calculate_lifespans(scored_dir, training_data):
    """Once well images have been scored, estimate worm lifespans.

    Parameters:
    scored_dir: corresponds to out_dir parameter to process_image_dir() --
        the parent directory of all of the extracted and scored images.
    training_data: paths to one or more training data files with calibration information.
    """
    scored_dir = pathlib.Path(scored_dir)
    data = load_data(scored_dir)
    training = load_training_data(training_data)
    states = estimate_lifespans.simple_hmm(data.scores, data.ages,
        training.lifespans, training.ages, training.scores, training.states,
        lifespan_sigma=6)[0]
    # states = estimate_lifespans.estimate_lifespans(data.scores, data.ages, training.states, training.scores, training.ages)
    lifespans = estimate_lifespans.states_to_lifespans(states, data.ages)
    last_alive_indices = estimate_lifespans.states_to_last_alive_indices(states)
    lifespans_out = [('well name', 'lifespan')]+[(well_name, str(lifespan)) for well_name, lifespan in zip(data.well_names, lifespans)]
    util.dump_csv(lifespans_out, scored_dir/'lifespans.csv')
    util.dump(scored_dir/'lifespans.pickle', well_names=data.well_names, ages=data.ages, states=states,
        lifespans=lifespans, last_alive_indices=last_alive_indices)

def evaluate_lifespans(scored_dir):
    """Once well images have been scored, manually evaluate lifespans.

    If the lifespans have been computationally esitmated, that data will be
    loaded; if not, then all wells will be set to "no lifespan" (i.e. "dead on
     arrival"), but that can of course be manually changed for each well of
     interest.

    Parameters:
    scored_dir: corresponds to out_dir parameter to process_image_dir() --
        the parent directory of all of the extracted and scored images.
    """
    # only import GUI stuff here, as it can play havoc with use with other GUIs (e.g. matplotlib)
    from . import evaluate_lifespans
    data = load_data(scored_dir)
    if not hasattr(data, 'last_alive_indices'):
        # Lifespans have not been estimated. This will allow manual annotation.
        data.last_alive_indices = [None] * len(data.well_names)
    evaluator = evaluate_lifespans.DeathDayEvaluator(scored_dir, data.ages, data.last_alive_indices, data.well_names)
    return evaluator

def evaluate_statuses(extracted_dir):
    """Once well images have been extracted, manually count the worms/well.

    Parameters:
    extracted_dir: corresponds to out_dir parameter to process_image_dir() --
        the parent directory of all of the extracted images.
    """
    # only import GUI stuff here, as it can play havoc with use with other GUIs (e.g. matplotlib)
    from . import evaluate_lifespans
    data = load_data(extracted_dir)
    if hasattr(data, 'eval_last_alive_indices'):
        # Lifespans have been manually evaluated. Only look at wells with index==None
        well_names = [wn for wn, lai in zip(data.well_names, data.eval_last_alive_indices) if lai == None]
        no_worms_i = evaluate_lifespans.DOAEvaluator.status_codes.index('No worms')
        statuses = [no_worms_i] * len(well_names)
    else:
        well_names = data.well_names
        statuses = None
    evaluator = evaluate_lifespans.DOAEvaluator(scored_dir, data.dates[0].isoformat(), well_names,
        status_codes=['One worm', 'No worms', 'Many worms', 'DOA'], statuses=statuses)
    return evaluator


def make_training_data(scored_dir, training_data, annotation_file=None):
    """Given a scored directory and a pickle file of manual annotation data,
    create a file of training data for the Hidden Markov Model for lifespan
    estimation.

    If manually provided well-status information is present, this function will
    skip empty wells and those with multiple worms. If no status information is
    available, this function will skip all wells which were ignored in the
    manual annotation data (empty / DOA / multiple worms).

    Parameters:
    scored_dir: corresponds to out_dir parameter to process_image_dir() --
        the parent directory of all of the extracted and scored images.
    training_data: path to write training data file with calibration information.
    annotation_file: if None (default), use 'evaluated_lifespans.csv' in the
        scored_dir. Otherwise load a custom file path.
    """
    data = load_data(scored_dir)
    if annotation_file is None:
        annotation_file = pathlib.Path(scored_dir) / 'evaluated_lifespans.csv'
    csv_well_names, csv_lifespans = read_lifespan_annotation_csv(annotation_file)
    good_well_names, good_lifespans = [], []
    if hasattr(data, 'eval_well_statuses'):
        statuses = dict(zip(data.well_names, data.eval_well_statuses))
    else:
        statuses = {}
    for name, lifespan in zip(csv_well_names, csv_lifespans):
        # If statuses are present, include DOA worms and living worms
        # in the training data. Exclude empty and  multi-worm wells.
        # If statuses are not present, exclude all wells with '-1' lifespans, which
        # refers to any of empty/DOA/multi-worm states.
        if (name in statuses and statuses[name] == 'DOA') or lifespan != -1 :
            good_well_names.append(name)
            good_lifespans.append(lifespan)
    states = estimate_lifespans.lifespans_to_states(good_lifespans, data.ages)

    # it could be that the wells in the CSV are only a subset of the wells in the data,
    # so find the indices in the data for just these wells.
    indices_of_data_wells = {well:i for i, well in enumerate(data.well_names)}
    good_well_indices = [indices_of_data_wells[well] for well in good_well_names]
    scores = data.scores[good_well_indices]
    util.dump(training_data, states=states, ages=data.ages, scores=scores)


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

def make_well_mask(out_dir, image_file, ignore_previous=False):
    """Calculate and store well mask if necessary.

    Parameters:
    out_dir: directory where well_mask.png should exist or be created.
    image_file: path to an image to create the mask from, if it doesn't exist.
    ignore_previous: if True, re-make mask even if it alredy exists on disk.
    """
    out_dir = pathlib.Path(out_dir)
    well_mask_f = out_dir / 'well_mask.png'
    if ignore_previous or not well_mask_f.exists():
        image = freeimage.read(image_file)
        if image.dtype == numpy.uint16:
            image = (image >> 8).astype(numpy.uint8)
        well_mask = extract_wells.get_well_mask(image)
        freeimage.write((well_mask * 255).astype(numpy.uint8), str(well_mask_f))

def process_image_set(image_files, out_dir, date, age, plate_params, score_params, re_extract, re_score):
    """Do all processing for a given date's images: extract the wells to
    separate image files, and then score each well's images.

    See extract_image_set() and score_image_set() for a description of the parameters.
    """
    extract_image_set(image_files, out_dir, date, age, plate_params, ignore_previous=re_extract)
    score_image_set(out_dir, score_params, ignore_previous=re_score)

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
    if metadata.exists() and not ignore_previous:
        return
    images = []
    print('extracting images for {}'.format(out_dir))
    well_mask = freeimage.read(str(out_dir.parent / 'well_mask.png')) > 0
    for image_file in image_files:
        image = freeimage.read(image_file)
        if image.dtype == numpy.uint16:
            image = (image >> 8).astype(numpy.uint8)
        images.append(image)
    well_names, well_images, well_centroids = extract_wells.extract_wells(images, well_mask, **plate_params)
    well_dir = util.get_dir(out_dir / 'well_images')
    for well_name, well_image_set in zip(well_names, well_images):
        for i, image in enumerate(well_image_set):
            freeimage.write(image, str(well_dir/well_name)+'-{}.png'.format(i))
    util.dump(metadata, date=date, age=age, well_names=well_names, well_centroids=well_centroids)

def score_image_set(out_dir, score_params, ignore_previous=False):
    """Score wells for a single day's scanned images.

    Parameters:
    out_dir: directory in which well_images directory is found, and to which score
        data will be written.
    score_params: configuration information for scoring wells for movement.
        This must be a parameter dictionary suitable to pass to score_wells.score_wells()
    ignore_previous: if False, and stored results already exist, skip processing
    """
    out_dir = pathlib.Path(out_dir)
    score_file = out_dir / 'scores.pickle'
    if score_file.exists() and not ignore_previous:
        return
    print('scoring images for {}'.format(out_dir))
    well_names = util.load(out_dir / 'metadata.pickle').well_names
    well_mask = freeimage.read(str(out_dir.parent / 'well_mask.png')) > 0
    well_dir = out_dir / 'well_images'
    well_images = []
    for well_name in well_names:
        images = [freeimage.read(str(image)) for image in sorted(well_dir.glob(well_name+'-*.png'))]
        well_images.append(images)
    well_scores = score_wells.score_wells(well_images, well_mask, **score_params)
    util.dump(score_file, well_names=well_names, well_scores=well_scores)
    scores_out = [[name, str(score)] for name, score in zip(well_names, well_scores)]
    util.dump_csv(scores_out, out_dir / 'scores.csv')

def aggregate_scores(out_dir):
    """Once all images have been scored, aggregate the per-image-set (i.e. per-day)
    score data to a single file for the whole experiment.
    """
    out_dir = pathlib.Path(out_dir)
    well_names = None
    all_scores = {}
    for scorefile in out_dir.glob('*/scores.pickle'): # find all such files below outdir
        scores = util.load(scorefile)
        data = util.load(scorefile.parent / 'metadata.pickle')
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
    util.dump_csv(data_out, out_dir/'scores.csv')
    util.dump(out_dir / 'scores.pickle', dates=dates, ages=ages, well_names=well_names, scores=scores)

def load_training_data(training_data_files):
    """Load data from one or more training data files. If multiple files are provided,
    merge their data."""
    states, scores, ages, lifespans = [], [], [], []
    if isinstance(training_data_files, (str, pathlib.Path)):
        training_data_files = [training_data_files]
    for training_data in training_data_files:
        training = util.load(training_data)
        # training.states, training.scores are shape-(n_worms, n_timepoints) arrays
        # training.ages is n_timepoints long. We want to flatten both into lists
        states.extend(training.states.flat)
        scores.extend(training.scores.flat)
        ages.extend(list(training.ages) * training.states.shape[0])
        training_lifespans = estimate_lifespans.states_to_lifespans(training.states, training.ages)
        lifespans.extend(estimate_lifespans.cleanup_lifespans(training_lifespans, training.ages))
    training_out = util.Data(states=numpy.array(states), scores=numpy.array(scores),
        ages=numpy.array(ages), lifespans=numpy.array(lifespans))
    return training_out

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
        last_alive_indices
    If manually-annotated well-statuses from 'statuses.pickle' are found, also:
        eval_well_statuses
    If manually-annotated lifespans from 'evaluations.pickle' are found, also:
        eval_last_alive_indices
    """
    scored_dir = pathlib.Path(scored_dir)
    data = util.Data()
    scores = util.load(scored_dir / 'scores.pickle')
    for name in ('dates', 'ages', 'well_names', 'scores'):
        setattr(data, name, getattr(scores, name))
    lifespan_data = scored_dir / 'lifespans.pickle'
    if lifespan_data.exists():
        lifespans = util.load(lifespan_data)
        assert numpy.all(scores.ages == lifespans.ages) and scores.well_names == lifespans.well_names
        for name in ('states', 'lifespans', 'last_alive_indices'):
            setattr(data, name, getattr(lifespans, name))
    eval_data = scored_dir / 'evaluations.pickle'
    if eval_data.exists():
        evaluated = util.load(eval_data)
        data.eval_last_alive_indices = evaluated.last_alive_indices
        data.eval_lifespans = estimate_lifespans.last_alive_indices_to_lifespans(evaluated.last_alive_indices, data.ages)
    status_data = scored_dir / 'statuses.pickle'
    if status_data.exists():
        statuses = util.load(status_data)
        status_codes = [statuses.status_codes[i] for i in statuses.statuses]
        status_dict = dict(zip(statuses.well_names, status_codes))
        all_statuses = []
        for well_name in data.well_names:
            if well_name in status_dict:
                all_statuses.append(status_dict[well_name])
            else:
                all_statuses.append(statuses.status_codes[0])
        data.eval_well_statuses = all_statuses
    return data

def read_lifespan_annotation_csv(csv):
    """ Read a csv of lifespans to lists of well names and lifespans."""
    well_names, lifespans = [], []
    csv = util.load_csv(csv)
    for line in csv[1:]: # skip header line
        well_names.append(line[0])
        lifespans.append(float(line[1]))
    return well_names, lifespans