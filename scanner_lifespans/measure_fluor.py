import numpy
import pathlib
import collections
import freeimage
from scipy import ndimage
import multiprocessing

from zplib.image import mask
from zplib.scalar_stats import mcd
from zplib.image import polyfit

from . import find_worm
from . import evaluate_wormfinding
from .util import split_image_name, BackgroundRunner

def find_bf_worm_masks(image_dir, max_workers=None):
    image_dir = pathlib.Path(image_dir)
    runner = BackgroundRunner(max_workers)
    wells = []
    for image_path in sorted(image_dir.glob('*brightfield.*')):
        well, rest = split_image_name(image_path)
        wells.append(well)
        runner.submit(_find_worms_task, image_path, well)
    results, was_error, error_indices, cancelled_indices = runner.wait()
    for i in error_indices:
        print("Error processing images for well {}:".format(wells[i]))
        print(results[i])
    if not was_error:
        valid_wells = [well for well, is_valid in zip(wells, results) if is_valid]
        with (image_dir / 'valid_wells.txt').open('w') as f:
            f.write('\n'.join(sorted(valid_wells)))


def evaluate_bf_worm_masks(image_dir):
    from ris_widget import ris_widget
    rw = ris_widget.RisWidget()
    return evaluate_wormfinding.validate(image_dir, rw)

def measure_worms_from_bf_mask(image_dir, max_workers=None):
    image_dir = pathlib.Path(image_dir)
    valid_well_f = image_dir / 'valid_wells.txt'
    if valid_well_f.exists():
        with valid_well_f.open() as f:
            valid_wells = {line.strip() for line in f}
    else:
        valid_wells = None
    image_types = collections.defaultdict(list)
    for image_path in image_dir.glob('*'):
        if image_path.suffix not in ('.tif', '.tiff', '.png'):
            continue
        well, rest = split_image_name(image_path)
        if rest != 'brightfield' and not 'mask' in rest:
            if valid_wells is not None and well not in valid_wells:
                continue
            image_types[rest].append((well, image_path))
    runner = BackgroundRunner(max_workers)
    for image_type, wells_and_paths in image_types.items():
        wells_and_paths.sort() # sort by well order
        for well, image_path in wells_and_paths:
            runner.submit(_measure_worms_task, image_path, well)
        results, was_error, error_indices, cancelled_indices = runner.wait()
        wells, paths = zip(*wells_and_paths)
        for i in error_indices:
            print("Error measuring images for well {}:".format(wells[i]))
            print(results[i])
        if not was_error:
            out = image_dir / 'measured_{}.csv'.format(image_type)
            write_measures(results, wells, out)

def measure_incyte_images(image_dir, image_glob='*FITC*'):
    image_dir = pathlib.Path(image_dir)
    assert image_dir.exists()
    image_files = list(image_dir.glob(image_glob))
    well_names = []
    for image_file in sorted(image_files):
        row = image_file.name[0] # first letter is row, then ' - ', then two-digit col
        col = image_file.name[4:6]
        well = row + col
        well_names.append(well)
    data_rows = []
    for image_file in image_files:
        print(str(image_file))
        image = freeimage.read(image_file)
        worm_mask = find_worm.find_worm_from_fluorescence(image)
        data_rows.append(measure_fluorescence(image, worm_mask)[0])
    write_measures(data_rows, well_names, image_dir.with_suffix('.csv'))

#### Below are helper functions
def _find_worms_task(image_path, well):
    print('Finding worm '+well)
    image = freeimage.read(image_path)
    well_mask, edges, worm_mask = find_worm.find_worm_from_brightfield(image)
    freeimage.write(well_mask.astype(numpy.uint8)*255, image_path.parent / (well + '_well_mask.png'))
    freeimage.write(worm_mask.astype(numpy.uint8)*255, image_path.parent / (well + '_worm_mask.png'))
    return is_valid_mask(worm_mask)

def is_valid_mask(worm_mask):
    return 8000 < worm_mask.sum() < 28000

def _measure_worms_task(image_path, well):
    print('Measuring worm '+well)
    image = freeimage.read(image_path)
    well_mask = freeimage.read(image_path.parent / (well + '_well_mask.png')) > 0
    worm_mask = freeimage.read(image_path.parent / (well + '_worm_mask.png')) > 0
    return measure_fluorescence(image, worm_mask, well_mask)[0]

data_row = collections.namedtuple('data_row',
    ['area', 'integrated', 'median', 'percentile95',
     'expression_area', 'expression_area_fraction', 'expression_mean',
     'high_expression_area', 'high_expression_area_fraction',
     'high_expression_mean', 'high_expression_integrated'])

def measure_fluorescence(image, worm_mask, well_mask=None):
    if well_mask is not None:
        restricted_mask = ndimage.binary_erosion(well_mask, iterations=15)
        background = polyfit.fit_polynomial(image[::4,::4], mask=restricted_mask[::4,::4], degree=2).astype(numpy.float32)
        background = ndimage.zoom(background, 4)
        background /= background[well_mask].mean()
        background[background <= 0.01] = 1 # we're going to divide by background, so prevent div/0 errors
        image = image.astype(numpy.float32) / background
        image[~well_mask] = 0

    worm_pixels = image[worm_mask]
    low_px_mean, low_px_std = mcd.robust_mean_std(worm_pixels[worm_pixels < worm_pixels.mean()], 0.5)
    expression_thresh = low_px_mean + 2.5*low_px_std
    high_expression_thresh = low_px_mean + 6*low_px_std
    fluo_px = worm_pixels[worm_pixels > expression_thresh]
    high_fluo_px = worm_pixels[worm_pixels > high_expression_thresh]

    area = worm_mask.sum()
    integrated = worm_pixels.sum()
    median, percentile95 = numpy.percentile(worm_pixels, [50, 95])
    expression_area = fluo_px.size
    expression_area_fraction = expression_area / area
    expression_mean = fluo_px.mean()
    high_expression_area = high_fluo_px.size
    high_expression_area_fraction = high_expression_area / area
    high_expression_mean = high_fluo_px.mean()
    high_expression_integrated = high_fluo_px.sum()

    expression_mask = (image > expression_thresh) & worm_mask
    high_expression_mask = (image > high_expression_thresh) & worm_mask

    return data_row(area, integrated, median, percentile95,
     expression_area, expression_area_fraction, expression_mean,
     high_expression_area, high_expression_area_fraction,
     high_expression_mean, high_expression_integrated), (image, background, expression_mask, high_expression_mask)

def write_measures(data_rows, well_names, csv_out):
    fields = data_rows[0]._fields
    for data_row in data_rows:
        assert data_row._fields == fields
    data_header = ['well'] + list(fields)
    data = [data_header]
    for well, values in zip(well_names, data_rows):
        data.append(map(str, [well] + list(values)))
    outdata = '\n'.join(','.join(row) for row in data)
    with csv_out.open('w') as f:
        f.write(outdata)

