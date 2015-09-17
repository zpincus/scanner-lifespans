import numpy
import scipy.ndimage as ndimage
import freeimage
import pathlib

from zplib.image import mask

def get_mask(image, low_pct, high_pct, max_hole_radius):
    low_thresh, high_thresh = numpy.percentile(image, [low_pct, high_pct])
    image_mask = mask.hysteresis_threshold(image, low_thresh, high_thresh)
    image_mask = mask.fill_small_radius_holes(image_mask, max_hole_radius)
    image_mask = mask.get_largest_object(image_mask)
    return image_mask

def measure_intensity(image):
    image_mask = get_mask(image, 99.2, 99.99, 12)
    background = mask.get_background(image_mask, 15, 20)
    background_value = numpy.percentile(image[background], 25)
    pixel_data = image[image_mask] - background_value
    return (image_mask.sum(), pixel_data.sum())+ tuple(numpy.percentile(pixel_data, [50, 95]))

def measure_intensities(image_files):
    all_values = []
    for image_file in image_files:
        print(str(image_file))
        image = freeimage.read(image_file)
        all_values.append(measure_intensity(image))
    return all_values

def get_well_names(image_files):
    wells = []
    for image_file in sorted(image_files):
        row = image_file.name[0] # first letter is row, then ' - ', then two-digit col
        col = image_file.name[4:6]
        well = row + col
        wells.append(well)
    return wells

def write_intensities(all_values, wells, csv_out):
    data_header = ['well', 'area', 'integrated', 'median', '95th']
    data = [data_header]
    for well, values in zip(wells, all_values):
        data.append(map(str, [well] + list(values)))
    outdata = '\n'.join(','.join(row) for row in data)
    with open(csv_out, 'w') as f:
        f.write(outdata)
    return wells

def measure_incyte_images(image_dir, image_glob='*FITC*'):
    image_dir = pathlib.Path(image_dir)
    assert image_dir.exists()
    image_files = list(image_dir.glob(image_glob))
    wells = get_well_names(image_files)
    all_values = measure_intensities(image_files)
    write_intensities(all_values, wells, image_dir.with_suffix('.csv'))