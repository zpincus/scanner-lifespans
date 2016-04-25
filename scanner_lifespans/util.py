import re

ROW_NAMES_384 = 'ABCDEFGHIJKLMNOP'
COL_NAMES_384 = ['{:02d}'.format(i) for i in range(1, 25)]

_well_regex = re.compile('[A-P][0-9][0-9]?')
def split_image_name(image_path):
    image_path = pathlib.Path(image_path)
    well_match = _well_regex.match(image_path.stem)
    well = well_match.group()
    rest = image_path.stem[well_match.end():]
    if rest.startswith('_'):
        rest = rest[1:]
    return well, rest