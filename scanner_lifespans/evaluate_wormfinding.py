import pathlib
import collections
import numpy
from scipy import ndimage

from concurrent import futures

from PyQt5 import Qt
import freeimage
from ris_widget import layer_stack
from ris_widget import image

from .util import split_image_name

def validate(image_dir, rw):
    image_dir = pathlib.Path(image_dir)
    well_images = collections.defaultdict(dict)
    for image_path in image_dir.glob('*'):
        if image_path.suffix not in ('.tif', '.tiff', '.png'):
            continue
        well, rest = split_image_name(image_path)
        well_images[well][rest] = image_path
    has_af = ['autofluorescence' in v for v in well_images.values()]
    if not all(has_af[0] == h for h in has_af):
        raise ValueError('some wells have autofluorescence images, but not all.')
    has_af = has_af[0]
    image_order = ['brightfield', 'well_mask', 'worm_mask', 'fluorescence']
    if has_af:
        image_order.append('autofluorescence')

    image_paths = []
    flipbook_names = []
    image_names = []
    for well, image_types in sorted(well_images.items()):
        image_types.pop('worm_mask_orig', None)
        images = [image_types.pop(name) for name in image_order]
        assert len(image_types) == 0
        image_paths.append(images)
        flipbook_names.append(well)
        image_names.append(image_order)
    rw.layers = layer_stack.LayerList.from_json(_layer_props)
    va = ValidAnnotator(rw, image_dir)
    rw.flipbook.add_image_files(image_paths, flipbook_names, image_names)
    va.show()
    return va

class ValidAnnotator(Qt.QWidget):
    def __init__(self, rw, image_dir, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)
        self.rw = rw
        self.image_dir = pathlib.Path(image_dir)
        layout = Qt.QFormLayout()
        self.setLayout(layout)
        self.valid = Qt.QCheckBox()
        layout.addRow("Valid", self.valid)
        self.edit = Qt.QPushButton("Edit Mask")
        self.editing = False
        layout.addRow(self.edit)
        self.rw.flipbook.layout().addWidget(self)

        self.current_page = None
        self.old_well = None

        valid_well_f = self.image_dir / 'valid_wells.txt'
        with valid_well_f.open() as f:
            self.valid_wells = {line.strip() for line in f}

        self.valid.stateChanged.connect(self._on_valid_state_changed)
        self.rw.flipbook.pages_view.selectionModel().currentRowChanged.connect(self._on_page_change)
        self.edit.clicked.connect(self._on_edit_clicked)
        self._on_page_change()

    def disconnect(self):
        self.valid.stateChanged.disconnect(self._on_valid_state_changed)
        self.rw.flipbook.pages_view.selectionModel().currentRowChanged.disconnect(self._on_page_change)
        self.edit.clicked.disconnect(self._on_edit_clicked)

    def _on_page_change(self):
        if self.current_page is not None:
            self.current_page.inserted.disconnect(self._on_page_change)
            if self.old_well is not None:
                # if we were previously looking at a non-empty list of images
                if self.editing:
                    self.stop_editing()
                else:
                    self.current_page[2].set(data=self.worm_mask)
        self.current_page = self.rw.flipbook.focused_page
        if self.current_page is None:
            return
        self.current_page.inserted.connect(self._on_page_change)
        if len(self.current_page) == 0:
            self.old_well = None
            return
        self.current_page[1].set(data=self.current_page[1].data.astype(bool))
        self.current_page[2].set(data=self.current_page[2].data.astype(bool))
        self.well = self.current_page.name
        valid = self.well in self.valid_wells
        self.valid.setChecked(valid)
        # below will write twice if the above changes the state. Who cares?
        self._on_valid_state_changed(valid)
        if self.old_well != self.well:
            self.outline_mask()
        self.old_well = self.well

    def _on_valid_state_changed(self, valid):
        if valid:
            self.rw.layers[0].tint = (1.0, 1.0, 1.0, 1.0)
            self.valid_wells.add(self.well)
        else:
            self.rw.layers[0].tint = (1.0, 0.9, 0, 1.0)
            self.valid_wells.discard(self.well)
        with (self.image_dir / 'valid_wells.txt').open('w') as f:
            f.write('\n'.join(sorted(self.valid_wells)))

    def _on_edit_clicked(self):
        if self.editing:
            self.stop_editing()
        else:
            self.start_editing()

    def stop_editing(self):
        self.editing = False
        self.edit.setText('Edit Mask')
        self.rw.qt_object.layer_stack_painter_dock_widget.hide()
        self.rw.layers[2].opacity = 1
        self.rw.layers[3].visible = True
        self.rw.layers[4].visible = True
        self.outline_mask()
        orig_mask = self.image_dir / '{}_worm_mask_orig.png'.format(self.well)
        worm_mask = self.image_dir / '{}_worm_mask.png'.format(self.well)
        if not orig_mask.exists():
            worm_mask.rename(orig_mask)
        freeimage.write(self.worm_mask.astype(numpy.uint8)*255, worm_mask)

    def outline_mask(self):
        # ONLY SAFE if current_page[2] known to contain mask, not outline
        self.worm_mask = self.current_page[2].data
        outline = self.worm_mask ^ ndimage.binary_erosion(self.worm_mask, iterations=3)
        self.current_page[2].set(data=outline)

    def start_editing(self):
        self.editing = True
        sm = self.rw.qt_object.layer_stack._selection_model
        m = sm.model()
        sm.setCurrentIndex(m.index(2, 0), Qt.QItemSelectionModel.SelectCurrent | Qt.QItemSelectionModel.Rows)
        self.edit.setText('Save Edits')
        self.rw.layers[2].opacity = 0.5
        self.rw.layers[3].visible = False
        self.rw.layers[4].visible = False
        self.current_page[2].set(self.worm_mask)
        self.rw.qt_object.layer_stack_painter_dock_widget.show()
        self.rw.qt_object.layer_stack_painter.brush_size_lse.value = 13

_layer_props = """{
     "layer property stack": [
      {
       "auto_min_max_enabled": true,
       "name": "brightfield"
      },
      {
       "tint": [
        1.0,
        1.0,
        0.0,
        0.5
       ],
       "auto_min_max_enabled": true,
       "visible": false,
       "name": "well_mask"
      },
      {
       "tint": [
        0.0,
        1.0,
        1.0,
        1.0
       ],
       "auto_min_max_enabled": true,
       "name": "worm_mask"
      },
      {
       "tint": [
        0.0,
        1.0,
        0.0,
        1.0
       ],
       "auto_min_max_enabled": true,
       "name": "GFP"
      },
      {
       "tint": [
        1.0,
        0.0,
        0.0,
        1.0
       ],
       "auto_min_max_enabled": true,
       "name": "autofluorescence"
      }
     ]
    }"""