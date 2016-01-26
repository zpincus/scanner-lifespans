import numpy
import pathlib
import pickle
import collections
import PyQt5.Qt as Qt
import freeimage

from zplib import util
from  ris_widget import ris_widget

from . import estimate_lifespans

class DeathDayEvaluator:
    def __init__(self, out_dir, ages, last_alive_indices, well_names):
        self.rw = ris_widget.RisWidget()
        self.rw.qt_object.layer_table_dock_widget.hide()
        self.rw.qt_object.flipbook_dock_widget.hide()
        self.rw.qt_object.histogram_dock_widget.hide()
        self.animator = Animator(self.rw)
        self.max_alive_index = len(ages)-1
        self.max_well_index = len(well_names)-1
        self.ages = ages
        self.last_alive_indices = last_alive_indices
        self.out_dir = pathlib.Path(out_dir)
        self.well_names = well_names
        self.rect = Qt.QGraphicsRectItem()
        pen = Qt.QPen(Qt.QColor(255,255,0,255))
        pen.setCosmetic(True)
        pen.setWidth(4)
        pen.setJoinStyle(Qt.Qt.MiterJoin)
        self.rect.setPen(pen)
        self.rect.hide()
        self.rw.main_scene.addItem(self.rect)
        self.set_well(0)
        self.actions = []
        self._add_action('left', Qt.Qt.Key_Left, lambda: self.update_last_alive(-1))
        self._add_action('right', Qt.Qt.Key_Right, lambda: self.update_last_alive(1))
        self._add_action('up', Qt.Qt.Key_Up, lambda: self.update_well(1))
        self._add_action('down', Qt.Qt.Key_Down, lambda: self.update_well(-1))
        self.rw.show()

    def save_lifespans(self):
        util.dump(self.out_dir / 'evaluations.pickle',
            last_alive_indices=self.last_alive_indices,
            well_index=self.well_index)
        lifespans = estimate_lifespans.last_alive_indices_to_lifespans(self.last_alive_indices, self.ages)
        lifespans_out = [('well name', 'lifespan')] + [(wn, str(ls)) for wn, ls in zip(self.well_names, lifespans)]
        util.dump_csv(lifespans_out, self.out_dir/'evaluated_lifespans.csv')

    def save(self):
        self.save_lifespans()

    def load(self):
        data = util.load(self.out_dir / 'evaluations.pickle')
        self.last_alive_indices = data.last_alive_indices
        self.well_index = data.well_index
        self.set_well(self.well_index)

    ## Helper functions

    def _add_action(self, name, key, function):
        action = Qt.QAction(name, self.rw.qt_object)
        action.setShortcut(key)
        self.rw.qt_object.addAction(action)
        action.triggered.connect(function)
        self.actions.append(action)

    def stop(self):
        self.animator.stop()
        self.rw.main_scene.removeItem(self.rect)
        for action in self.actions:
            self.rw.qt_object.removeAction(action)

    def set_well(self, index):
        self.well_index = index
        images, shape = assemble_image_sequence(self.out_dir, self.well_names[index])
        self.rect.hide()
        self.rect.setRect(0,0,*shape)
        self.animator.start(images)
        self.rect_offset = shape[0]
        self.rect_height = shape[1]
        self.set_last_alive(self.last_alive_indices[self.well_index], zoom_to=True)
        self.well = self.well_names[index]
        self.rw.qt_object.setWindowTitle('Well {} ({}/{})'.format(self.well, index+1, len(self.well_names)))

    def update_well(self, offset):
        new = self.well_index + offset
        self.set_well(min(max(0, new), self.max_well_index))

    def set_last_alive(self, index, zoom_to=False):
        self.last_alive_indices[self.well_index] = index
        self.rect.hide()
        if index is not None:
            self.rect.setX(self.rect_offset * index)
            self.rect.show()
        if zoom_to:
            x = 0 if index is None else self.rect_offset * (index + 0.5)
            self.rw.main_view.centerOn(x, self.rect_height/2)

    def update_last_alive(self, offset):
        current = self.last_alive_indices[self.well_index]
        if current == None:
            if offset <= 0:
                new = None
            else:
                new = min(offset-1, self.max_alive_index)
        else:
            new = min(current+offset, self.max_alive_index)
            if new < 0:
                new = None
        self.set_last_alive(new, zoom_to=True)


class DOAEvaluator:
    status_codes = ['One worm', 'No worms', 'Many worms', 'DOA']
    def __init__(self, out_dir, date_for_images, well_names, statuses=None):
        self.rw = ris_widget.RisWidget()
        self.rw.qt_object.layer_table_dock_widget.hide()
        self.rw.qt_object.flipbook_dock_widget.hide()
        self.rw.qt_object.histogram_dock_widget.hide()
        self.animator = Animator(self.rw)
        self.out_dir = pathlib.Path(out_dir)
        self.well_names = well_names
        self.max_well_index = len(well_names)-1
        if statuses is None:
            self.statuses = [0] * len(well_names)
        else:
            assert(len(statuses) == len(well_names))
            self.statuses = statuses
        self.date_dir = self.out_dir / date_for_images
        self.set_well(0)
        self.actions = []
        self._add_action('left', Qt.Qt.Key_Left, lambda: self.update_status(-1))
        self._add_action('right', Qt.Qt.Key_Right, lambda: self.update_status(1))
        self._add_action('up', Qt.Qt.Key_Up, lambda: self.update_well(1))
        self._add_action('down', Qt.Qt.Key_Down, lambda: self.update_well(-1))
        self.rw.show()

    def save_status(self):
        util.dump(self.out_dir / 'statuses.pickle',
            statuses=self.statuses,
            well_index=self.well_index)
        status_out = [('well name', 'status')] + [(wn, self.status_codes[i]) for wn, i in zip(self.well_names, self.statuses)]
        util.dump_csv(status_out, self.out_dir/'evaluated_statuses.csv')

    def save(self):
        self.save_status()

    def load(self):
        data = util.load(self.out_dir / 'statuses.pickle')
        self.statuses = data.statuses
        self.well_index = data.well_index
        self.set_well(self.well_index)

    ## Helper functions

    def _add_action(self, name, key, function):
        action = Qt.QAction(name, self.rw.qt_object)
        action.setShortcut(key)
        self.rw.qt_object.addAction(action)
        action.triggered.connect(function)
        self.actions.append(action)

    def stop(self):
        self.animator.stop()
        for action in self.actions:
            self.rw.qt_object.removeAction(action)

    def set_well(self, index):
        self.well_index = index
        well_name = self.well_names[index]
        images = [freeimage.read(str(image)) for image in sorted(self.date_dir.glob('well_images/{}-*.png'.format(well_name)))]
        self.animator.start(images)
        self.well = self.well_names[index]
        self.set_status(self.statuses[self.well_index])

    def update_well(self, offset):
        new = self.well_index + offset
        self.set_well(min(max(0, new), self.max_well_index))

    def set_status(self, status_index):
        self.statuses[self.well_index] = status_index
        self.rw.qt_object.setWindowTitle('Well {}: {} ({}/{})'.format(self.well, self.status_codes[status_index], self.well_index+1, len(self.well_names)))

    def update_status(self, offset):
        current = self.statuses[self.well_index]
        self.set_status((current + offset)%len(self.status_codes))


class Animator:
    def __init__(self, ris_widget):
        self.rw = ris_widget
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self._advance)

    def start(self, images, fps=5):
        self.timer.stop()
        self.images = images
        assert len(images) > 0
        self.i = 0
        self.timer.setInterval(int(1000 * 1/fps))
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def _advance(self):
        self.rw.image = self.images[self.i]
        self.i += 1
        self.i %= len(self.images)

def assemble_image_sequence(out_dir, well_name, crop=True):
    out_dir = pathlib.Path(out_dir)
    well_images = collections.defaultdict(list)
    for img in sorted(out_dir.glob('*/well_images/{}-*.png'.format(well_name))):
        date = img.parent.parent.name
        well_images[date].append(freeimage.read(str(img)))
    dates, images = zip(*sorted(well_images.items()))
    maxlen = max(len(ims) for ims in images)
    shapes = numpy.array([ims[0].shape for ims in images])
    if crop:
        shape = shapes.min(axis=0)
    else:
        shape = shapes.max(axis=0)
    new_images = []
    for ims in images:
        oldshape = ims[0].shape
        extra = numpy.abs(shape - oldshape)/2
        if numpy.any(extra):
            extra = list(zip(numpy.ceil(extra).astype(int), extra.astype(int)))
            if crop:
                (xl, xh), (yl, yh) = extra
                xh = -xh if xh else None
                yh = -yh if yh else None
                out = [i[xl:xh, yl:yh] for i in ims]
            else:
                out = [numpy.pad(i, extra, 'constant') for i in ims]
        else:
            out = ims
        extra_ims = maxlen - len(ims)
        if extra_ims:
            out += out[-1:]*extra_ims
        new_images.append(out)
    images_out = [numpy.concatenate([ims[i] for ims in new_images]) for i in range(maxlen)]
    return images_out, shape
