import numpy
import matplotlib.pyplot as plt
import pathlib
import pickle
import collections
import math

import PyQt5.Qt as Qt

import freeimage

class Animator:
    def __init__(self, ris_widget):
        self.ris_widget = ris_widget
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self._advance)

    def start(self, images, fps=5):
        self.timer.stop()
        self.images = images
        self.i = 0
        self.timer.setInterval(int(1000 * 1/fps))
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def _advance(self):
        self.ris_widget.image = self.images[self.i]
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

class DeathDayEvaluator:
    def __init__(self, ris_widget, out_dir, dates, ages, last_alive_dates, well_names):
        self.animator = Animator(ris_widget)
        self.ris_widget = ris_widget
        self.date_to_index = {d: i for i, d in enumerate(dates)}
        self.date_to_index[None] = None # if there is no last-alive date, deal properly with that
        self.max_alive_index = len(dates)-1
        self.max_well_index = len(well_names)-1
        self.ages = ages
        self.last_alive_indices = [self.date_to_index[ld] for ld in last_alive_dates]
        self.out_dir = pathlib.Path(out_dir)
        self.well_names = well_names
        self.rect = Qt.QGraphicsRectItem()
        pen = Qt.QPen(Qt.QColor(255,255,0,255))
        pen.setCosmetic(True)
        pen.setWidth(4)
        pen.setJoinStyle(Qt.Qt.MiterJoin)
        self.rect.setPen(pen)
        self.rect.hide()
        self.ris_widget.main_scene.addItem(self.rect)
        self.set_well(0)
        self.actions = []
        self._add_action('left', Qt.Qt.Key_Left, lambda: self.update_last_alive(-1))
        self._add_action('right', Qt.Qt.Key_Right, lambda: self.update_last_alive(1))
        self._add_action('up', Qt.Qt.Key_Up, lambda: self.update_well(1))
        self._add_action('down', Qt.Qt.Key_Down, lambda: self.update_well(-1))

    def get_lifespans(self):
        lifespans = []
        for lai in self.last_alive_indices:
            if lai is None:
                lifespan = -1 # worm was never alive
            elif lai < self.max_alive_index: # worm has died
                lifespan = (self.ages[lai] + self.ages[lai+1]) / 2 # assume death was between last live observation and first dead observation
            else:
                lifespan = numpy.nan # worm is still alive
            lifespans.append(lifespan)
        return numpy.array(lifespans)

    def save_lifespans(self):
        lifespans = self.get_lifespans()
        lifespans_out = [(wn, str(ls)) for wn, ls in zip(self.well_names, lifespans)]
        with (self.out_dir/'evaluated_lifespans.csv').open('w') as f:
            f.write('\n'.join(','.join(row) for row in lifespans_out))
        with (self.out_dir/'evaluated_lifespans.pickle').open('wb') as f:
            pickle.dump(lifespans, f)

    def _add_action(self, name, key, function):
        action = Qt.QAction(name, self.ris_widget)
        action.setShortcut(key)
        self.ris_widget.addAction(action)
        action.triggered.connect(function)
        self.actions.append(action)

    def stop(self):
        self.animator.stop()
        self.ris_widget.image_scene.removeItem(self.rect)
        for action in self.actions:
            self.ris_widget.removeAction(action)

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
        print(self.well)

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
            self.ris_widget.image_view.centerOn(x, self.rect_height/2)

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
        self.set_last_alive(new)

    def save(self):
        with (self.out_dir / 'evaluations.pickle').open('wb') as f:
            pickle.dump((self.last_alive_indices, self.well_index), f)

    def load(self):
        with (self.out_dir / 'evaluations.pickle').open('rb') as f:
            self.last_alive_indices, self.well_index = pickle.load(f)
            self.set_well(self.well_index)