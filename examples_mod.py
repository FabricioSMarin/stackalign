import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

def updateRoi(roi):
    global im1, im2, arr
    arr1 = roi.getArrayRegion(im1.image, img=im1)
    im2.setImage(arr1)

pg.setConfigOptions(imageAxisOrder='row-major')
## create GUI
app = pg.mkQApp("ROI Types Examples")
w = pg.GraphicsLayoutWidget(show=True, size=(800,800), border=True)

## Create image to display
arr = np.ones((100, 100), dtype=float)
arr[45:55, 45:55] = 0
arr[25, :] = 5
arr[:, 25] = 5

## Create image items, add to scene and set position 
roi = pg.CircleROI([0, 0], [20, 20], pen=(4,9))
roi.sigRegionChanged.connect(updateRoi)

im1 = pg.ImageItem(arr)
v = w.addViewBox()
v.addItem(im1)
v.addItem(roi)
v.invertY(True)  ## Images usually have their Y-axis pointing downward
v.setAspectLocked(True)

im2 = pg.ImageItem()
v2 = w.addViewBox(1,0)
v2.addItem(im2)
v2.invertY(True)
v2.setAspectLocked(True)

if __name__ == '__main__':
    pg.exec()