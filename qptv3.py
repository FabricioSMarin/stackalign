import sys
import PyQt5
from PyQt5.QtWidgets import QPushButton, QListWidget, QGraphicsView, QGraphicsScene, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QFrame, QApplication, QWidget, QMainWindow, QListWidgetItem
from PyQt5.QtCore import pyqtSignal
from PyQt5 import Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QTransform
import pyqtgraph as pg
import matplotlib.pyplot as plt
# import matplotlib.image as Image
import numpy as np
import math
import imageio

#TODO: add element dropdown
#TODO: add button to apply correction
#TODO: add 
#TODO: add function to calcaulate scale and angle offsets
#TODO: add function to scale colume 
#TODO: add function to rotate volume

class DragAndDropListWidget(QListWidget):
    filesAddedSig = pyqtSignal(name="fileAdded")

    def __init__(self):
        super().__init__()
        self.setMaximumHeight(200)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            for url in mime_data.urls():
                file_path = url.toLocalFile()
                self.addItem(file_path)
        self.filesAddedSig.emit()

    def clear_list_widget(self):
        self.clear()

class FileItemWidget(QWidget):
    def __init__(self, file_path):
        super().__init__()
        layout = PyQt5.QtWidgets.QHBoxLayout()
        self.path_name = QLabel(file_path)
        layout.addWidget(self.path_name)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setStyleSheet("background-color: black;")
        self.imgs_left = None
        self.imgs_right = None

        self.setWindowTitle("Drag and Drop Files")
        self.setGeometry(100, 100, 400, 800)
        self.stack_left = ImageView()
        self.sld_left = QSlider(Qt.Horizontal, self)
        self.sld_left.setObjectName("sld_left")
        self.sld_left.sliderReleased.connect(self.slider_changed)
        self.clear_left = QPushButton("clear left file list")
        self.clear_left.clicked.connect(self.clear_list)
        self.files_left = DragAndDropListWidget()
        self.files_left.setObjectName("left")
        self.files_left.filesAddedSig.connect(self.load_files)
        self.left = QVBoxLayout()
        self.left.addWidget(self.stack_left)
        self.left.addWidget(self.sld_left)
        self.left.addWidget(self.clear_left)
        self.left.addWidget(self.files_left)

        self.stack_right = ImageView()
        self.sld_right = QSlider(Qt.Horizontal, self)
        self.sld_right.setObjectName("sld_right")
        self.sld_right.sliderReleased.connect(self.slider_changed)
        self.clear_right = QPushButton("clear right file list")
        self.clear_right.clicked.connect(self.clear_list)
        self.files_right = DragAndDropListWidget()
        self.files_right.setObjectName("right")
        self.files_right.filesAddedSig.connect(self.load_files)
        self.right = QVBoxLayout()
        self.right.addWidget(self.stack_right)
        self.right.addWidget(self.sld_right)
        self.right.addWidget(self.clear_right)
        self.right.addWidget(self.files_right)

        # Create a rotated button
        self.apply = QPushButton("^^ Apply transformation ^^")
        self.apply.clicked.connect(self.apply_transform)
        scene = QGraphicsScene()
        gsw = scene.addWidget(self.apply)
        gsw.setPos(50,50)
        gsw.setRotation(90)
        gw = QGraphicsView()
        gw.setScene(scene)

        self.stack_combined = ImageView()
        self.sld_combined = QSlider(Qt.Horizontal, self)
        self.sld_combined.setObjectName("sld_combined")
        self.sld_combined.sliderReleased.connect(self.slider_changed)
        self.combined = QVBoxLayout()
        self.combined.addWidget(self.stack_combined)
        self.combined.addWidget(self.sld_combined)

        [self.combined.itemAt(i).widget().setVisible(False) for i in range(self.combined.count())]

        self.layout = QHBoxLayout()
        self.layout.addLayout(self.left)
        self.layout.addLayout(self.right)
        self.layout.addWidget(gw)
        self.layout.addLayout(self.combined)

        self.frame = QFrame()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

    def slider_changed(self):
        try: 
            sld = self.sender()
            if sld.objectName() == "sld_right":
                self.stack_right.image_view.setImage(self.imgs_right[sld.value()])
            else: 
                self.stack_left.image_view.setImage(self.imgs_left[sld.value()])
        except Exception as e:
            print(e)

    def clear_list(self):
        if "left" in self.sender().text():
            self.files_left.clear()
        else: 
            self.files_right.clear()

    def load_files(self):
        #get files list from sender. 
        listwidget = self.sender()
        files = [listwidget.item(x).text() for x in range(listwidget.count())]
        try: 
            img = imageio.v3.imread(files[0])
        except Exception as e: 
            print(e)
            return
        if len(files) ==1: 
            canvas = imageio.v3.imread(files[0])
        else: 
            shp = (len(files),*img.shape)
            canvas = np.zeros(shp)
            for i in range(len(files)):
                canvas[i] = imageio.v3.imread(files[i])

        if listwidget.objectName() == "right":
            self.imgs_right = canvas
            self.sld_right.setRange(0,canvas.shape[0]-1)
            self.stack_right.image_view.setImage(canvas[0])
            self.stack_right.reset_view()

        else: 
            self.imgs_left = canvas
            self.sld_left.setRange(0,canvas.shape[0]-1)
            self.stack_left.image_view.setImage(canvas[0])

    def apply_transform(self): 
        #TODO: if transforms valid and nothing else missing, reveal next windwo
        [self.combined.itemAt(i).widget().setVisible(True) for i in range(self.combined.count())]

                
class ImageView(pg.GraphicsLayoutWidget):
    mouseMoveSig = pyqtSignal(int,int, name= 'mouseMoveSig')
    mousePressSig =pyqtSignal(int,int,int, name= 'mousePressSig')
    def __init__(self):
        super(ImageView, self).__init__()
        self.setMinimumSize(300,600)
        self.initUI()

    def initUI(self):
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setBackground("k")
        self.arr = None
        self.spots_data = []
        self.spots_pos = [None, None, None]   
        self.last_moving_pos = None
        self.zoom_sf = 1
        self.image_view = pg.ImageItem()
        self.zoom_view = pg.ImageItem()

        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.roi = pg.CircleROI(pos=(20,20), size=40, pen=(255, 0, 0), handlePen=(0,255,0))
        self.roi.handleSize=1
        self.roi.sigRegionChanged.connect(self.updateRoi)
        self.vertical_line = pg.InfiniteLine(angle=90)
        self.vertical_line.setPen(pg.mkColor(255,0,255))
        self.vertical_line.setPos((20,20))

        self.horizontal_line = pg.InfiniteLine(angle=0, movable=False)
        self.horizontal_line.setPen(pg.mkColor(255,0,255))
        self.horizontal_line.addMarker(marker=">|<")
        self.horizontal_line.setPos((20,20))

        self.v1 = self.addViewBox()
        self.v1.addItem(self.zoom_view)
        self.v1.addItem(self.vertical_line)
        self.v1.addItem(self.horizontal_line)
        self.v1.invertY(True)
        self.v1.setAspectLocked(True)
        self.v1.setMenuEnabled(False)
        self.v1.setMouseEnabled(x=False, y=False)
        self.v1.setBorder(None)
        border_pen = pg.mkPen(color=(255, 255, 255), width=2)
        self.v1.setBorder(border_pen)

        self.v2 = self.addViewBox(1,0)
        self.v2.addItem(self.image_view)
        self.v2.addItem(self.scatter)
        self.v2.addItem(self.roi)
        self.v2.invertY(True)
        self.v2.setAspectLocked(True)
        self.v2.setMenuEnabled(False)
        self.v2.setCursor(Qt.BlankCursor) 
        self.v2.setMouseEnabled(x=False, y=False)
        self.v2.scene().sigMouseMoved.connect(self.mouseMoveEvent)
        self.v2.scene().sigMouseClicked.connect(self.mousePressEvent)
        self.v2.disableAutoRange()
        self.v2.setBorder(None)
        self.v2.setBorder(border_pen)


        # pos = np.random.normal(size=(2, 10), scale=100)
        # spots = [{'pos': pos[:, i], 'brush': pg.intColor(i * 10, 100)} for i in range(10)] + [{'pos': [0, 0], 'data': 1}]
        # self.scatter.addPoints(spots)
 
        # add item to plot window
        self.roi.removeHandle(0)

    def setZoomLimits(self, yrange, xrange):
        self.v2.setXRange(0, xrange, padding=0)
        self.v2.setYRange(0, yrange, padding=0)
        x = int(np.floor(xrange*0.025))
        y = int(np.floor(yrange*0.025))
        self.v1.setXRange(0, x, padding=0)
        self.v1.setYRange(0, y, padding=0)

    def updateRoi(self, roi):
        try:
            self.arr1 = roi.getArrayRegion(self.image_view.image, img=self.image_view)
            self.zoom_view.setImage(self.arr1)

        except: 
            pass
    def keyPressEvent(self, ev):
        if ev.key() == 45:
            self.zoom_sf=1.1
            self.zoom(self.zoom_sf)
        elif ev.key() == 61:
            self.zoom_sf=0.9
            self.zoom(self.zoom_sf)
        elif ev.key() == 48: #reset view
            self.reset_view()
        else:
            super().keyPressEvent(ev)

    def reset_view(self):
        self.zoom(1)
        self.setZoomLimits(self.image_view.height(), self.image_view.width())
        print(self.image_view.height(), self.image_view.height())
        self.image_view.setPos(0,0)

    def zoom(self, factor):
        # self.image_view.scaleBy((factor, factor), center=(self.moving_pos.x(), self.moving_pos.y()))
        try:
            self.v2.scaleBy((factor, factor), center=(self.moving_pos.x(), self.moving_pos.y()))
        except: 
            self.v2.scaleBy((factor, factor), center=(0, 0))

        # self.v2.setScale(factor)

    def wheelEvent(self,ev):
        print(ev.angleDelta().y())
        if ev.angleDelta().y()<0:
            # self.zoom_sf-=0.02
            self.zoom_sf=1.03
            self.zoom(self.zoom_sf)
        elif ev.angleDelta().y()>0:
            # self.zoom_sf+=0.02
            self.zoom_sf=0.97
            self.zoom(self.zoom_sf)
        self.moving_pos = self.v2.mapSceneToView(ev.pos())
        print(self.v2.pos())

    def mouseMoveEvent(self, ev):
        self.v2.setCursor(Qt.BlankCursor) 
        self.moving_pos = self.v2.mapSceneToView(ev.pos())
        self.mouseMoveSig.emit(self.moving_pos.x(), self.moving_pos.y())
        self.roi.setPos([self.moving_pos.x(), self.moving_pos.y()], finish=False)
        diff = self.moving_pos - self.start_pos
        if self.last_moving_pos is None:
            self.last_moving_pos = self.start_pos
        inc = self.moving_pos - self.last_moving_pos
        
        if ev.buttons() == Qt.LeftButton and self.start_pos !=  self.moving_pos:
            if self.image_view.width() is None: 
                return
            self.image_view.setPos(diff.x() + self.img_pos.x(), diff.y() + self.img_pos.y())
            self.scatter.moveBy(inc.x(), inc.y())
            # self.scatter.mapRectToView(self.image_view.viewRect())

        self.last_moving_pos = self.moving_pos

    def mousePressEvent(self, ev):
        self.start_pos = self.v2.mapSceneToView(ev.pos())
        self.img_pos = self.image_view.pos()
        p = self.v2.viewRange()

        # frame_height = self.image_view.height()
        # frame_width = self.image_view.width()

        if ev.button() == 1:
            pos = (self.start_pos.x()+20, self.start_pos.y()+20)
            if self.spots_pos[0] is None:
                self.spots_pos[0] = np.array(pos)
                self.spots_data.append({'pos': pos, 'brush': 'red', 'pen': 'red'})
            elif self.spots_pos[1] is None:
                self.spots_pos[1] = np.array(pos)
                self.spots_data.append({'pos': pos, 'brush': 'green', 'pen': 'green'})
            elif self.spots_pos[2] is None:
                self.spots_pos[2] = np.array(pos)
                self.spots_data.append({'pos': pos, 'brush': 'blue', 'pen': 'blue'})
            else: 
                return
            self.scatter.setData(self.spots_data)
            print(self.spots_pos)

        if ev.button() == 2:
            pos = (self.start_pos.x()+20, self.start_pos.y()+20)
            print(pos)
            print(self.spots_pos)
            diffs = []
            for i in self.spots_pos:
                if i is None:
                    diffs.append(np.inf)
                else:
                    diffs.append(math.dist(pos, i))
    
            if np.array(diffs).any() is not None: 
                if any(np.array(diffs)<100):
                    minpos = diffs.index(min(diffs))
                    self.spots_pos[minpos] = None
                else:
                    return
                self.spots_data = []
                if self.spots_pos[0] is not None:
                    self.spots_data.append({'pos': self.spots_pos[0], 'brush': 'red', 'pen': 'red'})
                if self.spots_pos[1] is not None:
                    self.spots_data.append({'pos': self.spots_pos[1], 'brush': 'green', 'pen': 'green'})
                if self.spots_pos[2] is not None:
                    self.spots_data.append({'pos': self.spots_pos[2], 'brush': 'blue', 'pen': 'blue'})
                self.scatter.clear()
                self.scatter.setData(self.spots_data)
            else:
                return
            
    def mouseReleaseEvent(self, ev):
        # Clear the starting position when the mouse button is released
        self.end_pos = self.v2.mapSceneToView(ev.pos())
        print("end pos:", self.end_pos.x(), self.end_pos.y())
        if self.start_pos ==  self.end_pos:
            print("mouse clicked")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
