import sys
import PyQt5
from PyQt5.QtWidgets import QPushButton, QListWidget, QComboBox, QGraphicsView, QGraphicsScene, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QFrame, QApplication, QWidget, QMainWindow, QListWidgetItem
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
from scipy import interpolate, ndimage
from skimage.transform import resize
from scipy.ndimage import zoom, map_coordinates



#TODO: add element dropdown
#TODO: add function to calcaulate scale and angle offsets
#TODO: add function to scale colume 
#TODO: add function to rotate volume
#TODO: add function to translate volume using the center point of combined spots 
#TODO: add "render" button to display high resolution combined stack. 
#TODO: downscale high resolution stack for displaying purposes. 
#TODO: add export button

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

        self.setWindowTitle("Drag and Drop Files")
        self.setGeometry(100, 100, 400, 500)
        self.setMinimumHeight(500)

        self.stack_left = customWidget()
        self.stack_right = customWidget()

        # Create a rotated button
        self.apply = QPushButton("^^ Apply transformation ^^")
        self.apply.clicked.connect(self.apply_transform)
        self.apply.setFixedHeight(30)
        self.apply.setCheckable(True)
        scene = QGraphicsScene()
        gsw = scene.addWidget(self.apply)
        gsw.setPos(50,50)
        gsw.setRotation(90)
        gw = QGraphicsView()
        gw.setFixedWidth(40)
        gw.setScene(scene)

        self.combined = customWidget()
        self.combined.setVisible(False)
        #TODO: disable hotspots 
        #TODO: disble clear files list 
        #TODO: ENABLE save new stack as tiff
        #TODO: disable file drag/drop field

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.stack_left)
        self.layout.addWidget(self.stack_right)
        self.layout.addWidget(gw)
        self.layout.addWidget(self.combined)

        self.frame = QFrame()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

    def apply_transform(self): 
        if self.sender().isChecked():
            self.combined.setVisible(True)

        
            if any(elem is None for elem in self.stack_left.markers) or any(elem is None for elem in self.stack_right.markers):
                return
            else: 
                # #temporary: downscale larger volume (speed)
                # downscaled = self.downscale_larger(np.array(self.stack_left.markers), self.stack_left.imgs, np.array(self.stack_right.markers), self.stack_right.imgs)
                new_shape = tuple(np.round(np.array(self.stack_right.imgs.shape)*0.08).astype(int))
                downscaled = resize(self.stack_right.imgs, new_shape, mode='constant', anti_aliasing=True)
                
                                # get normal vectors 
                v1, v2 = self.get_vectors_from_planes(np.array(self.stack_left.markers), np.array(self.stack_right.markers))
                #get rotation matrix
                R = self.align_vectors(v1,v2)
                #apply rotation matrix
                corrected_vol = self.apply_R2img(downscaled, R)
                self.combined.imgs = corrected_vol
                self.combined.imgs_dict["combined"] = corrected_vol
                self.combined.imgs_dict["2ide"]= self.stack_left.imgs 
                self.combined.elements.currentIndexChanged.disconnect()
                self.combined.elements.clear()
                items = list(self.combined.imgs_dict.keys())
                self.combined.elements.addItems(items)
                self.combined.elements.currentIndexChanged.connect(self.combined.dropdown_changed)
                self.combined.stack.image_view.setImage(self.combined.imgs_dict["combined"][0])
                # self.combined.sld.valueChanged.disconnect()
                self.combined.sld.setRange(0,self.combined.imgs_dict["combined"].shape[0]-1)   
                self.combined.sld.setValue(0)
                # self.combined.sld.valueChanged.connect(self.combined.slider_changed)
                self.combined.stack.reset_view()
                #db =  np.sqrt((1194.7-1589.6)**2 + (129-617)**2) = 
                # ds = np.sqrt((93.25460122699387-146.34509202453987)**2 + (157.19401840490798-171.04371165644173)**2) = 54
                scale = 54.7/627.7



        else: 
            self.combined.setVisible(False)

    def get_vectors_from_planes(self, ptsA, ptsB):
        # u = np.array([u1,u2,u3])
        # v = np.array([v1,v2,v3])

        #find normal vector of plane, 
        #find angle of normal vector with respect to each cartesian vector (1,0,0) (0,1,0), (0,0,1)
        # to find the angle with respect to each plane. 

        #normal vector of first plane
        va1 = ptsA[1] - ptsA[0]
        va2 = ptsA[2] - ptsA[0]
        nv1 = np.cross(va1,va2)
        v1 = nv1/ np.linalg.norm(nv1)

        #normal vector of second plane
        vb1 = ptsB[1] - ptsB[0]
        vb2 = ptsB[2] - ptsB[0]
        nv2 = np.cross(vb1,vb2)
        v2 = nv2/ np.linalg.norm(nv2)

        # #dot product of noprmal vector 2 with respect to each component of normal vector 1
        # dx = np.dot(nv2_normalized[0],nv1_normalized[0])
        # dy = np.dot(nv2_normalized[1],nv1_normalized[1])
        # dz = np.dot(nv2_normalized[2],nv1_normalized[2])

        # #angles between plane A and plane B with respect to each component from plane A
        # ax = np.degrees(np.arccos(dx))
        # ay = np.degrees(np.arccos(dy))
        # az = np.degrees(np.arccos(dz))
                    
        return v1, v2

    # def calculate_angles2(self,u,v):

    def apply_R2img(self, vol, R):
        depth,height,width = vol.shape
        coords = np.indices((depth,height,width)).reshape(3,-1)
        coords_centered = coords - np.array([[depth//2],[height//2],[width//2]])
        rotated_coords = np.dot(R, coords_centered)
        rotated_coords += np.array([[depth//2], [height//2], [width//2]])
        rotated_coords = rotated_coords.reshape(3,depth,height,width)
        rotated_vol = np.zeros_like(vol)
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    coords = rotated_coords[:,z,y,x]
                    if np.all((coords>=0) & (coords< [depth,height,width])):
                        rotated_vol[z,y,x] = map_coordinates(vol, coords[:,None], order=1, mode = "nearest")
        return rotated_vol

        
    def align_vectors(self, v1,v2):
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        axis = np.cross(v1,v2)
        theta = np.arccos(np.dot(v1,v2))
        R = self.rot_mat(axis,theta)
        return R

    def rot_mat(self, axis,theta):
        axis =axis/np.linalg.norm(axis)
        a = np.cos(theta/2)
        b,c,d = -axis*np.sin(theta/2)
        aa,bb,cc,dd = a*a,b*b,c*c,d*d
        bc,ad,ac,ab,bd,cd = b*c,a*d,a*c,a*b,b*d,c*d
        R = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                      [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                      [2*(bd+ac), 2*(cd-ab),aa+dd-bb-cc]])
        return R

    def rebin_3d_array(self, arr, scale_factor):
        # Calculate the new shape after rebinning
        new_shape = tuple(int(old_dim * factor) for old_dim, factor in zip(arr.shape, scale_factor))
        
        # Create an array to hold the rebinned data
        rebinned_arr = np.zeros(new_shape)
        
        # Calculate the size of each bin in the rebinned array
        bin_size = tuple(int(old_dim / new_dim) for old_dim, new_dim in zip(arr.shape, new_shape))
        
        # Iterate over each element in the rebinned array
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                for k in range(new_shape[2]):
                    # Calculate the range of indices in the original array corresponding to this bin
                    start_i = int(i * bin_size[0])
                    end_i = int((i + 1) * bin_size[0])
                    start_j = int(j * bin_size[1])
                    end_j = int((j + 1) * bin_size[1])
                    start_k = int(k * bin_size[2])
                    end_k = int((k + 1) * bin_size[2])
                    
                    # Take the average of the elements in this bin
                    rebinned_arr[i, j, k] = np.mean(arr[start_i:end_i, start_j:end_j, start_k:end_k])
        
        return rebinned_arr

    
    def rotate_volume(self, volume, angles):
        # angles = [x_deg,y_deg,z_deg]
        if angles[0] != 0:
            axes = (0, 1)  # z,y
            volume = ndimage.rotate(volume, angles[0], axes=axes)

        if angles[1] != 0:
            axes = (1, 2)  # y,x
            volume = ndimage.rotate(volume, angles[1], axes=axes)

        if angles[2] != 0:
            axes = (0, 2)  # x,z
            volume = ndimage.rotate(volume, angles[2], axes=axes)
        return volume
    
    def upscale_smaller(self, markers1, stack_1, markers2, stack_2):
        # resized_array = resize(vol, (x,y,z), mode='constant', anti_aliasing=True)
        # volume = zoom(volume, (0.5, 0.5, 2))
        # volume = resize(volume, (x,y,z))
        pass
        # return volume
    
    def downscale_larger(self, markers1, stack_1, markers2, stack_2):
        z_levels = markers1[:,0]
        out = [np.where(z_levels == element)[0].tolist() for element in np.unique(z_levels)]
        idx_pair = out[[len(i) for i in out].index(max([len(i) for i in out]))]
        point_pair = np.array(markers1)[:,1:][idx_pair]
        dist1 = np.linalg.norm(point_pair[0]- point_pair[1])

        z_levels = np.array(markers2)[:,0]
        out = [np.where(z_levels == element)[0].tolist() for element in np.unique(z_levels)]
        idx_pair = out[[len(i) for i in out].index(max([len(i) for i in out]))]
        point_pair = np.array(markers2)[:,1:][idx_pair]
        dist2 = np.linalg.norm(point_pair[0]- point_pair[1])

        if dist1 < dist2: 
            scale = dist1/dist2
            new_shape = tuple(np.round(np.array(stack_2.shape)*scale).astype(int))
            downscaled_stack = resize(stack_2, new_shape, mode='constant', anti_aliasing=True)
            # volume = zoom(volume, (0.5, 0.5, 2))
            # volume = resize(volume, (x,y,z))
        else: 
            scale = dist2/dist1
            new_shape = tuple(np.round(np.array(stack_1.shape)*scale).astype(int))
            downscaled_stack = resize(stack_1, new_shape, mode='constant', anti_aliasing=True)
            # volume = zoom(volume, (0.5, 0.5, 2))
            # volume = resize(volume, (x,y,z))

        return downscaled_stack

    def shift_volume(self,volume, x,y,z):
        pass



        #TODO: if transforms valid and nothing else missing, reveal next windwo
class customWidget(QWidget):
    def __init__(self):
        super(customWidget, self).__init__()
        self.setMinimumSize(300,300)
        self.markers = [None,None,None]
        self.stack = ImageView()
        self.imgs = None
        self.imgs_dict = {}
        self.stack.mouseClickSig.connect(self.marker_event)
        self.elements = QComboBox()
        self.elements.addItems(["Channel1"])
        self.elements.setMaximumWidth(60)
        self.elements.currentIndexChanged.connect(self.dropdown_changed)
        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setObjectName("sld")
        self.sld.valueChanged.connect(self.slider_changed)
        self.el_sl = QHBoxLayout()
        self.el_sl.addWidget(self.elements)
        self.el_sl.addWidget(self.sld)
        self.clr = QPushButton("clear file list")
        self.clr.clicked.connect(self.clear_list)
        self.files = DragAndDropListWidget()
        self.files.setObjectName("files")
        self.files.filesAddedSig.connect(self.load_files)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stack)
        self.layout.addLayout(self.el_sl)
        self.layout.addWidget(self.clr)
        self.layout.addWidget(self.files)
        self.setLayout(self.layout)

    def dropdown_changed(self):
        try: 
            el = self.elements.currentText()
            self.stack.image_view.setImage(self.imgs_dict[el][0])
            self.imgs = self.imgs_dict[el]
            self.sld.setRange(0,self.imgs_dict[el].shape[0]-1)
            self.sld.setValue(0)
        except: 
            print("probably got here durring the initial transformation function, ignore this. ")

    def slider_changed(self):
        self.stack.image_view.setImage(self.imgs[self.sld.value()])
        color = ["red", "green", "blue"]
        self.spots_data = []
        for i in range(3):
            if self.markers[i] is not None and self.markers[i][0] == self.sld.value(): 
                self.spots_data.append({'pos':self.markers[i][1::], 'brush': f'{color[i]}', 'pen': f'{color[i]}'})
        self.stack.scatter.clear()
        self.stack.scatter.setData(self.spots_data)

    def marker_event(self, button, x_pos, y_pos):
        if button == 1: #new marker 
            if self.markers[0] is None: 
                self.markers[0] = np.array([self.sld.value(), x_pos, y_pos])
            elif self.markers[1] is None: 
                self.markers[1] = np.array([self.sld.value(), x_pos, y_pos])
            elif self.markers[2] is None: 
                self.markers[2] = np.array([self.sld.value(), x_pos, y_pos])
        if button == 2: #remove marker
            diffs = []
            for i in self.markers:
                if i is None: 
                    diffs.append(np.inf)
                else: 
                    diffs.append(math.dist((x_pos, y_pos), i[1::]))
            if np.array(diffs).any() is not np.inf: 
                if any(np.array(diffs)<100):
                    minpos = diffs.index(min(diffs))
                    self.markers[minpos] = None
                else: 
                    return
                    
        color = ["red", "green", "blue"]
        self.spots_data = []
        for i in range(3):
            if self.markers[i] is not None and self.markers[i][0] == self.sld.value(): 
                self.spots_data.append({'pos':self.markers[i][1::], 'brush': f'{color[i]}', 'pen': f'{color[i]}'})
        self.stack.scatter.clear()
        self.stack.scatter.setData(self.spots_data)

    def clear_list(self):
        self.files.clear()
        self.stack.image_view.clear()
        self.stack.scatter.clear()
        self.elements.clear()
        self.elements.addItem("Channel1")

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
            canvas = np.empty(shp)
            for i in range(len(files)):
                canvas[i] = imageio.v3.imread(files[i])
            
        if canvas.shape[1]>500:
            ##temp resize 
            downscaled = self.rescale_3d_array(canvas,0.08)
            self.imgs = downscaled
        else: 
            self.imgs = canvas

        self.sld.setRange(0,self.imgs.shape[0]-1)
        self.stack.image_view.setImage(self.imgs[0])
        self.stack.original_pos = self.stack.image_view.pos()
        print(self.stack.original_pos)
        self.stack.reset_view()

    def rebin_3d_array(self, arr, scale_factor):
        # Determine the new shape after rebinning
        new_shape = tuple(np.array(arr.shape) // scale_factor)
        
        # Reshape the array into a shape compatible with rebinning
        reshaped_arr = arr[:new_shape[0]*scale_factor[0], :new_shape[1]*scale_factor[1], :new_shape[2]*scale_factor[2]]
        reshaped_arr = reshaped_arr.reshape((new_shape[0], scale_factor[0], new_shape[1], scale_factor[1], new_shape[2], scale_factor[2]))
        
        # Sum elements within each bin
        rebinned_arr = reshaped_arr.sum(axis=(1, 3, 5))
        
        return rebinned_arr

def rebin_3d_array(arr, scale_factor):
    # Determine the new shape after rebinning
    new_shape = tuple(np.array(arr.shape) // scale_factor)
    
    # Reshape the array into a shape compatible with rebinning
    reshaped_arr = arr[:new_shape[0]*scale_factor[0], :new_shape[1]*scale_factor[1], :new_shape[2]*scale_factor[2]]
    reshaped_arr = reshaped_arr.reshape((new_shape[0], scale_factor[0], new_shape[1], scale_factor[1], new_shape[2], scale_factor[2]))
    
    # Sum elements within each bin
    rebinned_arr = reshaped_arr.sum(axis=(1, 3, 5))
    
    return rebinned_arr


class ImageView(pg.GraphicsLayoutWidget):
    mouseMoveSig = pyqtSignal(int,int, name= 'mouseMoveSig')
    mouseClickSig =pyqtSignal(int,float,float, name= 'mouseClickSig')
    def __init__(self):
        super(ImageView, self).__init__()
        self.setMinimumSize(300,300)
        self.initUI()

    def initUI(self):
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setBackground("k")
        self.arr = None
        self.vol_layers = {0:None} #keep record of which layer contains scatter spots {5:[0,2]} layer 5 contains spots red(0) and blue(2)
        self.spots_data = []
        self.spots_pos = [None, None, None]   
        self.last_moving_pos = None
        self.start_pos = None
        self.offset_x = None
        self.offset_y = None
        self.img_pos = None
        self.original_pos = None
        self.zoom_sf = 1
        self.image_view = pg.ImageItem()
        self.zoom_view = pg.ImageItem()

        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.roi = pg.CircleROI(pos=(20,20), size=40, pen=pg.mkPen(color=(255, 0, 0), width=3), handlePen=(0,255,0))
        self.roi.handleSize=1
        self.roi.sigRegionChanged.connect(self.updateRoi)
        border_pen = pg.mkPen(color=(255, 255, 255), width=2)

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
        self.roi.removeHandle(0)

    def setZoomLimits(self, yrange, xrange):
        self.v2.setXRange(0, xrange, padding=0)
        self.v2.setYRange(0, yrange, padding=0)
        x = int(np.floor(xrange*0.025))
        y = int(np.floor(yrange*0.025))

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
        elif ev.key() == 91: #[
            size = self.roi.size()
            self.roi.setSize(size-1)
        elif ev.key() == 93: #]
            size = self.roi.size()
            self.roi.setSize(size+1)
        else:
            super().keyPressEvent(ev)

    def reset_view(self):
        #get image_view position first
        prev_pos = self.image_view.pos()
        self.scatter.moveBy(-prev_pos.x(), -prev_pos.y())
        self.zoom(1)
        self.setZoomLimits(self.image_view.height(), self.image_view.width())
        print(self.image_view.height(), self.image_view.height())
        self.image_view.setPos(0,0)

    def zoom(self, factor):
        try:
            self.v2.scaleBy((factor, factor), center=(self.moving_pos.x(), self.moving_pos.y()))
        except: 
            self.v2.scaleBy((factor, factor), center=(0, 0))

    def wheelEvent(self,ev):
        print(ev.angleDelta().y())
        if ev.angleDelta().y()<0:
            self.zoom_sf=1.03
            self.zoom(self.zoom_sf)
        elif ev.angleDelta().y()>0:
            self.zoom_sf=0.97
            self.zoom(self.zoom_sf)
        self.moving_pos = self.v2.mapSceneToView(ev.pos())
        print(self.v2.pos())

    def mouseMoveEvent(self, ev):
        self.v2.setCursor(Qt.BlankCursor) 
        self.moving_pos = self.v2.mapSceneToView(ev.pos())
        self.mouseMoveSig.emit(self.moving_pos.x(), self.moving_pos.y())
        self.roi.setPos([self.moving_pos.x(), self.moving_pos.y()], finish=False)
        if self.start_pos is None: 
            self.start_pos = self.moving_pos
        diff = self.moving_pos - self.start_pos
        if self.last_moving_pos is None:
            self.last_moving_pos = self.start_pos
        inc = self.moving_pos - self.last_moving_pos
        
        if ev.buttons() == Qt.LeftButton and self.start_pos !=  self.moving_pos: #if left-click-hold and dragging
            if self.image_view.width() is None: 
                return
            self.image_view.setPos(diff.x() + self.img_pos.x(), diff.y() + self.img_pos.y())
            self.scatter.moveBy(inc.x(), inc.y())
            
        # print(self.moving_pos.x(), self.moving_pos.y())
        self.last_moving_pos = self.moving_pos

    def mousePressEvent(self, ev):
        self.start_pos = self.v2.mapSceneToView(ev.pos())
        return
            
    def mouseReleaseEvent(self, ev):
        self.end_pos = self.v2.mapSceneToView(ev.pos())
        self.img_pos = self.image_view.pos()

        print("end pos:", self.end_pos.x(), self.end_pos.y())
        print(self.img_pos.x(), self.img_pos.y())

        if self.image_view.image is None: 
            return
        if self.start_pos ==  self.end_pos:
            pos = (self.end_pos.x()-self.img_pos.x()+20, self.end_pos.y()-self.img_pos.y()+20)
            btn = ev.button()
            self.mouseClickSig.emit(btn, pos[0], pos[1])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


