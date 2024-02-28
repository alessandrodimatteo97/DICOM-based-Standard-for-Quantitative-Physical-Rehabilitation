import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QComboBox, QCheckBox, QStackedWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from pydicom import dcmread
from AnimatedHand import AnimatedHand
import pandas as pd
from pydicom.waveforms import multiplex_array  
import numpy as np

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton
from DICOMDetailsWindow import DICOMDetailsWindow

class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Waveform Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.dicomFiles = []
        self.selectedDicomFiles = []
        self.initUI()

    # Inside DICOMViewer's initUI method where FullHandAnimation is instantiated
    def initUI(self):
        self.stackedWidget = QStackedWidget(self)
        
        self.mainPage = QWidget()
        self.mainLayout = QHBoxLayout(self.mainPage)
        self.setupMainPage()

        self.newPage = AnimatedHand()  # Use FullHandAnimation as the second page
        self.stackedWidget.addWidget(self.mainPage)
        self.stackedWidget.addWidget(self.newPage)
        
        self.newPage.requestPageChange.connect(self.switchToMainPage)
        
        self.setCentralWidget(self.stackedWidget)

        
    def setupMainPage(self):
        self.controlsLayout = QVBoxLayout()
        # Button to load DICOM files
        self.loadButton = QPushButton("Load DICOM File")
        self.loadButton.setStyleSheet("font-weight: bold;")

        self.loadButton.clicked.connect(self.loadDICOM)
        self.controlsLayout.addWidget(self.loadButton)
        
        # ComboBox to select multiplex groups
        self.multiplexGroupSelector = QComboBox()
        self.multiplexGroupSelector.setStyleSheet("font-weight: bold;")

        self.controlsLayout.addWidget(self.multiplexGroupSelector)
        
        # Canvas for plotting
        self.canvas = FigureCanvas(plt.Figure())
        self.controlsLayout.addWidget(self.canvas)
        
        # Calculation buttons
        self.positionButton = QPushButton("Show Raw Data")
        self.positionButton.setStyleSheet("font-weight: bold;")

        self.positionButton.clicked.connect(lambda: self.calculateAndDisplay('raw'))
        self.controlsLayout.addWidget(self.positionButton)

        self.velocityButton = QPushButton("Calculate Velocity")
        self.velocityButton.setStyleSheet("font-weight: bold;")

        self.velocityButton.clicked.connect(lambda: self.calculateAndDisplay('velocity'))
        
        self.controlsLayout.addWidget(self.velocityButton)

        self.accelerationButton = QPushButton("Calculate Acceleration")
        self.accelerationButton.setStyleSheet("font-weight: bold;")

        self.accelerationButton.clicked.connect(lambda: self.calculateAndDisplay('acceleration'))
        self.controlsLayout.addWidget(self.accelerationButton)

        # New button to change page
        changePageButton = QPushButton("Go to Full Hand Animation")
        changePageButton.setStyleSheet("font-weight: bold;")

        changePageButton.clicked.connect(self.switchToFullHandAnimation)  # Remove lambda for direct method call

        self.controlsLayout.addWidget(changePageButton)
        
        self.checkBoxContainer = QWidget()
        self.checkBoxLayout = QVBoxLayout(self.checkBoxContainer)

        # Optionally, you can set a fixed size for the checkbox container
        self.checkBoxContainer.setFixedSize(200, 300)

        # Adding the controls layout and the checkbox container to the main layout
        self.mainLayout.addLayout(self.controlsLayout)
        self.mainLayout.addWidget(self.checkBoxContainer)
        

    
    def switchToMainPage(self):
        self.stackedWidget.setCurrentIndex(0)

    
    def loadDICOM(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)", options=options)
        if fileName:
            dicom_file = dcmread(fileName)
            checkBox = QCheckBox(fileName.split('/')[-1])
            checkBox.stateChanged.connect(self.onCheckboxStateChanged)
            checkBox.setChecked(True)
            details_button = QPushButton('Details')
            details_button.clicked.connect(lambda: self.showDICOMDetails(dicom_file))
            file_layout = QHBoxLayout()
            file_layout.addWidget(checkBox)
            file_layout.addWidget(details_button)
            self.checkBoxLayout.addLayout(file_layout)
            self.dicomFiles.append((fileName, dicom_file, checkBox))
            self.multiplex_groups = self.parse_dicom_for_load()
            self.prepareMultiplexGroupSelector()
    
    def onCheckboxStateChanged(self, state):
    
        print("Checkbox state changed. New state:", state)
         
        self.prepareMultiplexGroupSelector()
    
    def showDICOMDetails(self, dicom_file):
        # Convert DICOM data to a dictionary for easier display
        dicom_data = {
            tag: {
                'VR': dicom_file[tag].VR,
                'Value': str(dicom_file[tag].value),
                'Name': dicom_file[tag].name
            } for tag in dicom_file.keys()
        }
        details_window = DICOMDetailsWindow(dicom_data, self)
        details_window.exec_()


    
    def prepareMultiplexGroupSelector(self):
        self.multiplexGroupSelector.clear()
        
       
        selectedFiles = [] 
        multiplexLabel = []
        for filePath, dicomFile, checkBox in self.dicomFiles:
            if checkBox.isChecked():
                for multipleGroup in dicomFile.WaveformSequence:
                    multiplexLabel.append(multipleGroup.MultiplexGroupLabel)
                selectedFiles.append(dicomFile) 
                
        self.multiplexGroupSelector.addItems(pd.unique(multiplexLabel))
        

    
    
    def parse_dicom_for_load(self):
        waveform_info = []
        for filePath, dicomFile, checkBox in self.dicomFiles:
            if 'WaveformSequence' in dicomFile:
                dicom_dataset = dicomFile
                for idx, waveform in enumerate(dicom_dataset.WaveformSequence):
                    multiplex_group_label = waveform.MultiplexGroupLabel
                    channels = []
                    for ii, channel in enumerate(waveform.ChannelDefinitionSequence):
                        source = channel.ChannelSourceSequence[0].CodeMeaning
                        units = 'unitless'  # Default units
                        if 'ChannelSensitivity' in channel and 'ChannelSensitivityUnitsSequence' in channel:
                            units = channel.ChannelSensitivityUnitsSequence[0].CodeMeaning
                        channels.append({
                            'source': source,
                            'units': units,
                            'label': f"Channel {ii + 1}: {source} ({units})"
                        })
                    if {'label': multiplex_group_label,'channels': channels} not in waveform_info:
                        waveform_info.append({
                            'label': multiplex_group_label,
                            'channels': channels
                        })
        return waveform_info   
     
    
    
    def getSelectedDicomFiles(self):
        """Collects and returns selected DICOM files."""
        selected_dicomFile = []
        for _, dicomFile, checkBox in self.dicomFiles:
            #print(dicomFile)
            if checkBox.isChecked() and dicomFile.SOPClassUID == '1.1.111.10008.5.1.4.1.1.1.1.1':
                selected_dicomFile.append(dicomFile)
        return selected_dicomFile
    
    def switchToFullHandAnimation(self):
        selectedDicomFiles = self.getSelectedDicomFiles()
        if len(selectedDicomFiles) == 0: 
                QMessageBox.warning(self, "Selection Error", "at least one PR file needs to be selected.")
        else:
            print(len(selectedDicomFiles))
            self.newPage.loadData(selectedDicomFiles)  
            self.stackedWidget.setCurrentIndex(1)  
        
    def getRawData(self, dicomFile, index):
    # Extract and convert the waveform data for each lead
        waveform = dicomFile.WaveformSequence[index]
        channels = (waveform[0x03a0010].value)
        points = waveform[(0x03a0005)].value

        lead_data_bytes = waveform.WaveformData
        num_bits = waveform.WaveformBitsAllocated
        dtype = 'float16'
        if num_bits in (8, 16, 32, 64):
            dtype_prefix = 'float'  
            dtype = f'{dtype_prefix}{num_bits}'
        else:
            raise ValueError("Unsupported number of bits. Choose from 8, 16, 32, or 64.")
        lead_data_floats = np.frombuffer(lead_data_bytes, dtype=dtype)
        reshaped_data = lead_data_floats.reshape((channels, points))
        return reshaped_data



    def calculateAndDisplay(self, calculation_type):
        group_index = None
        groupLabel = self.multiplexGroupSelector.currentText()

        self.selectedDicomFiles = []
        timestampGroupIndices = []  

        for _, dicomFile, checkBox in self.dicomFiles:
            not_present = True
            timestampGroupIndex = None  
            if checkBox.isChecked():
                for idx, waveform in enumerate(dicomFile.WaveformSequence):
                    if waveform.MultiplexGroupLabel == groupLabel:
                        self.selectedDicomFiles.append(dicomFile)
                        group_index = idx
                        not_present = False
                    elif waveform.MultiplexGroupLabel == 'Timestamp':
                        timestampGroupIndex = idx  

                timestampGroupIndices.append(timestampGroupIndex) 

                if not_present:
                    QMessageBox.warning(self, "Selection Error", f"the multiplex group selected is not in {_.split('/')[-1]}")

        if len(self.selectedDicomFiles) == 0:
            QMessageBox.warning(self, "Selection Error", "Please select at least one DICOM file.")
            return

        items = [item[1] for item in self.dicomFiles]

        self.canvas.figure.clear()  # Clear previous plots if any
        if not self.selectedDicomFiles:
            return  # Guard against no selected files
        num_channels = multiplex_array(self.selectedDicomFiles[0], group_index, as_raw=False).shape[1]

        axs = self.canvas.figure.subplots(num_channels, 1, squeeze=False)

        self.canvas.figure.suptitle("Trajectory" if calculation_type == 'raw' and groupLabel != 'Timestamp' and dicomFile.SOPClassUID == '1.1.111.10008.5.1.4.1.1.1.1.1'  else calculation_type.capitalize(), fontsize=16)

        for file_idx, dicomFile in enumerate(self.selectedDicomFiles):
            waveform = dicomFile.WaveformSequence[group_index]
            raw_data = self.getRawData(dicomFile, group_index)

            timestamp = None 
            if timestampGroupIndices[file_idx] is not None:
                try:
                    
                    timestamp = self.getRawData(dicomFile, timestampGroupIndices[file_idx])
                    timestamp = timestamp - timestamp[0]
                except IndexError:
                    print(f"Warning: Timestamp group not found in file {file_idx+1}. Falling back to index-based plotting.")

            for i in range(min(num_channels, len(waveform.ChannelDefinitionSequence))):  # Ensure channel index is in bounds
                if calculation_type == 'raw':
                    data_to_plot = raw_data[:, i]
                    time_x_axis = np.arange(len(data_to_plot)) if timestamp is None else timestamp.flatten()[:len(data_to_plot)]
                else:
                    if timestamp is not None:
                        time_intervals = np.diff(timestamp.flatten())
                    else:
                        # Assume uniform sampling interval based on SamplingFrequency
                        time_intervals = np.full(raw_data.shape[0] - 1, 1 / waveform.SamplingFrequency)

                    if calculation_type == 'velocity':
                        # Calculate velocity as the first derivative of position
                        data_diffs = np.diff(raw_data[:, i])
                        data_to_plot = data_diffs / time_intervals
                        time_x_axis = timestamp.flatten()[:-1] if timestamp is not None else np.arange(len(data_to_plot))
                    elif calculation_type == 'acceleration':
                        # Calculate acceleration as the first derivative of velocity
                        data_diffs = np.diff(raw_data[:, i])
                        velocities = data_diffs / time_intervals
                        acceleration = np.diff(velocities) / time_intervals[:-1]
                        data_to_plot = acceleration
                        time_x_axis = timestamp.flatten()[:-2] if timestamp is not None else np.arange(len(data_to_plot))

                label = f"file: {items.index(dicomFile)+1}" if len(items) > 1 else ' '
                axs[i, 0].plot(time_x_axis, data_to_plot, label=label)
                
                if dicomFile.SOPClassUID == '1.1.111.10008.5.1.4.1.1.1.1.1':
                    type_ = ''
                    measure = ' (m)'
                    if calculation_type == 'velocity':
                        type_ = calculation_type[0].capitalize()
                        measure = ' (m/s)'
                    elif calculation_type == 'acceleration':
                        type_ = calculation_type[0].capitalize()
                        measure = ' (m/s^2)'
                            
                    axs[i, 0].set_ylabel(type_+waveform.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning+measure)
                    #"waveform.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning")
                    #axs[i, 0].set_ylabel(f"{waveform.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning}s{measure}")
                    axs[i, 0].set_xlabel("Time (sec)" if timestamp is not None else "sample (points)")
                    axs[i, 0].legend()
                else:
                    type_ = ''
                    measure = ' (m)'
                    if calculation_type == 'velocity':
                        type_ = calculation_type[0].capitalize()
                        measure = ' (m/s)'
                    elif calculation_type == 'acceleration':
                        type_ = calculation_type[0].capitalize()
                        measure = ' (m/s^2)'

                    #axs[i, 0].set_ylabel(type_+waveform.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning+measure) #"waveform.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning")
                    axs[i, 0].set_ylabel(waveform.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning, rotation='horizontal', labelpad=50)
                    #axs[i, 0].set_ylabel(f"{waveform.ChannelDefinitionSequence[i].ChannelSourceSequence[0].CodeMeaning}s{measure}")
                    axs[i, 0].set_xlabel("Time (sec)" if timestamp is not None else "sample (points)")
                    axs[i, 0].legend()
                    

        self.canvas.figure.tight_layout()
        self.canvas.draw()
    
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())
