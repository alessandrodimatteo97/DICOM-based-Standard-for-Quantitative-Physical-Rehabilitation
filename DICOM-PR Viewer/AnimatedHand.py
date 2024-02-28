import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import pydicom
from pydicom import dcmread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont


class AnimatedHand(QWidget):
    requestPageChange = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Animation Trajectory')
        self.current_frame = 0
        self.paused = False
        self.start = True
        self.selected_finger = 'All'
        self.plot_limits = None
        self.ani = None
        self.userAdjustedAxes = False  # Track if the user has adjusted axes
        self.initUI()
        font = QFont()

        #self.loadData([self.create_df('QPR_VG_1.dcm'), self.create_df('my_waveform_10Feb2024.dcm')])
        #self.calculatePlotLimits()
        #self.initAnimation()

    def initUI(self):
        main_layout = QVBoxLayout()

        controls_layout = QHBoxLayout()

        self.finger_selection_combo = QComboBox()
        self.finger_selection_combo.addItem('All')
        self.finger_selection_combo.addItems(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
        self.finger_selection_combo.setStyleSheet("font-weight: bold;")
        self.finger_selection_combo.currentTextChanged.connect(self.onFingerSelectionChange)
        controls_layout.addWidget(self.finger_selection_combo)

        self.pause_button = QPushButton('Start')
        self.pause_button.setStyleSheet("font-weight: bold;")

        self.pause_button.clicked.connect(self.togglePause)
        
        controls_layout.addWidget(self.pause_button)

        self.next_frame_button = QPushButton('Next Frame')
        self.next_frame_button.setStyleSheet("font-weight: bold;")

        self.next_frame_button.clicked.connect(self.nextFrame)
        controls_layout.addWidget(self.next_frame_button)

        self.prev_frame_button = QPushButton('Previous Frame')
        self.prev_frame_button.setStyleSheet("font-weight: bold;")

        self.prev_frame_button.clicked.connect(self.prevFrame)
        controls_layout.addWidget(self.prev_frame_button)

        main_layout.addLayout(controls_layout)

        self.full_width_button = QPushButton('Back to Main Page')
        self.full_width_button.setStyleSheet("font-weight: bold;")

        self.full_width_button.clicked.connect(self.onBackButtonClicked)
        main_layout.addWidget(self.full_width_button)

        self.figure = Figure(figsize=(10, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)  # Connect scroll event
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def loadData(self, dicomFiles):
        #self.loadData([self.create_df('QPR_VG_1.dcm'), self.create_df('my_waveform_10Feb2024.dcm')])
        #self.calculatePlotLimits()
        #self.initAnimation()
        #self.dataframes = [pd.DataFrame(path) for path in file_paths]
        self.dataframes = [pd.DataFrame(self.create_df(dcm)) for dcm in dicomFiles]
        self.calculatePlotLimits()
        self.initAnimation()
    
        
    def onBackButtonClicked(self):
        if self.ani is not None:
            self.ani.event_source.stop()  # Stop the animation
        self.figure.clear()  # Clear the figure to prevent background plots
        self.canvas.draw_idle()  # Update the canvas after clearing the figure
        self.dataframes = []
        self.current_frame = 0
        self.paused = False
        self.start = True
        self.plot_limits = None
        self.userAdjustedAxes = False
        self.pause_button.setText('Start')
        self.requestPageChange.emit()
        # Optionally reinitialize the plot or figure here if needed



    def create_df(self,dicom):
        print("creazione dei DF")
        dfs = {}
        ds = (dicom)
        generator = []
        for waveform in ds.WaveformSequence:
            channels = waveform[0x03a0010].value
            points = waveform[0x03a0005].value

            lead_data_bytes = waveform.WaveformData
            num_bits = waveform.WaveformBitsAllocated
            dtype = 'float16'
            if num_bits in (8, 16, 32, 64):
                dtype_prefix = 'float'  # or 'float' for floating points, though float needs to match exact sizes (e.g., 32, 64)
                dtype = f'{dtype_prefix}{num_bits}'
            else:
                raise ValueError("Unsupported number of bits. Choose from 8, 16, 32, or 64.")
    
            lead_data_floats = np.frombuffer(lead_data_bytes, dtype=dtype)
            reshaped_data = lead_data_floats.reshape((channels, points))
            generator.append(reshaped_data)

        for mplx, arr in zip(ds.WaveformSequence, generator):
            nr_channels = mplx.NumberOfWaveformChannels
            nr_samples = mplx.NumberOfWaveformSamples
            mplx_label = mplx.MultiplexGroupLabel  # Assuming this gives each multiplex group a unique label

            data = {}

            for ch_idx in range(nr_channels):
                ch_item = mplx.ChannelDefinitionSequence[ch_idx]
                ch_source = ch_item.ChannelSourceSequence[0].CodeMeaning

                column_name = f"{mplx_label}_{ch_source}"

                # Exclude columns with "Timestamp" in the name
                if "Timestamp" not in column_name:
                    # Extract the channel data
                    data[column_name] = arr[..., ch_idx]

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(data)
            print(df.columns)
            dfs[mplx_label] = df

        # Concatenate all DataFrames while ignoring any columns related to Timestamps
        return pd.concat(dfs.values(), axis=1)
    
    def create_df_from_dicom(self, dicom_file):
        print("Creating DataFrame from DICOM")
        ds = dcmread(dicom_file)  # Load DICOM file
        dfs = {}  # Dictionary to hold DataFrames for each waveform sequence

        for waveform in ds.WaveformSequence:
            mplx_label = waveform.MultiplexGroupLabel
            channels = waveform.NumberOfWaveformChannels
            if mplx_label == 'Timestamp':
                continue
            points = waveform.NumberOfWaveformSamples

            lead_data_bytes = waveform.WaveformData
            
            expected_size = channels * points * np.dtype(np.float16).itemsize
            if len(lead_data_bytes) != expected_size:
                print(f"Error: Mismatch in expected data size. Expected {expected_size} bytes, got {len(lead_data_bytes)} bytes.")
                continue  # Skip this waveform if sizes do not match

            lead_data_floats = np.frombuffer(lead_data_bytes, dtype=np.float16)
            reshaped_data = lead_data_floats.reshape((channels, points))
                
            data = {}
            for ch_idx in range(channels):
                ch_item = waveform.ChannelDefinitionSequence[ch_idx]
                ch_source = ch_item.ChannelSourceSequence[0].CodeMeaning
                column_name = f"{mplx_label}_{ch_source}"
                if "Timestamp" not in column_name:
                    data[column_name] = reshaped_data[ch_idx, :]

            df = pd.DataFrame(data)
            dfs[mplx_label] = df

        final_df = pd.concat(dfs.values(), axis=1)
        return final_df



    def calculateNormalizationParameters(self):
        all_data = pd.concat(self.dataframes, ignore_index=True)
        self.min_max_values = {
            'x': (all_data.filter(like='_x').min().min(), all_data.filter(like='_x').max().max()),
            'y': (all_data.filter(like='_y').min().min(), all_data.filter(like='_y').max().max()),
            'z': (all_data.filter(like='_z').min().min(), all_data.filter(like='_z').max().max()),
        }

    def normalizeData(self, values, axis):
        if self.min_max_values is None:
            raise ValueError("Normalization parameters not calculated.")
        min_val, max_val = self.min_max_values[axis]
        return [(val - min_val) / (max_val - min_val) for val in values]

    def plotFingerData(self, df, frame, color, hand_label):
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] if self.selected_finger == 'All' else [self.selected_finger]
        for index, finger in enumerate(fingers):
            x, y, z = [], [], []
            for joint in ['_Carp_', '_Mcp_', '_Pip_', '_Dip_', '_Btip_']:
                x += df.filter(like=f"{finger}{joint}x").iloc[frame].tolist()
                y += df.filter(like=f"{finger}{joint}y").iloc[frame].tolist()
                z += df.filter(like=f"{finger}{joint}z").iloc[frame].tolist()

            if x and y and z:
                # Label only the first finger plot of each hand
                if index == 0:
                    self.ax.plot(x, z, y, '-o', color=color,  markerfacecolor='white', label=hand_label)
                else:
                    self.ax.plot(x, z, y, '-o', color=color, markerfacecolor='white')

    def handCenter(self, df, frame):
        """
        Calculate the center of the hand data for a given frame.
        """
        x, y, z = [], [], []
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        for finger in fingers:
            for joint in ['_Carp_', '_Mcp_', '_Pip_', '_Dip_', '_Btip_']:
                x += df.filter(like=f"{finger}{joint}x").iloc[frame].tolist()
                y += df.filter(like=f"{finger}{joint}y").iloc[frame].tolist()
                z += df.filter(like=f"{finger}{joint}z").iloc[frame].tolist()
        center_x = np.mean(x)
        center_y = np.mean(y)
        center_z = np.mean(z)
        return center_x, center_y, center_z

    def calculatePlotLimits(self):
        """
        Calculate and store the plot limits based on the first frame of the first hand data.
        This method assumes the hand to center on is represented by the first dataframe.
        """
        center_x, center_y, center_z = self.handCenter(self.dataframes[0], 0)
        plot_radius = 0.5  # Adjust as needed
        self.plot_limits = {
            'x': [center_x - plot_radius, center_x + plot_radius],
            'y': [center_z - plot_radius, center_z + plot_radius],
            'z': [center_y - plot_radius, center_y + plot_radius]
        }

    
        
    def update(self, frame):
        if self.current_frame >= max(len(df) for df in self.dataframes) - 1:
            if not self.paused:
                self.ani.event_source.stop()
                self.pause_button.setText('Restart')
                self.paused = True  # Indicate that the animation is paused at the end
            return  # Stop updating if beyond the last frame

        self.current_frame = frame
        self.ax.clear()
        # Set axis labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)')
        self.ax.set_zlabel('Y (m)')

        if self.plot_limits:
            self.ax.set_xlim(self.plot_limits['x'])
            self.ax.set_ylim(self.plot_limits['y'])
            self.ax.set_zlim(self.plot_limits['z'])

        # Dynamic hand labels and colors, could be predefined or generated
        hand_labels = [f'Right Hand' for i in range(len(self.dataframes))]
        colors = ['red', 'blue', 'black', 'green', 'orange'] * (len(self.dataframes) // 5 + 1)  # Example way to cycle colors

        for df, color, hand_label in zip(self.dataframes, colors, hand_labels):
            if frame < len(df):
                self.plotFingerData(df, frame, color, hand_label)
            else:
                # If the frame exceeds the length of df, keep the last frame displayed
                self.plotFingerData(df, len(df) - 1, color, hand_label)

        self.ax.legend()
        self.canvas.draw()

        if all(self.current_frame >= len(df) for df in self.dataframes):
            self.ani.event_source.stop()
            self.pause_button.setText('Restart')
            self.paused = True  # Set paused to True to avoid automatic restart


    def togglePause(self):
        if self.start:
            # This is the first start, initialize the animation
            self.startAnimation()
        elif self.paused:
            # If paused, resume or restart
            if self.current_frame >= max(len(df) for df in self.dataframes) - 1:
                # If at the end, restart
                self.current_frame = 0
                self.startAnimation()
            else:
                # If not at the end, resume
                self.ani.event_source.start()
                self.pause_button.setText('Pause')
            self.paused = False
        else:
            # If running, pause
            self.ani.event_source.stop()
            self.pause_button.setText('Resume')
            self.paused = True


    def startAnimation(self):
        self.ani = FuncAnimation(self.figure, self.update, frames=range(self.current_frame, max(len(df) for df in self.dataframes)), interval=50)
        self.pause_button.setText('Pause')
        self.start = False
        self.paused = False
        # You might need to ensure the animation starts from the current frame
        self.canvas.draw_idle()



    # Adjust nextFrame and prevFrame to respect the end of dataframes
    def nextFrame(self):
        max_frames = max(len(df) for df in self.dataframes)
        if self.paused and self.current_frame + 1 < max_frames:
            self.current_frame += 1
            self.update(self.current_frame)
            self.canvas.draw()

    def prevFrame(self):
        if self.paused and self.current_frame - 1 >= 0:
            self.current_frame -= 1
            self.update(self.current_frame)
            self.canvas.draw()


    def resetUserAdjustments(self):
        self.userAdjustedAxes = False
        
    def on_scroll(self, event):
        # Decide the zoom factor (how much you want to zoom in or out)
        zoom_factor = 0.1  # Smaller value for finer control, adjust as needed
        
        # Determine the scale factor based on the scroll direction
        if event.button == 'up':  # Zoom in
            scale_factor = 1 - zoom_factor
        elif event.button == 'down':  # Zoom out
            scale_factor = 1 + zoom_factor
        else:
            return  # Do nothing for other scroll events
        
        # Apply the scale factor to zoom in or out
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
        self.ax.set_xlim([x * scale_factor for x in xlim])
        self.ax.set_ylim([y * scale_factor for y in ylim])
        self.ax.set_zlim([z * scale_factor for z in zlim])
        
        # After adjusting the limits, capture the new limits to maintain them across updates
        self.plot_limits['x'] = self.ax.get_xlim()
        self.plot_limits['y'] = self.ax.get_ylim()
        self.plot_limits['z'] = self.ax.get_zlim()
        
        self.userAdjustedAxes = True  # Indicate that user has adjusted the plot
        
        self.canvas.draw()  # Redraw the canvas with the new zoom level


    def onFingerSelectionChange(self, selection):
        self.selected_finger = selection
        self.update(self.current_frame)  # Update the plot with the new selection
        self.canvas.draw()

    def initAnimation(self):
        # Initialize the 3D subplot and apply fixed plot limits here
        self.max_frames = max(len(df) for df in self.dataframes)
        self.ax = self.figure.add_subplot(111, projection='3d', auto_add_to_figure=False)
        self.figure.add_axes(self.ax)
        # Apply fixed plot limits
        if self.plot_limits:
            self.ax.set_xlim(self.plot_limits['x'])
            self.ax.set_ylim(self.plot_limits['y'])
            self.ax.set_zlim(self.plot_limits['z'])
        self.update(0)  # Display the first frame

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = AnimatedHand()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec_())
