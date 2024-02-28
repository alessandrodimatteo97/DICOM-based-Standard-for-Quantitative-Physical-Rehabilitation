from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView

class DICOMDetailsWindow(QDialog):
    def __init__(self, dicom_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DICOM File Details")
        self.setGeometry(100, 100, 600, 400)  # Adjust size as needed
        self.dicom_data = dicom_data
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(self.dicom_data))  # Set the number of rows
        self.tableWidget.setColumnCount(4)  # Tag, VR, Value, Name
        
        # Set the column headers
        self.tableWidget.setHorizontalHeaderLabels(['Tag', 'VR', 'Name','Value'])
        
        # Populate the table rows with DICOM data
        for row, (tag, data) in enumerate(self.dicom_data.items()):
            self.tableWidget.setItem(row, 0, QTableWidgetItem(str(tag)))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(data['VR']))
            self.tableWidget.setItem(row, 2, QTableWidgetItem(data['Name']))
            self.tableWidget.setItem(row, 3, QTableWidgetItem(data['Value']))
        
        # Resize columns to fit content
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Optional: Resize rows to fit content
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout.addWidget(self.tableWidget)
        self.setLayout(layout)
