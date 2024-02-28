# DICOM-based-Standard-for-Quantitative-Physical-Rehabilitation

## Description

This repository showcase a proof of concept of the DICOM data format with PR (Physical Rehabiliation) data, for the Virtual Glove assistive system for the QPR (Quantitative Physical Rehabilitation) of the hand. The principal objective of this code is to demonstrate the efficacy of employing DICOM format for PR data without requiring substantial adjustments. 
 


### Code Structure

1. **Data/**: Folder containing all the data in csv format extracted from a performed task by a patient using the VG in Unity3D
2. **DICOM-PR Viewer/**: Folder containing the code of the DICOM-PR Viewer
3.  **Screen recording Viewer/**: Folder containing a video recording of the DICOM-PR Viewer to showcase the execution
4.  **Translator/**: Folder containing the python notebook 'Translator' to create the DICOM file with PR data with all the CSV Data included 
5.  **Screenshots Viewer/**: Folder containing screenshots of the DICOM-PR Viewer 


### Prerequisites

1. **Clone or Download the Pipeline Code:**
   - Clone this repository to your local machine


2. **IDE Selection:**
   - [Download](https://code.visualstudio.com/download) and install Visual Studio Code (VS Code) or use an IDE of your choice.


3. **Environment Setup:**
   - Open terminal in VS Code (View → Terminal).
   - Execute the following commands to set the conda environment "approach"
     ```bash
      conda env create -f environment.yml
      conda activate dcmenv
      ```


4.  **Viewer Execution:**
    - Execute the following command to run the pipeline:
        ```bash
        cd DICOM-PR\ Viewer/
        python DicomViewer.py
        ```
    - The viewer is running
    - click on 'Load DICOM File' button and select the DICOM file 'example.dcm' placed in Translator directory





