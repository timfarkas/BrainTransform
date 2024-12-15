from torch.utils.data import Dataset
import os
import logging
import traceback
import shutil
import zipfile
import random
import re
import numpy as np
import contextlib
import mne
import h5py
import pickle as pkl
import warnings

class DatasetDownloader:
    """
    A class to handle downloading and preparing datasets for processing.

    Methods
    -------
    __init__(downloadAndPrepareImmediately=True, processImmediately=True, processingMode='raw', downloadURLs=None, datasetPath=None, logger=None, verbose=False)
        Initializes the DatasetDownloader with optional immediate download and processing.

    startDownloadingAndPrepare()
        Initiates the downloading and preparation of the dataset. (Call if downloadAndPrepareImmediately = False)
    """
    def __init__(self, downloadAndPrepareImmediately = True, processImmediately = True,**kwargs):
        defaultDownloadURLs = [ ## links to normalized data of all participants
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub01-04.zip",
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub05-08.zip",
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub09-12.zip",
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub13-16.zip"
        ]
        passedDownloadURLs = kwargs.get('downloadURLs', defaultDownloadURLs)
        self.downloadURLs = passedDownloadURLs if passedDownloadURLs else defaultDownloadURLs

        self.datasetPath = kwargs.get('datasetPath', os.path.join('/srv','synaptech_openfmri'))

        self.logger = kwargs.get('logger', None)
        self.verbose = kwargs.get('verbose', False)

        self.processingMode = kwargs.get("processingMode",'raw')

        if self.logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        if downloadAndPrepareImmediately:
            if os.path.exists(os.path.join(self.datasetPath,'.randomized')) and os.path.exists(os.path.join(self.datasetPath,'.arranged')) and os.path.exists(os.path.join(self.datasetPath,'.unzipped')) and os.path.exists(os.path.join(self.datasetPath,'.downloaded')):
                self.logger.info("Skipping downloading and preparing... (Found .downloaded, .unzipped, .arranged, .randomized files)")
            else:
                self.startDownloadingAndPrepare()
        if processImmediately:
            if os.path.exists(os.path.join(self.datasetPath,'.processed')):
                self.logger.info("Skipping processing...")
            else:
                
                processer = DatasetPreprocesser(datasetPath = self.datasetPath, 
                                                processImmediately = True, # is the pre-processor converting the data into a different format? 
                                                mode = self.processingMode, 
                                                logger = self.logger,
                                                verbose = self.verbose)

    def startDownloadingAndPrepare(self):
        assert self.downloadURLs is not None and len(self.downloadURLs)>0, f"No download URLs specified. (Given {self.downloadURLs})"
        self.logger.info(f"Downloading and processing entire dataset using URLs: {self.downloadURLs}")
        self._downloadDataset()
        self._prepareDataset()
        return True

    def _downloadDataset(self, datasetPath=None, downloadURLs=None):
        """
        Downloads the dataset from the specified URLs to the given dataset path.

        Parameters:
        - datasetPath (str, optional): The directory path where the dataset will be downloaded.
            Defaults to the instance's datasetPath attribute if not provided.
        - downloadURLs (list, optional): A list of URLs from which to download the dataset.
            Defaults to the instance's downloadURLs attribute if not provided.

        This method checks if the dataset path exists and creates it if it doesn't.
        It then iterates over the download URLs, downloading each file if it doesn't
        already exist in the dataset path. Successfully downloaded files are logged.
        """
        if datasetPath is None:
            datasetPath = self.datasetPath
        if downloadURLs is None:
            downloadURLs = self.downloadURLs
        if not os.path.exists(datasetPath):
            os.makedirs(datasetPath)
        
        self.downloadedFolders = []
        downloaded_marker = os.path.join(datasetPath, '.downloaded')
        
        if not os.path.exists(downloaded_marker):
            self.logger.info(f"Downloading {len(downloadURLs)} files...")
            for url in downloadURLs:
                file_name = os.path.join(datasetPath, url.split('/')[-1])
                if not os.path.exists(file_name):
                    self.logger.info(f"Downloading {file_name}...")
                    try:
                        os.system(f"wget -O {file_name} {url}")
                        self.downloadedFolders.append(file_name)
                    except Exception as e:
                        self.logger.error(f"Error downloading file {file_name}, {e}")
                        traceback.print_exc()
                else:
                    self.logger.info(f"{file_name} already exists, skipping download.")
            
            if len(self.downloadedFolders) == len(downloadURLs):
                with open(downloaded_marker, 'w') as f:
                    f.write('Download completed successfully.')
                self.logger.info(f"Successfully downloaded {len(self.downloadedFolders)} files.")
        else:
            self.logger.info("Dataset already downloaded. Skipping download.") 


    def _prepareDataset(self):
        """
        Iterates through downloaded folders (each holding part of the dataset) 
        and the participant folders within them. Removes all MRI data and moves 
        participant folders to one dataset folder (specified in self.datasetPath).             
        """
        self.participantCount = 0
        datasetFolder = self.datasetPath
        #### unzips folders in datasetfolder (after downloading earlier)
        self._unzipAndRenameInFolder(datasetFolder)
        ## rearranges and randomizes (on a subject level)
        self._arrangeFolders(datasetFolder)
        self._randomizeSubjectData(datasetFolder)

    
    def _unzipAndRenameInFolder(self, folder, remove=False):
        """
        Unzips all zip files in the specified folder and renames the extracted folders.

        This method checks if the folder has already been unzipped by looking for a marker file.
        If not, it unzips all zip files in the folder, optionally removes the zip files after extraction,
        and renames the extracted folders to a standardized format.

        Parameters:
        folder (str): The path to the folder containing zip files to be unzipped.
        remove (bool): If True, the zip files will be deleted after extraction. Defaults to False.

        Returns:
        None
        """
        unzipped_marker = os.path.join(folder, '.unzipped')
        if os.path.exists(unzipped_marker):
            self.logger.info(f"Folder {folder} is already unzipped. Exiting early.")
            return

        assert all(f.endswith('.zip') or f.startswith('.') for f in os.listdir(folder)), f"Not all files in {folder} are zip files or ignored files, please delete non-zip files and re-run."
        zip_file_count = sum(1 for f in os.listdir(folder) if f.endswith('.zip'))
        self.logger.info(f"Unzipping and renaming {zip_file_count} files in folder: {folder}")
        for zipFile in os.listdir(folder):
            ## unzip
            zipFile = os.path.join(folder, zipFile)
            if zipFile.endswith('.zip'):
                self.logger.info(f"Unzipping {zipFile}...")
                try:
                    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(zipFile))
                    if remove:
                        os.remove(zipFile)  # Remove the zip file after extraction
                        self.logger.info(f"Unzipped and removed {zipFile}")
                    else:
                        self.logger.info(f"Unzipped {zipFile}")
                except Exception as e:
                    self.logger.error(f"Error unzipping {zipFile}, {e}")
                    traceback.print_exc()
        for i, unzippedFolder in enumerate(f for f in os.listdir(folder) if not f.endswith('.zip') and not f.startswith('.') and os.path.isdir(os.path.join(folder, f))):
            self.logger.debug(f"{i}, {unzippedFolder}")
            fromName = os.path.join(folder, unzippedFolder)
            toName = os.path.join(folder, f'folder_{i}')
            os.rename(fromName, toName)

        # Create the .unzipped marker file
        with open(unzipped_marker, 'w') as marker_file:
            marker_file.write('')


    def _arrangeFolders(self, datasetFolder):
        """
        Rearranges the files in the datasetFolder after downloading and unzipping.
        
        This method organizes the dataset by moving participant folders into the main dataset folder,
        ensuring that relevant files are not nested within subdirectories. It prepares the dataset
        for randomization and later access by ensuring a consistent folder structure.

        Args:
            datasetFolder (str): The path to the main dataset folder where the unzipped files are located.
        
        Raises:
            AssertionError: If the number of participant folders does not match the expected count.
        """
        arranged_marker = os.path.join(datasetFolder, '.arranged')
        if os.path.exists(arranged_marker):
            self.logger.info(f"Dataset folder {datasetFolder} is already arranged. Exiting early.")
            return

        self.participantCount = 0
        unzippedFolders = [f for f in os.listdir(datasetFolder) if os.path.isdir(os.path.join(datasetFolder, f))]
        self.logger.debug(f"Unzipped folders found: \n {unzippedFolders} \nRearranging.")
        
        for folder in unzippedFolders:
            ### move participant folders into dataset folder (so they are not nested)
            self.logger.debug(f"Moving subjects in {folder} to parent ({datasetFolder}) and deleting.")
            desiredSubFolder = os.path.join(folder, 'derivatives', 'meg_derivatives')
            self.logger.debug(f"sub:{desiredSubFolder},  parent:{datasetFolder}")
            DatasetDownloader.moveContentsToParentAndDeleteSub(datasetFolder, desiredSubFolder,folderOnly = True)
        
        ### go through participant folders and move nested relevant files upwards to participant folder
        participantFolders = [f for f in os.listdir(datasetFolder) if not f.startswith('.') and not f.endswith('.zip')]
        self.logger.debug(f"Participant folders found: \n {participantFolders} \nRearranging.")
        for participantFolder in participantFolders:
            self.participantCount += 1
            desiredSubFolder = os.path.join('ses-meg', 'meg')
            participantFolderPath = os.path.join(datasetFolder,participantFolder)
            self.logger.debug(f"Moving data in {participantFolder} to parent ({datasetFolder}) and deleting.")
            self.logger.debug(f"sub:{desiredSubFolder},  parent:{datasetFolder}")
            DatasetDownloader.moveContentsToParentAndDeleteSub(participantFolderPath,desiredSubFolder)

        ### go through participant folders again and renames fif files to standardized format
        participantFolders = [f for f in os.listdir(datasetFolder) if not f.startswith('.') and not f.endswith('.zip')]
        self.logger.debug(f"Participant folders: \n{participantFolders}")
        for participantFolder in participantFolders: 
            participantFolderPath = os.path.join(datasetFolder,participantFolder)
            for file in os.listdir(participantFolderPath):
                match = re.match(r'.*run-(0[1-6]).*', file)
                if match:
                    new_file_name = f'run_{match.group(1)}' + os.path.splitext(file)[1]
                    os.rename(os.path.join(participantFolderPath, file), os.path.join(participantFolderPath, new_file_name))

        ### at this stage dataset folder should contain n folders (with n being number of participants)
        non_zip_non_dot_folders = [f for f in os.listdir(datasetFolder) if not f.startswith('.') and not f.endswith('.zip')]
        assert len(non_zip_non_dot_folders) == self.participantCount, f"ERROR: Dataset folder contains {len(non_zip_non_dot_folders)} folders, but {self.participantCount} (participant) folders are expected."

        # Create the .arranged marker file
        with open(arranged_marker, 'w') as marker_file:
            marker_file.write('')

            
    def _randomizeSubjectData(self,datasetFolder,train_percentage=70, val_percentage=20):
        """
        Randomizes and splits participant data into training, validation, and test sets.

        This method shuffles the participant folders and divides them into three subsets:
        training, validation, and test, based on the specified percentages. The folders
        are then moved into corresponding subdirectories within the dataset path.

        Args:
            train_percentage (int): The percentage of participant data to allocate to the training set.
            val_percentage (int): The percentage of participant data to allocate to the validation set.

        Raises:
            ValueError: If the sum of train_percentage and val_percentage exceeds 100.
        """
        randomized_marker = os.path.join(datasetFolder, '.randomized')

        if not os.path.exists(randomized_marker):
            ### split participant folders into train, test, and val subfolders        
            participantFolders = [f for f in os.listdir(datasetFolder) if os.path.isdir(os.path.join(datasetFolder,f))]
            
            random.seed(42)
            random.shuffle(participantFolders)

            total_participants = len(participantFolders)
            train_percentage = int(train_percentage)
            val_percentage = int(val_percentage)
            train_count = int(total_participants * train_percentage / 100)
            val_count = int(total_participants * val_percentage / 100)

            train_folder = os.path.join(datasetFolder, 'train')
            val_folder = os.path.join(datasetFolder, 'val')
            test_folder = os.path.join(datasetFolder, 'test')

            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)

            for i, participantFolder in enumerate(f for f in participantFolders):
                participantFolderPath = os.path.join(datasetFolder, participantFolder)
                if i < train_count:
                    shutil.move(participantFolderPath, train_folder)
                elif i < train_count + val_count:
                    shutil.move(participantFolderPath, val_folder)
                else:
                    shutil.move(participantFolderPath, test_folder)

            # Create the .randomized marker file
            with open(randomized_marker, 'w') as marker_file:
                marker_file.write('')
        else:
            self.logger.info("Dataset has already been randomized. Skipping randomization step.")


    @staticmethod
    def moveContentsToParentAndDeleteSub(parentfolder, intermediateFolders, expectedContentCount = None, folderOnly = False):
        """
        Moves contents from intermediate folder to parent folder and deletes the intermediate folders.
        E.g.
            parentFolder: "data/openfmri"
            intermediateFolders: "folder_0/derivatives/meg_derivatives" (containing subject folders)
            --> moves all subject folders to parent folder
            
        Parameters:
        subfolder (str): The name (relative path) of the subfolder whose contents are to be moved.
        parentfolder (str): The absolute path to the parent folder where contents will be moved.

        Returns:
        None
        """
        intermediateFolderPath = os.path.join(parentfolder, intermediateFolders)
        if expectedContentCount is not None:
            actualContentCount = len(os.listdir(intermediateFolderPath))
            assert actualContentCount == expectedContentCount, (
                f"Expected {expectedContentCount} items, but found {actualContentCount} in {intermediateFolderPath}"
            )
        # Move contents of subfolder to parentfolder
        for item in os.listdir(intermediateFolderPath):
            item_path = os.path.join(intermediateFolderPath, item)
            if (folderOnly and os.path.isdir(item_path)) or not folderOnly:
                shutil.move(item_path, parentfolder)

        # Remove the folder out of which the contents were moved upwards
        normalized_path = os.path.normpath(intermediateFolders)
        outermost_folder = normalized_path.split(os.sep)[0]
        outermost_folder_path = os.path.join(parentfolder,outermost_folder)
        
        shutil.rmtree(outermost_folder_path)


class DatasetPreprocesser():
    def __init__(self,**kwargs):
        self.datasetPath = kwargs.get('datasetPath', os.path.join('/srv','synaptech_openfmri'))

        self.logger = kwargs.get('logger', None)
        self.verbose = kwargs.get('verbose', False)

        if self.logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.processImmediately = kwargs.get('processImmediately', True)
        self.mode = str(kwargs.get('mode', "raw")).lower()


        self.windowLength_ms = kwargs.get('windowLength', 250)
        self.samplingRate_hz = kwargs.get('samplingRate', 1100)

        self.windowLength = int(self.windowLength_ms * self.samplingRate_hz / 1000)

        self._checkDatasetIntegrity(self.datasetPath)

        if self.processImmediately:
            self.process()
    
    def _checkDatasetIntegrity(self, datasetPath):
        """
        Checks the integrity of the dataset by ensuring that the dataset folder
        contains only directories for subjects and that each subject directory
        contains only files with the extensions '.txt' or '.fif'.

        Parameters:
        - datasetPath (str): The path to the dataset directory to be checked.

        Returns:
        - bool: True if the dataset integrity is confirmed, raises an assertion error otherwise.
        """
        ### iterate through mode folders (train, test, val)
        for modeFolder in os.listdir(datasetPath):
            if ".zip" in modeFolder or modeFolder[0] == ".":
                continue
            assert os.path.isdir(os.path.join(datasetPath,modeFolder)), f"Dataset folder contains unexpected file: {modeFolder}"
            assert modeFolder == "train" or modeFolder == "val" or modeFolder == "test", f"Dataset folder contains unexpected folder: {modeFolder} (expected train, test and val)"
            ### iterate through subject folders in dataset folder
            for subjectFolder in os.listdir(os.path.join(datasetPath, modeFolder)):
                assert os.path.isdir(os.path.join(datasetPath,modeFolder,subjectFolder)), f"Dataset folder contains unexpected file: {modeFolder}/{subjectFolder}"
                assert len(os.listdir(os.path.join(datasetPath,modeFolder,subjectFolder)))>0, f"Subject folder {modeFolder}/{subjectFolder} unexpectedly empty"
                ### iterate through run files in subject folders
                for file in os.listdir(os.path.join(datasetPath,modeFolder,subjectFolder)):
                    assert ".txt" in file or ".fif" in file, f"Unexpected file {file} in folder {modeFolder}/{subjectFolder}"
        return True
    
    def process(self):
        """
        Starts processing of .fif files in dataset folder based on self.mode (default: 'raw').

        1. Turns data into windows
        2. 
        3. 

        """
        pass


    def identifyChannels(self):
        pass 


    def makeWindows(self):
        pass

class DataConverter:
    def __init__(self, source_dataset_path, target_dataset_path, hdf5_filename, **kwargs):
        """
        Converts the OpenFMRI dataset from FIF files to HDF5 format.

        Parameters:
        - source_dataset_path (str): Path to the existing dataset (e.g., '/srv/synaptech_openfmri').
        - target_dataset_path (str): Path to save the HDF5 file (e.g., '/srv/synaptech_openfmri_new').
        - hdf5_filename (str): Name of the HDF5 file to create (e.g., 'openfmri_hybrid.h5').
        - sample_rate (int): Sampling rate of the data (default: 1100 Hz).
        - window_length (int): Length of each window in milliseconds (default: 250 ms).
        - window_overlap (float): Overlap between windows (default: 0.3).
        - logger (logging.Logger): Logger instance.
        - verbose (bool): Verbosity flag.
        """
        self.source_dataset_path = source_dataset_path
        self.target_dataset_path = target_dataset_path
        self.hdf5_filepath = os.path.join(self.target_dataset_path, hdf5_filename)
        self.sample_rate = kwargs.get('sample_rate', 1100)
        self.window_length_ms = kwargs.get('window_length', 250)
        self.window_overlap = kwargs.get('window_overlap', 0.3)
        self.logger = kwargs.get('logger', None)
        self.verbose = kwargs.get('verbose', False)

        if self.logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.window_length_frames = int(self.window_length_ms / 1000 * self.sample_rate)

        # Define EEG and MEG indices (assuming they are same across files)
        self.meg_indices = list(range(0, 306))
        self.eeg_indices = list(range(306, 380))

        # Initialize data structures
        self._participantRunsDict = {}
        self._participantFrameCounts = {}
        self._participantWindowCounts = {}
        self._participantWindowIndices = {}

        # Create target directory if it doesn't exist
        os.makedirs(self.target_dataset_path, exist_ok=True)

    def convert(self):
        """
        Converts the dataset from FIF files to HDF5 format.
        """
        modes = ['train', 'val', 'test']
        with h5py.File(self.hdf5_filepath, 'w') as h5f:
            # Add attributes
            h5f.attrs['sample_rate'] = self.sample_rate
            h5f.attrs['window_length'] = self.window_length_ms
            h5f.attrs['window_overlap'] = self.window_overlap

            for mode in modes:
                self.logger.info(f"Processing mode: {mode}")
                mode_group = h5f.create_group(f'synaptech_openfmri/{mode}')
                windows_group = h5f.create_group(f'windows/{mode}')

                mode_path = os.path.join(self.source_dataset_path, mode)
                subjects = [d for d in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, d))]
                self._participantRunsDict[mode] = {}

                # Prepare window indices list
                window_indices_list = []

                for subject_idx, subject in enumerate(subjects):
                    subject_path = os.path.join(mode_path, subject)
                    runs = sorted([f for f in os.listdir(subject_path) if f.endswith('.fif')])

                    self._participantRunsDict[mode][subject] = runs

                    for run_idx, run_file in enumerate(runs):
                        run_name = run_file.replace('.fif', '')
                        self.logger.info(f"Processing {mode}/{subject}/{run_name}")

                        fif_file_path = os.path.join(subject_path, run_file)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                                raw = mne.io.read_raw_fif(fif_file_path, preload=True)

                        # Get data
                        data = raw.get_data()
                        meg_data = data[self.meg_indices, :]
                        eeg_data = data[self.eeg_indices, :]

                        # Get EMOV
                        emov = self.find_emov(raw.info['chs'], raw.info['dev_head_t'])

                        # Create group for run
                        run_group_path = f'synaptech_openfmri/{mode}/{subject}_{run_name}'
                        run_group = h5f.create_group(run_group_path)

                        # Create datasets for MEG, EEG, EMOV
                        meg_dataset = run_group.create_dataset(
                            'meg',
                            data=meg_data,
                            chunks=(306, self.window_length_frames),
                            compression=None
                        )
                        eeg_dataset = run_group.create_dataset(
                            'eeg',
                            data=eeg_data,
                            chunks=(74, self.window_length_frames),
                            compression=None
                        )
                        emov_dataset = run_group.create_dataset('emov', data=emov)

                        # Calculate window indices
                        n_samples = data.shape[1]
                        step_size = int(self.window_length_frames * (1 - self.window_overlap))
                        run_window_indices = []
                        for start_idx in range(0, n_samples - self.window_length_frames + 1, step_size):
                            window_indices_list.append((subject_idx, run_idx, start_idx))
                            run_window_indices.append(start_idx)

                        # Store window counts and indices
                        self._participantFrameCounts[(mode, subject, run_name)] = n_samples
                        self._participantWindowCounts[(mode, subject, run_name)] = len(run_window_indices)
                        self._participantWindowIndices[(mode, subject, run_name)] = run_window_indices

                # Save window indices for the mode
                indices_dataset = windows_group.create_dataset(
                    'indices',
                    data=np.array(window_indices_list, dtype=np.int64)
                )

    @staticmethod
    def find_emov(channel_locs, dev_head_t):
        """
        Calculate the EEG-MEG Offset Vector (EMOV).

        Parameters:
        - channel_locs: List of channel info (e.g., raw.info['chs']).
        - dev_head_t: Transformation matrix from device to head coordinates.

        Returns:
        - emov (np.ndarray): The EMOV vector (3-element array).
        """
        eeg_reference_electrode_index = 379  # Index of the reference EEG electrode
        meg_reference_electrode_index = 236  # Index of the reference MEG electrode

        # Get EEG electrode location
        eeg_loc = channel_locs[eeg_reference_electrode_index]['loc'][:3]

        # Get MEG sensor location
        meg_loc = channel_locs[meg_reference_electrode_index]['loc'][:3]

        # Apply device-to-head transformation to MEG location
        meg_loc_transformed = mne.transforms.apply_trans(dev_head_t, meg_loc)

        # Calculate EMOV
        emov = np.array(eeg_loc) - np.array(meg_loc_transformed)

        return emov