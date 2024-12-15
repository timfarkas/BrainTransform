import os
import numpy as np
import logging
import h5py
import torch
from torch.utils.data import Dataset

class OpenFMRIDataSet(Dataset):
    """
    A PyTorch Dataset class for the OpenFMRI dataset stored in HDF5 format.
    """

    def __init__(self, datasetPath, mode='train', windowLength_ms=250, windowOverlap=0.3, logger=None):
        """
        Initialize the OpenFMRIDataSet.

        Parameters:
        - datasetPath (str): Path to the HDF5 dataset file.
        - mode (str): One of 'train', 'val', 'test'.
        - windowLength (int): Length of each window in milliseconds.
        - windowOverlap (float): Overlap between windows as a fraction.
        - logger (logging.Logger): Logger instance.
        """
        self.datasetPath = datasetPath
        self.mode = mode
        self.windowLength_ms = windowLength_ms  # renamed to be explicit
        self.windowOverlap = windowOverlap

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        # Open the HDF5 file once
        self.h5f = h5py.File(self.datasetPath, 'r')

        self._load_dataset_attributes()
        self._prepare_window_indices()

        self.eegValuesType = np.float32
        self.megValuesType = np.float32

    def _load_dataset_attributes(self):
        """
        Load dataset attributes from the HDF5 file.
        """
        # Use the already opened HDF5 file
        h5f = self.h5f
        self.sample_rate = h5f.attrs['sample_rate']
        self.windowLength_frames = int(self.windowLength_ms * self.sample_rate / 1000)
        self.windowStride_frames = int(self.windowLength_frames * (1 - self.windowOverlap))
        self.logger.info(f"Sample Rate: {self.sample_rate}")
        self.logger.info(f"Window Length (frames): {self.windowLength_frames}")
        self.logger.info(f"Window Overlap: {self.windowOverlap}")
        self.logger.info(f"Window Stride (frames): {self.windowStride_frames}")

    def _prepare_window_indices(self):
        """
        Prepare the list of window indices for the dataset.
        """
        self.window_indices = []

        # Use the already opened HDF5 file
        h5f = self.h5f
        indices_dataset = h5f[f'windows/{self.mode}/indices']
        self.n_windows = indices_dataset.shape[0]
        self.logger.info(f"Total windows in {self.mode} mode: {self.n_windows}")

        # Load all window indices into memory
        self.window_indices = indices_dataset[:]

        # Build mappings from subject and run indices to names
        self.subject_run_map = {}
        synaptech_group = h5f[f'synaptech_openfmri/{self.mode}']
        for idx, subject_run in enumerate(synaptech_group.keys()):
            self.subject_run_map[idx] = subject_run

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):


        import time
        start_time = time.time()


        # Use the already opened HDF5 file
        h5f = self.h5f
        subject_run_idx, _, start_idx = self.window_indices[idx]
        subject_run_name = self.subject_run_map[subject_run_idx]
        loc_time = time.time()


        # Ensure correct path format
        data_path = f'synaptech_openfmri/{self.mode}/{subject_run_name}'

        try:
            # First verify the group exists
            if data_path not in h5f:
                raise KeyError(f"Path {data_path} not found in HDF5 file")

            eeg_dataset = h5f[f'{data_path}/eeg']
            meg_dataset = h5f[f'{data_path}/meg']
            emov_dataset = h5f[f'{data_path}/emov']
            read_time = time.time()


            # Calculate end index and validate bounds
            dataset_length = eeg_dataset.shape[1]
            end_idx = min(start_idx + self.windowLength_frames, dataset_length)

            # Adjust start_idx if window would exceed dataset bounds
            if end_idx - start_idx < self.windowLength_frames:
                start_idx = max(0, dataset_length - self.windowLength_frames)
                end_idx = start_idx + self.windowLength_frames

            # Extract data windows
            eeg_data = eeg_dataset[:, start_idx:end_idx]
            # Only select magnetometer channels (every 3rd channel starting from index 2)
            meg_data = meg_dataset[2::3, start_idx:end_idx]
            emov = emov_dataset[:]

            # Verify data is not empty
            if eeg_data.size == 0 or meg_data.size == 0:
                raise ValueError(f"Empty data encountered at idx {idx}. start_idx: {start_idx}, end_idx: {end_idx}, dataset_length: {dataset_length}")

            # Normalize EEG and MEG data
            eeg_mean = np.mean(eeg_data, axis=1, keepdims=True)
            eeg_std = np.std(eeg_data, axis=1, keepdims=True)
            meg_mean = np.mean(meg_data, axis=1, keepdims=True)
            meg_std = np.std(meg_data, axis=1, keepdims=True)

            # Add small epsilon to prevent division by zero
            eeg_data = (eeg_data - eeg_mean) / (eeg_std + 1e-8)
            meg_data = (meg_data - meg_mean) / (meg_std + 1e-8)

            # Append EMOV rows to EEG data
            emov_rows = np.tile(emov[:, np.newaxis], (1, self.windowLength_frames))
            eeg_data_with_emov = np.vstack([eeg_data, emov_rows])

            process_time = time.time()
            if idx % 100 == 0:
                self.logger.info(f"Window {idx} timing:")
                self.logger.info(f"  Location lookup: {(loc_time - start_time)*1000:.2f}ms")
                self.logger.info(f"  Data read: {(read_time - loc_time)*1000:.2f}ms")
                self.logger.info(f"  Processing: {(process_time - read_time)*1000:.2f}ms")

            return eeg_data_with_emov.astype(self.eegValuesType), meg_data.astype(self.megValuesType)

        except Exception as e:
            self.logger.error(f"Error accessing data for idx {idx}: {str(e)}")
            self.logger.error(f"subject_run_idx: {subject_run_idx}")
            self.logger.error(f"subject_run_name: {subject_run_name}")
            self.logger.error(f"data_path: {data_path}")
            self.logger.error(f"start_idx: {start_idx}")
            raise e  # Re-raise the exception after logging

    def __del__(self):
        # Close the HDF5 file when the dataset object is destroyed
        if hasattr(self, 'h5f') and self.h5f is not None:
            self.h5f.close()