# Enhancing EEG Accuracy via cross modal MEG inference.
A transformer model trained on simultaneous EEG and MEG recordings to achieve near-perfect MEG-like accuracy on EEG headsets.

[]Demo[]

## Overview 🔎
MEG and EEG share not only temporal resolution, but also similar electrical signal features. This makes them well-suited for transformer models, which can accurately learn their dependancies and predict MEG signals from EEG data, thereby enhancing the performance and precision of EEG-based BCIs.
This code aims to provide developers and researchers a tool to train their own transformer to enhance real world EEG accuracy.

---

## Set-up 🔧
Create anaconda environment
```
conda create -n synaptech_env python=3.10 -y 
conda activate synaptech_env
```

Install dependencies
```
pip install -r requirements.txt
```

Set up .env to point to dataset
```
DATASET_PATH="openfrmi/data"
```

Start your first training run!
```
python training.py
```
(first run init will take some time as it will automatically load, prepare, and preprocess the dataset)

Train the model: 
## Run the model 💥🏃‍♂️🔥
To use the application:
1. Run ```main.py``` to feed some validation data into the transformer and see the model at work..
2. Enjoy :) Go save the world!


---

## Technical Overview 
EEG Channel Count: 74
MEG Channel Count: 306
Frame Count: 6042300 (in test dataset)

---

### Dataset Explained

The .fif (Functional Imaging Format) file is a format specific to MNE-Python and Elekta Neuromag systems designed to store MEG and EEG data along with all relevant metadata.


It measures:
- Raw time series data from sensores
- Channel Information: Names, types, locations, orientations, and other metadata for each channel.
- Measurement Information: Sampling frequency, filter settings, recording device information, etc.
- Event Information: Markers indicating stimuli or responses during the recording.
- Annotations: Labels for specific time intervals, such as bad segments or experimental conditions.
- Additional Metadata: Subject information, head position data, and more.


1.  Channel information: 
Channel number: 400 (74 EEG) (306 MEG) (20 the rest of MEG)
Channel types: {'grad': 204, 'mag': 102, 'eeg': 74, 'misc': 12, 'chpi': 9, 'stim': 3}

EEG (electro encephelogram):
- Number represent the electrode's position according to a standard EEG montage (e.g., 10-20 system).

MEG: 
    MAG (magnetometer):
    - MEG measure the absolute strength of the magnetic field at a single point in space.

    GRAD (gradiometer):
    - measure the spatial gradient (difference) of the magnetic field between two closely spaced points.

STIM (stimuluts channel):
- Stimulus channels, used to record event markers or triggers during the experiment.
- e.g: STI101, STI201 Indicates that these are stimulus channels, records stimuli / events

MISC (miscellaneous channels):
- miscellaneous channels, used for custom measurements.

CHPI(Continuous Head Position Indicator):
- Used to track head movement during the MEG recording.




.fif files contain:
Opening raw data file data/openfmri/train/sub-05/run_01.fif...
    Range : 141900 ... 681999 =    129.000 ...   619.999 secs
Ready.
<Raw | run_01.fif, 404 x 540100 (491.0 s), ~7.0 MB, data not loaded>
Channel names: ['MEG0113', 'MEG0112', 'MEG0111', 'MEG0122', 'MEG0123', 'MEG0121', 'MEG0132', 'MEG0133', 'MEG0131', 'MEG0143', 'MEG0142', 'MEG0141', 'MEG0213', 'MEG0212', 'MEG0211', 
...
'EEG015', 'EEG016', 'EEG017', 'EEG018', 'EEG019', 'EEG020', 'EEG021', 'EEG022', 'EEG023', 'EEG024', 'EEG025', 'EEG026', 'EEG027', 'EEG028', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 
...
'MISC303', 'MISC304', 'MISC305', 'MISC306', 'CHPI001', 'CHPI002', 'CHPI003', 'CHPI004', 'CHPI005', 'CHPI006', 'CHPI007', 'CHPI008', 'CHPI009']
