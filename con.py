import mne
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load an EEG file (.edf format)
def load_eeg_data(edf_file):
    raw_data = mne.io.read_raw_edf(edf_file, preload=True)  # Ensure preload=True
    raw_data.filter(0.5, 50.)  # Adjust filter to include Delta (0.5-50 Hz range)
    return raw_data

# Function to get specific frequency bands
def bandpass_filter(raw_data, l_freq, h_freq):
    return raw_data.copy().filter(l_freq=l_freq, h_freq=h_freq)

# Function to extract band powers
def extract_band_powers(filtered_data):
    epochs = mne.make_fixed_length_epochs(filtered_data, duration=2.0, overlap=1.0)
    epochs.load_data()
    psds, freqs = mne.time_frequency.csd_multitaper(epochs, fmin=1, fmax=50)[:2]
    avg_psd = np.mean(psds, axis=0)
    return avg_psd

# Define frequency ranges for bands
frequency_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 7),
    'Alpha': (8, 12),
    'Beta': (13, 30),
    'Gamma': (31, 50)
}

# File path of the EEG .edf file (replace with your downloaded file path)
edf_file = r'C:\Codes\Python\S001R01.edf'

# Load the EEG data
raw_data = load_eeg_data(edf_file)

# Extract features (band powers for each frequency band)
X = []
for band, (l_freq, h_freq) in frequency_bands.items():
    filtered_data = bandpass_filter(raw_data, l_freq, h_freq)
    band_power = extract_band_powers(filtered_data)
    X.append(band_power)

# Reshape X to a proper format (features in rows)
X = np.array(X).T  # Transpose so that each row represents an EEG segment, and each column represents a frequency band

# Simulate labeled data (this would be replaced by your actual concentration labels)
# 0 = low concentration, 1 = high concentration
y = np.random.randint(0, 2, size=X.shape[0])  # Random labels for illustration

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier on the band powers
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of concentration prediction: {accuracy * 100:.2f}%")

# Example: Predict concentration on a new EEG sample
new_sample_band_powers = X_test[0].reshape(1, -1)  # Take one sample from the test set
concentration_prediction = clf.predict(new_sample_band_powers)
print(f"Predicted Concentration Level: {'High' if concentration_prediction[0] == 1 else 'Low'}")
