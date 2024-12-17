import mne
import argparse

def get_channel_names(edf_path):
    """Load EEG file and retrieve channel names."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        channel_names = raw.ch_names
        print(f"\nLoaded EDF file: {edf_path}")
        print(f"Number of channels: {len(channel_names)}")
        print("Channel Names:")
        for idx, name in enumerate(channel_names):
            print(f"{idx + 1}: {name}")
    except Exception as e:
        print(f"Error loading EDF file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Retrieve channel names from an EEG EDF file.")
    parser.add_argument("edf_file", type=str, help="Path to the EDF file")
    args = parser.parse_args()
    
    get_channel_names(args.edf_file)

if __name__ == "__main__":
    main()
