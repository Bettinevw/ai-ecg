###############################################################################
# Author: b.g.vanwilligen@amsterdamumc.nl
#
# Couple all ECGs of MUSE to a patient
# 1. PIN should be at least 9 digits, otherwise zeros are added in front
# 2. Letter o is replaced with number 0
# 3. PINS with letters are excluded
# 4. Filter data that contains a lead of all zeros
###############################################################################

import os
import re
import xmltodict
import base64
import pickle
import random
import numpy as np
from copy import copy
from datetime import datetime


def _detect_xml_encoding(file_path, default="ISO-8859-1"):
    """Detect encoding from the XML header line; fallback to default."""
    try:
        with open(file_path, "rb") as fr:
            first_line = fr.readline()
        match = re.search(br'encoding="([^"]+)"', first_line)
        return match.group(1).decode("ascii") if match else default
    except Exception:
        return default


def _clean_pin(patient_id):
    """Zero-pad to 9 chars, replace o/O with 0, and return PIN or None if invalid."""
    pin = str(patient_id).zfill(9)
    pin = pin.replace("o", "0").replace("O", "0")
    return pin if pin.isdigit() else None


def _decode_wave(lead_xml):
    
    LeadAmplitudeUnitsPerBit = float(lead_xml['LeadAmplitudeUnitsPerBit'])
    
    # Read the b64 data
    b64_encoded = ''.join(lead_xml['WaveFormData'].split('\n'))
    # Decode it from b64
    decoded = base64.b64decode(b64_encoded)
    # Convert decoded bytes to int
    signal = np.frombuffer(decoded, dtype='int16')
    # Fix amplitude    
    signal = signal*LeadAmplitudeUnitsPerBit
    
    return signal


def filter_data(xml_dict, PIN, date, maxabs_thr=10000):
    """
    Keep only ECGs where:
      - Waveform + LeadData exist
      - No NaNs in any lead
      - No abs(value) > maxabs_thr in any lead

    Returns:
      keep (bool)
    """

    # Check if waveform exists
    if not ('RestingECG' in xml_dict and 'Waveform' in xml_dict['RestingECG']):
        print(PIN, date, ": Date removed, waveform is not there")
        return False

    mean_wave, leads = xml_dict['RestingECG']['Waveform']

    # Check if LeadData exists
    if 'LeadData' not in leads:
        print(PIN, date, ": Date removed, LeadData is not there")
        return False

    lead_list = leads['LeadData'][:]  # list-like
    L = len(lead_list)
    if L == 0:
        print(PIN, date, ": Date removed, LeadData empty")
        return False

    # Loop through leads and reject immediately if anything bad is found
    for lead in range(L):
        wave = _decode_wave(lead_list[lead])

        # NaN check
        if np.isnan(wave).any():
            print(PIN, date, f": Date removed, NaN in lead {lead}")
            return False

        # Too-high check
        if np.max(np.abs(wave)) > maxabs_thr:
            print(PIN, date, f": Date removed, abs(wave) > {maxabs_thr} in lead {lead}")
            return False
        
        # All-zero lead check
        if np.all(wave == 0):
            print(PIN, date, f": Date removed, lead {lead} is all zeros")
            return False

    return True


def save_file_paths(xml_folders, pkl_file, save_every_folder=True, progress_every=100):
    """
    Builds/updates a dict: patient_file_paths[PIN][date_time] = {"path": file_path}

    Only keeps records where:
      - XML parses
      - PIN is valid digits (after cleaning)
      - filter_data(xml_dict, PIN, date) returns True
      - file endswith .xml

    Parameters
    ----------
    xml_folders : list[str]
        List of folders containing xml files. Assumes filenames can be joined with os.path.join.
    pkl_file : str
        Output pickle file storing the dict (updated incrementally).
    save_every_folder : bool
        Save pickle after each folder (safer for long runs).
    progress_every : int
        Print progress every N files.

    Returns
    -------
    patient_file_paths : dict
    """
    # Load or init
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            patient_file_paths = pickle.load(f)
    else:
        patient_file_paths = {}

    # Loop folders
    for folder in xml_folders:
        print("Folder:", folder)

        try:
            files = os.listdir(folder)
        except Exception as e:
            print("Could not list folder:", folder, "|", e)
            continue

        n_files = len(files)

        for idx, fname in enumerate(files):
            if progress_every and (idx % progress_every == 0):
                print(idx, "of", n_files)

            if not fname.lower().endswith(".xml"):
                # keep noise down: comment out if you really want prints
                # print("Skipped (not xml):", fname)
                continue

            file_path = os.path.join(folder, fname)

            # Parse XML
            try:
                encoding = _detect_xml_encoding(file_path)
                with open(file_path, encoding=encoding) as fd:
                    xml_dict = xmltodict.parse(fd.read(), process_namespaces=True)
            except Exception as e:
                print("Data excluded:", fname, "| XML parse/read failed:", e)
                continue

            # Extract metadata
            try:
                patient_id = xml_dict["RestingECG"]["PatientDemographics"]["PatientID"]
                date_str = xml_dict["RestingECG"]["TestDemographics"]["AcquisitionDate"]
                time_str = xml_dict["RestingECG"]["TestDemographics"]["AcquisitionTime"]

                date = datetime.strptime(date_str, "%m-%d-%Y").strftime("%Y-%m-%d")
                date_time = f"{date} {time_str}"
            except Exception as e:
                print("Data excluded:", fname, "| missing required fields:", e)
                continue

            # Clean PIN
            PIN = _clean_pin(patient_id)
            if PIN is None:
                print("Data excluded:", patient_id, "| PIN not digits after cleaning")
                continue

            # Filter (your strict rule: reject if NaN or >10000 or all-zero lead etc.)
            try:
                keep = filter_data(xml_dict, PIN, date)
            except Exception as e:
                print("Data excluded:", PIN, date, "| filter_data error:", e)
                continue

            if not keep:
                # keep noise down: filter_data already prints the reason
                continue

            # Store
            patient_file_paths.setdefault(PIN, {})
            patient_file_paths[PIN][date_time] = {"path": file_path}

        # Save incrementally (safer)
        if save_every_folder:
            try:
                with open(pkl_file, "wb") as f:
                    pickle.dump(patient_file_paths, f)
            except Exception as e:
                print("Warning: could not save pickle after folder:", folder, "|", e)

    # Final save
    try:
        with open(pkl_file, "wb") as f:
            pickle.dump(patient_file_paths, f)
    except Exception as e:
        print("Warning: could not save final pickle:", e)

    return patient_file_paths


if __name__=='__main__':
    # Folders
    main_folder     = "G:/diva1/Research/MUSE ECGs/"

    xml_folders     = [main_folder+"00_01_02_03/", 
                        main_folder+"04_05_06_07/",
                        main_folder+"12_13_14/",
                        main_folder+"15_16_17/",
                        main_folder+"2018/",
                        main_folder+"tm_1999/"]
    
    destination_folder  = main_folder+"PROJECTS/Bettine/Data/"
    pkl_file            = destination_folder + 'patient_file_paths.pkl'
    
    patient_file_paths = save_file_paths(xml_folders, pkl_file)
    
    
