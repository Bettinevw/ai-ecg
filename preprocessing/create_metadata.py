###############################################################################
#
# Create meta dataset for data exploration, everything is saved except for the
# signals.
#
###############################################################################

import os
import pickle
import xmltodict
import numpy as np
import pandas as pd
import re

def create_metadata(data_file):
    # Open file with file paths
    with open(data_file, 'rb') as f:
        patient_file_paths = pickle.load(f)
    
    # Empty lists
    meta_data = []
    
    count = 0
    
    PINs = np.array(list(patient_file_paths.keys()))
    # Go through all files
    for PIN in PINs:        
        # print(PIN)
        
        if count%100 == 0:
            print(count, ' of ', len(patient_file_paths.keys()))

        for date in patient_file_paths[PIN].keys():
            # print(date)
            file_path = patient_file_paths[PIN][date]['path']
            
            # Obtain file encoding
            with open(file_path, "rb") as fr:
                first_line = fr.readline()
            match = re.search(br'encoding="([^"]+)"', first_line)
            if match:
                encoding = match.group(1).decode("ascii")
            else:
                encoding = "ISO-8859-1"  # 
            
            # XML to dict
            with open(file_path, encoding=encoding) as fd:
                xml_dict = xmltodict.parse(fd.read(), process_namespaces=True)

            row = {}
            
            row['PIN'] = PIN 
            row['MuseVersion'] = xml_dict['RestingECG']['MuseInfo']['MuseVersion']
            for demo in xml_dict['RestingECG']['PatientDemographics'].keys():
                row[demo] = xml_dict['RestingECG']['PatientDemographics'][demo]
            for test in xml_dict['RestingECG']['TestDemographics'].keys():
                row[test] = xml_dict['RestingECG']['TestDemographics'][test]
            for meas in xml_dict['RestingECG']['RestingECGMeasurements'].keys():
                row[meas] = xml_dict['RestingECG']['RestingECGMeasurements'][meas]
            
            if 'OriginalRestingECGMeasurements' in xml_dict['RestingECG']:
                for meas in xml_dict['RestingECG']['OriginalRestingECGMeasurements'].keys():
                    name = 'Original'+meas
                    row[name] = xml_dict['RestingECG']['OriginalRestingECGMeasurements'][meas]
                            
            if 'Diagnosis' in xml_dict['RestingECG']:
                for diag in xml_dict['RestingECG']['Diagnosis'].keys():
                    if diag == 'DiagnosisStatement':
                        statements = xml_dict['RestingECG']['Diagnosis'][diag]
                        if isinstance(statements, dict):
                            statements = [statements]
                        
                        diagnose_statements = ''
                        for state in range(len(statements)):
                            if not statements[state]['StmtText'] == None:
                                diagnose_statements = diagnose_statements + statements[state]['StmtText'] + ' '
                            if 'StmtFlag' in statements[state]:
                                if 'ENDSLINE' in statements[state]['StmtFlag']:
                                    diagnose_statements = diagnose_statements + '| '
                        row['DiagnosisStatement'] = diagnose_statements
                        
                        # print(row['DiagnosisStatement'])
                

            if 'OriginalDiagnosis' in xml_dict['RestingECG']:
                for diag in xml_dict['RestingECG']['OriginalDiagnosis'].keys():
                    if diag == 'DiagnosisStatement':
                        statements = xml_dict['RestingECG']['OriginalDiagnosis'][diag]
                        if isinstance(statements, dict):
                            statements = [statements]
                        
                        diagnose_statements = ''
                        for state in range(len(statements)):
                            if not statements[state]['StmtText'] == None:
                                diagnose_statements = diagnose_statements + statements[state]['StmtText'] + ' '
                            if 'StmtFlag' in statements[state]:
                                if 'ENDSLINE' in statements[state]['StmtFlag']:
                                    diagnose_statements = diagnose_statements + '| '
                        row['OriginalDiagnosisStatement'] = diagnose_statements
            
                        # print(row['OriginalDiagnosisStatement'])  
                        
            if 'QRSTimesTypes' in xml_dict['RestingECG']:
                for qrs in xml_dict['RestingECG']['QRSTimesTypes'].keys():
                    qrs_numbers = ''
                    qrs_types = ''
                    qrs_time = ''
                    
                    if qrs !='QRS':
                        row[qrs] = xml_dict['RestingECG']['QRSTimesTypes'][qrs]
                    else:    
                        qrs_info = xml_dict['RestingECG']['QRSTimesTypes'][qrs]
                        if isinstance(qrs_info, dict):
                            qrs_info = [qrs_info]
                        for i_qrs in range(len(qrs_info)):
                            if not (qrs_info[i_qrs]['Number'] == None or 
                                    qrs_info[i_qrs]['Type'] == None or 
                                    qrs_info[i_qrs]['Time'] == None):
                                qrs_numbers = qrs_numbers + ' | ' + qrs_info[i_qrs]['Number']
                                qrs_types = qrs_types + ' | ' + qrs_info[i_qrs]['Type']
                                qrs_time = qrs_time + ' | ' + qrs_info[i_qrs]['Time']

                        row['QRSNumber'] = qrs_numbers
                        row['QRSType'] = qrs_types
                        row['QRSTime'] = qrs_time
                        
            row['FilePath'] = file_path

            meta_data.append(row)
            
        count = count + 1
    
    return meta_data


if __name__=='__main__':
    # Main folder
    main_folder     = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    
    data_file       = main_folder+"patient_file_paths.pkl"
    meta_data       = create_metadata(data_file)

    df              = pd.DataFrame(meta_data)
    df.to_parquet(main_folder+"metadata.parquet", index=False)

    