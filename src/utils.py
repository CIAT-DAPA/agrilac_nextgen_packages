import csv

def csv_to_dict(filename):
    data_dict = {}
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 2:
                key, value = row
                data_dict[key] = value 
    return data_dict

def process_params(data_dict):
    # Convertir start_year, end_year, start_step, last_step, eof_x, eof_y, cca_modos, forecast_year a enteros o flotantes
    int_keys = ['start_year', 'end_year', 'eof_x', 'eof_y', 'cca_modos', 'forecast_year']
    float_keys = ['start_step', 'last_step']

    for key in int_keys:
        data_dict[key] = int(data_dict[key])
    
    for key in float_keys:
        data_dict[key] = float(data_dict[key])
    
    # Asegurarse de que models sea una lista
    data_dict['models'] = data_dict['models'].split(',')

    return data_dict