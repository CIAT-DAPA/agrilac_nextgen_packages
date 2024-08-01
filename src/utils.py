import csv
import datetime
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
    int_keys = ['eof_x', 'eof_y', 'cca_modos']

    for key in int_keys:
        data_dict[key] = int(data_dict[key])
    
   
    # Asegurarse de que models sea una lista
    data_dict['models'] = data_dict['models'].split(',')

    return data_dict

def train_step_one():
    last_step = 3.5
    start_step = 1.5
    return last_step, start_step

def train_step_two():
    last_step = 6.5
    start_step = 4.5
    return last_step, start_step

def test_step_one():
    now = datetime.datetime.now()
    if now.day < 15:
        start_step = 2.5
        last_step = 4.5
    else:
        start_step = 1.5
        last_step = 3.5
    return last_step, start_step

def test_step_two():
    now = datetime.datetime.now()
    if now.day < 15:
        start_step = 5.5
        last_step = 7.5
    else:
        start_step = 4.5
        last_step = 6.5
    return last_step, start_step