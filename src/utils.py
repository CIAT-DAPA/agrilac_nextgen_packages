import csv
import datetime
import pandas as pd
import os
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

def process_dataframe(path):
    df = pd.read_excel(path)
    
    # Define the names for the dictionaries
    names = ['modelos', 'modos_x', 'modos_y', 'modos_cca', 'lat_min', 'lat_max', 'lon_min', 'lon_max']
    
    # Function to handle conversion to list if needed
    def to_list_if_needed(value):
        if isinstance(value, str) and ',' in value:
            return value.split(',')
        return value
    
    # Function to get the current month in Spanish
    def get_current_month_in_spanish():
        months_translation = {
            'January': 'Enero',
            'February': 'Febrero',
            'March': 'Marzo',
            'April': 'Abril',
            'May': 'Mayo',
            'June': 'Junio',
            'July': 'Julio',
            'August': 'Agosto',
            'September': 'Septiembre',
            'October': 'Octubre',
            'November': 'Noviembre',
            'December': 'Diciembre'
        }

        # Get the current month in English
        current_month_english = datetime.datetime.now().strftime('%B')

        # Translate to Spanish
        current_month_spanish = months_translation[current_month_english]

        return current_month_spanish

    # Get the current month in Spanish
    current_month = get_current_month_in_spanish()

    # Create the first dictionary from the first 8 elements using the current month
    first_dict = {names[i]: to_list_if_needed(df.iloc[i][current_month]) for i in range(8)}
    first_dict['LT'] = 0

    # Create the second dictionary from the next 8 elements using the current month
    second_dict = {names[i]: to_list_if_needed(df.iloc[i+8][current_month]) for i in range(8)}
    second_dict['LT'] = 3

    # Retrieve 'username' and 'password' from the last two rows of the DataFrame
    username = df.iloc[-2][current_month]
    password = df.iloc[-1][current_month]

    # Add 'username' and 'password' to both dictionaries
    first_dict['username'] = username
    first_dict['password'] = password
    
    second_dict['username'] = username
    second_dict['password'] = password

    return first_dict, second_dict
def create_monthly_folders(base_path):
    month_suffixes = {
        'Enero': ['EFM', 'AMJ'],
        'Febrero': ['FMA', 'MJJ'],
        'Marzo': ['MAM', 'JJA'],
        'Abril': ['AMJ', 'JAS'],
        'Mayo': ['MJJ', 'ASO'],
        'Junio': ['JJA', 'SON'],
        'Julio': ['JAS', 'OND'],
        'Agosto': ['ASO', 'NDE'],
        'Septiembre': ['SON', 'DEF'],
        'Octubre': ['OND', 'EFM'],
        'Noviembre': ['NDE', 'FMA'],
        'Diciembre': ['DEF', 'MAM']
    }
    month_names = [
        'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
    ]
    
    now = datetime.datetime.now()
    current_month = month_names[now.month - 1]
    previous_month_index = (now.month - 2) % 12
    previous_month = month_names[previous_month_index]


    current_month_folder = os.path.join(base_path, current_month)
    os.makedirs(current_month_folder, exist_ok=True)

    suffixes = month_suffixes[current_month]


    folder_names = [f"{previous_month[:3]}_{suffix}" for suffix in suffixes]

    created_paths = []
    for folder_name in folder_names:
        folder_path = os.path.join(current_month_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        created_paths.append(folder_path)

    return created_paths


