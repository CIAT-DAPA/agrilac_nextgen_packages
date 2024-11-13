import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import xcast as xc
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import kendalltau
from scipy.linalg import svd
from tqdm import tqdm
import requests
import re
import os
import calendar
import datetime
import time
from datetime import timedelta
import math
import itertools
import cdsapi
from utils import train_step_one,train_step_two
def process_data(file_path, nrows_per_block):
    dates = pd.read_csv(file_path, sep='\t', skiprows=2, header=None, on_bad_lines='skip')
    years = dates[0].str.extract(r'cpt:T=(\d{4})')[0].dropna().values
    years = pd.to_numeric(years, errors='coerce')
    df = pd.read_csv(file_path, sep='\t', skiprows=3, header=None)
    latitudes = pd.to_numeric(df.iloc[1:nrows_per_block, 0], errors='coerce')
    longitudes = pd.to_numeric(df.iloc[0, 1:], errors='coerce')
    data_list = [pd.to_numeric(df.iloc[start:start + nrows_per_block, 1:].stack(), errors='coerce').unstack().values * 90
                 for start in range(1, len(years) * (nrows_per_block + 1), nrows_per_block + 1)]
    data_xarray = xr.DataArray(data_list, dims=['S', 'lat', 'lon'], coords={'S': years, 'lat': latitudes, 'lon': longitudes})
    data_xarray['lon'] = xr.where(data_xarray['lon'] > 180, data_xarray['lon'] - 360, data_xarray['lon'])
    return data_xarray.sortby(['lat', 'lon'])
saved_paths = []
def generate_nmme_url_hindcast(model, month, start_month, end_month, hindcast_start_year, hindcast_end_year):
    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/"
    hindcast = 'seasonal_nmme_hindcast_in_cpt_format'
    range_years = f'{start_month}-{end_month}'  
    url = f"{base_url}{hindcast}/{model}_precip_hcst_{month}ic_{range_years}_{hindcast_start_year}-{hindcast_end_year}.txt"
    return url

def generate_nmme_url_forecast(model, month, start_month, end_month, year_start, year_end):
    base_url = "https://ftp.cpc.ncep.noaa.gov/International/nmme/"
    forecast = 'seasonal_nmme_forecast_in_cpt_format'
    range_years = f'{start_month}-{end_month}' 
    url = f"{base_url}{forecast}/{model}_precip_fcst_{month}ic_{range_years}_{year_start}-{year_end}.txt"
    return url

def download_and_save_as_tsv(model, month, start_month, end_month, folder_path, year_start=None, year_end=None, hindcast=True, hindcast_start_year=1991, hindcast_end_year=2020):
    if hindcast:
        url = generate_nmme_url_hindcast(model, month, start_month, end_month, hindcast_start_year, hindcast_end_year)
        output_filename = f"hindcast_{month}_{calendar.month_abbr[start_month]}-{calendar.month_abbr[end_month]}.tsv"
    else:
        url = generate_nmme_url_forecast(model, month, start_month, end_month, year_start, year_end)
        output_filename = f"{year_start}_{month}_{calendar.month_abbr[start_month]}-{calendar.month_abbr[end_month]}.tsv"
    print(f"URL generada: {url}")
    output_path = os.path.join(folder_path, output_filename)
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"Archivo descargado y guardado como {output_path}")
            saved_paths.append(output_path)
        else:
            print(f"Error al descargar el archivo. Código de estado: {response.status_code}")
    except Exception as e:
        print(f"Ocurrió un error durante la descarga: {e}")

def download_hindcast_and_forecast(model, month, start_month, end_month, folder_path):
    current_year = datetime.datetime.now().year+1
    month_number = list(calendar.month_abbr).index(month)
    if month_number > start_month and month_number > end_month:
        hindcast_start_year = 1992
        hindcast_end_year = 2021  
    elif month_number > start_month or month_number > end_month:
        hindcast_start_year = 1991
        hindcast_end_year = 2021  
    else:
        hindcast_start_year = 1991
        hindcast_end_year = 2020  
    download_and_save_as_tsv(model, month, start_month, end_month, folder_path, hindcast=True, hindcast_start_year=hindcast_start_year, hindcast_end_year=hindcast_end_year)
    forecast_years = []
    for year in range(2020, current_year + 1):
        if month_number < start_month and month_number < end_month:
            forecast_years.append((year, year))  
        elif month_number > start_month and month_number > end_month:
            forecast_years.append((year, year)) 
        else:
            forecast_years.append((year, year + 1))  
    for year_start, year_end in forecast_years:
        download_and_save_as_tsv(model, month, start_month, end_month, folder_path, year_start=year_start, year_end=year_end, hindcast=False)
def create_model_directory(main_dir, model):
    model_dir = os.path.join(main_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def download_and_process_data(model, month, start_month, end_month, main_dir, nrows_per_block):
    model_dir = create_model_directory(main_dir, model)
    download_hindcast_and_forecast(model, month, start_month, end_month, model_dir)
    files_paths = os.listdir(model_dir)
    forecast_paths = sorted([os.path.join(model_dir, f) for f in files_paths if 'hindcast' not in f],
                            key=lambda f: int(re.search(r'(\d{4})', f).group(1)))
    data_arrays_forecast = [process_data(file, nrows_per_block) for file in forecast_paths]
    forecast = xr.concat(data_arrays_forecast, dim='S')
    forecast = forecast.rename({'lon': 'X', 'lat': 'Y'})
    
    return forecast, model_dir

def process_and_merge_models(models, month, start_month, end_month, main_dir, nrows_per_block):
    if isinstance(models, str):
        models = [models]

    model_mapping = {
        "NCEP-CFSv2": "cfsv2",
        "Can-1": "cmc1",
        "Can-2": "cmc2",
        "CCSM4": "ncar_ccsm4",
        "GFDL": "gfdl",
        "NASA": "nasa"
    }
    mapped_models = []
    
    for model in models:
        if model in model_mapping:
            mapped_models.append(model_mapping[model])
        else:
            print(f"Modelo '{model}' no se encuentra disponible en el mapeo.")
    if not mapped_models:
        print("No se encontraron modelos de NMEE.")
        return None
    print(f"Modelos ajustados: {mapped_models}")  
    
    combined_data_xarray = []
    
    for model in mapped_models:
        forecast_data, model_dir = download_and_process_data(model, month, start_month, end_month, main_dir, nrows_per_block)
        combined_data_xarray.append(forecast_data)

    if len(combined_data_xarray) == 1:
        combined_data_xarray = combined_data_xarray[0]
        combined_data_xarray = combined_data_xarray.expand_dims('M')
        combined_data_xarray = combined_data_xarray.assign_coords(M=(mapped_models))
    else:
        combined_data_xarray = xr.concat(combined_data_xarray, dim='M')
        combined_data_xarray = combined_data_xarray.assign_coords(M=(mapped_models))

    return combined_data_xarray
def pearson_to_kendall(r):
    return (2 / np.pi) * np.arcsin(r)
def kendall_tau(a, b):
    """
    Calcula el coeficiente de correlación de Kendall Tau entre dos arrays.

    Parámetros:
    - a (array-like): Primer conjunto de datos.
    - b (array-like): Segundo conjunto de datos.

    Retorna:
    - float: El coeficiente de correlación de Kendall Tau.
    
    Lanza:
    - ValueError: Si las entradas no tienen la misma longitud.
    - Exception: Si ocurre un error inesperado durante el cálculo.
    """
    
    try:
        # Verificar si las entradas tienen la misma longitud
        if len(a) != len(b):
            raise ValueError("Los arrays de entrada deben tener la misma longitud.")
        
        # Calcular el coeficiente de correlación de Kendall Tau
        tau, _ = kendalltau(a, b)
        
        return tau
    
    except ValueError as ve:
        print(f"Error en los datos de entrada: {ve}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None
def get_gcm_url(start_time, start_year, last_year, start_step, last_step, lat_1, lat_2, lon_1, lon_2, model):
    """
    Esta función genera una URL para descargar datos de precipitación de diferentes modelos GCM.

    Parámetros:
    - start_time (str): La hora de inicio en formato 'HHMM'.
    - start_year (int): El primer año del período de datos.
    - last_year (int): El último año del período de datos.
    - start_step (int): El paso de inicio (en días).
    - last_step (int): El paso final (en días).
    - lat_1 (float): La latitud inicial del área de interés.
    - lat_2 (float): La latitud final del área de interés.
    - lon_1 (float): La longitud inicial del área de interés.
    - lon_2 (float): La longitud final del área de interés.
    - model (str): El nombre del modelo GCM a utilizar. Debe ser una de las siguientes opciones:
        'NCEP-CFSv2', 'Can-1', 'Can-2', 'CCSM4', 'GFDL', 'NASA', 'Meteo-France', 
        'Glosea', 'CMCC', 'DWD', 'JMA'.
    
    Retorna:
    - str: La URL generada para el modelo especificado.
    
    Lanza:
    - ValueError: Si el modelo especificado no es válido.
    """
    dic = {
        'NCEP-CFSv2': "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/S/(0000%201%20" + start_time + "%20" + str(start_year) + "-" + str(last_year) + ")/VALUES/L/" + str(start_step) + "/" + str(last_step) + "/RANGE/%5BL%5D//keepgrids/average/c%3A/90//units//days/def/%3Ac/mul/M/1/28/RANGE/%5BM%5Daverage/Y/(" + str(lat_1) + ")/(" + str(lat_2) + ")/RANGEEDGES/X/(" + str(lon_1) + ")/(" + str(lon_2) + ")/RANGEEDGES/-999/setmissing_value/data.nc",
        'Can-1': "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPS-IC3/.HINDCAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%20" + str(start_year) + "-2020)/VALUES/SOURCES/.Models/.NMME/.CanSIPS-IC3/.FORECAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%202021-" + str(last_year) + ")/VALUES/appendstream/L/" + str(start_step) + "/" + str(last_step) + "/RANGE/%5BL%5D//keepgrids/average/c%3A/90//units//days/def/%3Ac/mul/M/1/20/RANGE/%5BM%5Daverage/Y/(" + str(lat_1) + ")/(" + str(lat_2) + ")/RANGEEDGES/X/(" + str(lon_1) + ")/(" + str(lon_2) + ")/RANGEEDGES/-999/setmissing_value/data.nc",
        'Can-2': "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPS-IC3/.CanCM4i-IC3/.HINDCAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%20" + str(start_year) + "-2020)/VALUES/SOURCES/.Models/.NMME/.CanSIPS-IC3/.CanCM4i-IC3/.FORECAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%202021-" + str(last_year) + ")/VALUES/appendstream/L/" + str(start_step) + "/" + str(last_step) + "/RANGE/%5BL%5D//keepgrids/average/c%3A/90//units//days/def/%3Ac/mul/M/1/10/RANGE/%5BM%5Daverage/Y/(" + str(lat_1) + ")/(" + str(lat_2) + ")/RANGEEDGES/X/(" + str(lon_1) + ")/(" + str(lon_2) + ")/RANGEEDGES/-999/setmissing_value/data.nc",
        'CCSM4': "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%20" + str(start_year) + "-" + str(last_year) + ")/VALUES/L/" + str(start_step) + "/" + str(last_step) + "/RANGE/%5BL%5D//keepgrids/average/c%3A/90//units//days/def/%3Ac/mul/M/1/10/RANGE/%5BM%5Daverage/Y/(" + str(lat_1) + ")/(" + str(lat_2) + ")/RANGEEDGES/X/(" + str(lon_1) + ")/(" + str(lon_2) + ")/RANGEEDGES/-999/setmissing_value/data.nc",
        'GFDL': "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-SPEAR/.HINDCAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%20" + str(start_year) + "-2019)/VALUES/M/1/15/RANGE/%5BM%5Daverage/SOURCES/.Models/.NMME/.GFDL-SPEAR/.FORECAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%202020-" + str(last_year) + ")/VALUES/M/1/30/RANGE/%5BM%5Daverage/appendstream/L/" + str(start_step) + "/" + str(last_step) + "/RANGE/%5BL%5D//keepgrids/average/c%3A/90//units//days/def/%3Ac/mul/Y/(" + str(lat_1) + ")/(" + str(lat_2) + ")/RANGEEDGES/X/(" + str(lon_1) + ")/(" + str(lon_2) + ")/RANGEEDGES/-999/setmissing_value/data.nc",
        'NASA': "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%20" + str(start_year) + "-2016)/VALUES/M/1/4/RANGE/%5BM%5Daverage/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.FORECAST/.MONTHLY/.prec/S/(0000%201%20" + start_time + "%202017-" + str(last_year) + ")/VALUES/M/1/10/RANGE/%5BM%5Daverage/appendstream/L/" + str(start_step) + "/" + str(last_step) + "/RANGE/%5BL%5D//keepgrids/average/c%3A/90//units//days/def/%3Ac/mul/Y/(" + str(lat_1) + ")/(" + str(lat_2) + ")/RANGEEDGES/X/(" + str(lon_1) + ")/(" + str(lon_2) + ")/RANGEEDGES/-999/setmissing_value/data.nc",
        'ECMWF':"https://iridl.ldeo.columbia.edu/SOURCES/.EU/.Copernicus/.CDS/.C3S/.ECMWF/.SEAS51/.hindcast/.prcp/S/(0000%201%20"+ start_time +"%20"+ str(start_year) +"-2021)/VALUES/M/1/25/RANGE/[M]average/SOURCES/.EU/.Copernicus/.CDS/.C3S/.ECMWF/.SEAS51/.forecast/.prcp/S/(0000%201%20"+ start_time +"%202022-"+ str(last_year) +")/VALUES/M/1/51/RANGE/[M]average/appendstream/L/" + str(start_step) + "/" + str(last_step) +  "/RANGE/[L]//keepgrids/average/%28mm/day%29/unitconvert/c:/90//units//days/def/:c/mul/Y/("+ str(lat_1) + ")/(" + str(lat_2) +")/RANGEEDGES/X/("+ str(lon_1) +")/("+str(lon_2)+")/RANGEEDGES/-999/setmissing_value/data.nc"
    }
    
    if model not in dic:
        print(f"El modelo '{model}' no está disponible en la lista.")
        return None
    # Retornar la URL correspondiente al modelo especificado
    return dic[model]
def download_nc_file(url, path,username,password, timeout=300):
    """
    Descarga un archivo NetCDF desde una URL y lo guarda en la ruta especificada.
    """
    username = str(username)
    password = str(password)
    login_url = 'https://iridl.ldeo.columbia.edu/auth/login/local/submit/login'

    try:
        session = requests.Session()
        payload = {'email': username, 'password': password, 'redirect': 'https://iridl.ldeo.columbia.edu/auth'}
        login_response = session.post(login_url, data=payload)
        if login_response.status_code != 200 or 'login' in login_response.url:
            print(f"Error al iniciar sesión: {login_response.status_code}")
            return False

        print("Inicio de sesión exitoso")
        time.sleep(6)

        response = session.get(url, stream=True, timeout=timeout, verify=True)
        response.raise_for_status()

        expected_size = int(response.headers.get('Content-Length', 0))
        downloaded_size = 0
        progress_bar = tqdm(total=expected_size, unit='B', unit_scale=True, desc=path.split('/')[-1], miniters=1)

        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded_size += len(chunk)
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

        if downloaded_size == expected_size:
            print(f"Archivo descargado exitosamente: {path}")
            return True
        else:
            print(f"El tamaño del archivo descargado ({downloaded_size} bytes) no coincide con el tamaño esperado ({expected_size} bytes).")
            if os.path.exists(path):
                os.remove(path)
    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Error en la descarga: {e}")
        if os.path.exists(path):
            os.remove(path)

    return False
def generate_chirps_url(lat_min, lat_max, lon_min, lon_max):
    """
    Función para generar la URL de descarga de datos CHIRPS basado en coordenadas de latitud y longitud.

    Parámetros:
    lat_min -- Latitud mínima (ejemplo: '12N')
    lat_max -- Latitud máxima (ejemplo: '16N')
    lon_min -- Longitud mínima (ejemplo: '90W')
    lon_max -- Longitud máxima (ejemplo: '83W')

    Retorna:
    url -- La URL generada para descargar los datos en formato NetCDF.
    """
    
    # URL base para los datos de CHIRPS mensuales
    base_url = "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation"
    
    # Crear el rango de latitudes y longitudes en el formato correcto
    lat_range = f"/Y/%28{lat_min}%29%28{lat_max}%29RANGEEDGES"
    lon_range = f"/X/%28{lon_min}%29%28{lon_max}%29RANGEEDGES"
    
    # Construir la URL completa
    url = f"{base_url}{lat_range}{lon_range}/data.nc"
    
    return url
def download_chirps_data(url, output_path):
    """
    Descarga un archivo de datos CHIRPS desde una URL y lo guarda en el disco.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Datos descargados exitosamente y guardados en {output_path}")
        else:
            print(f"Error al descargar los datos. Código de estado: {response.status_code}")
            return None
        
        try:
            nc_data = Dataset(output_path, mode='r')
            return nc_data
        except Exception as e:
            print(f"Error al cargar el archivo NetCDF: {e}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error durante la solicitud de descarga: {e}")
        return None
def download_data_chirps(params):
    try:
        # Obtener la fecha actual
        now = datetime.datetime.now()
        print(f"Fecha actual: {now}")

        # Calcular la fecha de fin como tres meses antes del mes actual
        three_months_ago = now - datetime.timedelta(days=90)
        end_year = three_months_ago.year
        end_month = three_months_ago.month
        end_day = (datetime.date(end_year, end_month + 1, 1) - datetime.timedelta(days=1)).day
        end_month_str = three_months_ago.strftime('%b')

        print(f"Fecha de fin calculada: {end_day}-{end_month_str}-{end_year}")

        base_path = params['save_path']
        chirps_path = os.path.join(base_path, 'chirps')
        os.makedirs(chirps_path, exist_ok=True)

        url = generate_chirps_url(params['lat_min'], params['lat_max'], params['lon_min'], params['lon_max'])
        download_chirps_data(url, os.path.join(chirps_path, 'chirps_daily.nc'))

        return "Datos CHIRPS descargados exitosamente."
    except Exception as e:
        print(f"Error al descargar datos CHIRPS: {e}")
        return f"Error al descargar datos CHIRPS: {e}"
def convert_coordinates(coord_str):
    """
    Convierte una coordenada con formato 12N, 90W, etc. a un valor numérico.
    N y E se mantienen positivos, S y W se convierten a negativos.
    """
    if coord_str[-1] == 'N':
        return float(coord_str[:-1])  # Latitud norte es positiva
    elif coord_str[-1] == 'S':
        return -float(coord_str[:-1])  # Latitud sur es negativa
    elif coord_str[-1] == 'E':
        return float(coord_str[:-1])  # Longitud este es positiva
    elif coord_str[-1] == 'W':
        return -float(coord_str[:-1])  # Longitud oeste es negativa
    else:
        raise ValueError("Formato de coordenada no reconocido")
def download_data_gcm(params):
    base_path = params['save_path']
    descarga_path = os.path.join(base_path, 'descarga')
    ensamble_path = os.path.join(base_path, 'ensamble')
    os.makedirs(descarga_path, exist_ok=True)
    os.makedirs(ensamble_path, exist_ok=True)

    lat_1 = convert_lat_lon(params['lat_min'])
    lat_2 = convert_lat_lon(params['lat_max'])
    lon_1 = convert_lat_lon(params['lon_min'])
    lon_2 = convert_lat_lon(params['lon_max'])
    models = params['modelos']
    start_year = 1982
    now = datetime.datetime.now()
    end_year = now.year - 1

    start_time  = (now).strftime("%b")
    if isinstance(models, str):
        models = [models]

    try:
        urls = list(map(lambda model: get_gcm_url(start_time, start_year, end_year, params['start_step'], params['last_step'], lat_1, lat_2, lon_1, lon_2, model), models))
        print("URLs generadas correctamente:")
        for url in urls:
            print(url)
    except ValueError as e:
        return f"Error al generar URLs: {e}"

    generate_path = lambda model: f"{descarga_path}/{model}_{start_time}_{params['start_step']}-{params['last_step']}.nc"
    paths = list(map(generate_path, models))

    try:
        list(map(lambda url, path: download_nc_file(url, path, params['username'], params['password']), urls, paths))
        print("Archivos descargados correctamente en las siguientes rutas:")
        for path in paths:
            print(path)
    except Exception as e:
        return f"Error al descargar archivos NC: {e}"
    for path in paths:
        if not os.path.exists(path):
            print(f"El archivo no se encontró: {path}")
        else:
            print(f"Archivo encontrado: {path}, tamaño: {os.path.getsize(path)} bytes")
    return paths
def ensure_numpy_array(value):
    if isinstance(value, xr.DataArray):
        return value.values
    return value
def calculate_winter_precip(path, months):
    """
    Calcula la precipitación mensual acumulada para los meses de invierno especificados,
    a partir del año 1981 hasta el año final determinado por el tamaño de la dimensión T.
    """
    try:
        # Cargar el dataset y crear una secuencia mensual desde enero 1981
        precip_chirps_daily = xr.open_dataset(path, decode_times=False).precipitation
        
        # Asegurarse de que la dimensión T sea de tipo numpy array antes de convertir a fechas
        T_size = precip_chirps_daily['T'].size  # Obtener el tamaño de la dimensión T
        precip_chirps_daily['T'] = pd.date_range(start='1981-01', periods=T_size, freq='M')
        precip_chirps_daily = precip_chirps_daily.expand_dims({'M': [0]})
        results = []

        # Obtener el último año del rango de fechas en la dimensión 'T'
        last_year = precip_chirps_daily['T'].dt.year.max().item()  # Convertir a un valor entero

        # Iterar sobre los años en función del rango de fechas del dataset
        for year in range(1981, last_year + 1):
            start_month = months[0]
            end_month = months[-1]

            if start_month <= end_month:
                # Si los meses están dentro del mismo año
                start_date = f'{year}-{int(start_month):02d}-01'
                end_date = f'{year}-{int(end_month):02d}-{pd.Timestamp(f"{year}-{int(end_month):02d}-01").days_in_month}'
                save_year = year
            else:
                # Si los meses cruzan el año, acumulamos en el siguiente año
                start_date = f'{year}-{int(start_month):02d}-01'
                end_date = f'{year + 1}-{int(end_month):02d}-{pd.Timestamp(f"{year + 1}-{int(end_month):02d}-01").days_in_month}'
                save_year = year + 1

            # Filtrar los datos para el periodo seleccionado
            filtered_precip_chirps_daily = precip_chirps_daily.sel(T=slice(start_date, end_date))

            # Asegurarse de que tenemos todos los meses requeridos
            available_dates = pd.to_datetime(ensure_numpy_array(filtered_precip_chirps_daily['T']))
            months_present = set(available_dates.month)
            required_months = set(months)

            if required_months.issubset(months_present):
                # Sumar la precipitación mensual para los meses seleccionados
                total_precipitation = filtered_precip_chirps_daily.sum(dim='T')
                total_precipitation = total_precipitation.expand_dims(year=[save_year])
                results.append(total_precipitation)
            else:
                print(f"No se encontraron todos los meses requeridos para el año {year}.")

        # Concatenar los resultados para todos los años
        if results:
            winter_precip = xr.concat(results, dim='year')
            return winter_precip
        else:
            print('No se encontraron datos completos para el rango de fechas especificado.')
            return None

    except FileNotFoundError:
        print(f"El archivo en la ruta {path} no fue encontrado.")
        return None
    except Exception as e:
        print(f"Ocurrió un error durante el cálculo de precipitación: {e}")
        return None
def convert_lat_lon(value):
    if 'N' in value:
        return float(value.replace('N', ''))
    elif 'S' in value:
        return -float(value.replace('S', ''))
    elif 'E' in value:
        return float(value.replace('E', ''))
    elif 'W' in value:
        return -float(value.replace('W', ''))
    return float(value)
def convert_month(month_str):
    months = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    return months.get(month_str, 1)
def create_values(limit_A, limit_B, limit_C):
    A = list(range(1, limit_A + 1))
    B = list(range(1, limit_B + 1))
    C = list(range(1, limit_C + 1))
    return A, B, C
def save_raster(data_array, output_path):
    if isinstance(data_array, xr.Dataset):
        # Asumiendo que necesitas un DataArray específico dentro del Dataset
        data_array = data_array.to_array().isel(variable=0)  # Modifica según sea necesario para obtener el DataArray correcto

    lon = data_array.coords['x']
    lat = data_array.coords['y']

    # Voltear el array de datos a lo largo del eje de latitud
    flipped_data = np.flip(data_array.values, axis=0)
    
    transform = from_origin(west=lon.min().item(), north=lat.max().item(), xsize=(lon.max().item()-lon.min().item())/len(lon), ysize=(lat.max().item()-lat.min().item())/len(lat))

    # Guardar como un archivo raster
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=flipped_data.shape[0],
        width=flipped_data.shape[1],
        count=1,
        dtype=flipped_data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(flipped_data, 1)

    print(f"Archivo raster guardado en {output_path}")
def run_forecast(params, obs_f):
    base_path = params['save_path']
    ensamble_path = os.path.join(base_path, 'ensamble')
    fore_path = os.path.join(base_path, 'forecast')
    fore_path_input=os.path.join(fore_path, 'input')
    files_path = os.path.join(fore_path, 'files')
    img_path = os.path.join(fore_path, 'img')
    tif_path=os.path.join(fore_path, 'raster')
    os.makedirs(files_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(tif_path, exist_ok=True)
    os.makedirs(fore_path,exist_ok=True)
    os.makedirs(fore_path_input,exist_ok=True)
    now = datetime.datetime.now()
    start_time = (now).month
    start_step = params['start_step']
    last_step = params['last_step']
    start_month = int(start_time + start_step - 0.5)
    end_month = int(start_time + last_step - 0.5)
    if start_month > 12:
        start_month -= 12
    if end_month > 12:
        end_month -= 12

    months = [start_month, end_month]
    print(months)
    obs = calculate_winter_precip(os.path.join(base_path, 'chirps', 'chirps_daily.nc'), months)
    
    obs = obs.rename({'year': 'S'})
    drymask = xc.drymask(obs, dry_threshold=0.1, quantile_threshold=0.1)
    obs = obs * drymask
    if obs.rio.crs is None:
        obs = obs.rio.write_crs("EPSG:4326")

    model1_h = xr.open_dataset(os.path.join(ensamble_path, 'ensamble.nc'), decode_times=False).prec
    model1_h = model1_h.expand_dims('M').assign_coords(M=[0])

    if model1_h.rio.crs is None:
        model1_h = model1_h.rio.write_crs("EPSG:4326")
    # Imprimir el primer y último elemento de obs en S sin filtrar
    print("Periodo inicial en datos observados:")
    print(obs.isel(S=0))
    print("Periodo final de los datos observados:")
    print(obs.isel(S=-1))

    # Imprimir el primer y último elemento de model1_h antes del filtro
    print("Periodo inicial en datos modelados:")
    print(model1_h.isel(S=0))
    print("Periodo final en datos modelados:")
    print(model1_h.isel(S=-1))
    drymask = xc.drymask(model1_h, dry_threshold=0.1, quantile_threshold=0.1)
    model1_h = model1_h * drymask
    model1_f = model1_h.dropna(dim='S', how='all')
    model1_f_S = model1_f['S'].values
    obs_S = obs['S'].values
    common_S = np.intersect1d(model1_f_S, obs_S)
    model1_f = model1_f.sel(S=common_S)
    obs_f = obs.sel(S=common_S)
    nrows_per_block = 182  # Número de filas por bloque

    combined_data = process_and_merge_models(params['modelos'],(now).strftime("%b"), start_month, end_month, fore_path_input, nrows_per_block)
    ecmwf=None
    # Verifica si 'ECMWF' está en los modelos de params
    ecmwf = None
    
    # Verifica si 'ECMWF' está en los modelos de params
    if 'ECMWF' in params['modelos']:
        now = datetime.datetime.now()
        dataset = "seasonal-monthly-single-levels"
        lat_min = convert_coordinates(params['lat_min'])
        lat_max = convert_coordinates(params['lat_max'])
        lon_min = convert_coordinates(params['lon_min'])
        lon_max = convert_coordinates(params['lon_max'])
        area = [lat_max, lon_min, lat_min, lon_max]
        if start_step == 1.5:
            leadtime_month = ['1', '2', '3']
        elif start_step == 4.5:
            leadtime_month = ['4', '5', '6']
        request = {
            'originating_centre': 'ecmwf',
            'system': '51',
            'variable': ['total_precipitation'],
            'product_type': ['monthly_mean'],  # Cambiado a monthly_mean
            'year': [str(now.year)],  # Año actual
            'month': [str(now.month-1).zfill(2)],  # Mes actual en formato de 2 dígitos
            'leadtime_month':leadtime_month,
            'data_format': 'netcdf',
            'area': area  # Coordenadas convertidas
        }
        
        print(request)
        
        # Crear carpeta para ECMWF si no existe
        ecmwf_folder = os.path.join(fore_path_input, "ecmwf")
        os.makedirs(ecmwf_folder, exist_ok=True)

        output_file_path = os.path.join(ecmwf_folder, "forecast_data.nc")
        client = cdsapi.Client()

        # Espera activa para la descarga
        while True:
            try:
                client.retrieve(dataset, request).download(target=output_file_path)
                print(f"El archivo ha sido descargado: {output_file_path}")
                break  # Salir del bucle si la descarga es exitosa
            except ValueError as e:
                # Si el trabajo aún está en ejecución, esperar 180 segundos
                if "job is running" in str(e):
                    print("El trabajo aún está en ejecución, esperando 180 segundos...")
                    time.sleep(180)  # Esperar 180 segundos antes de volver a intentar
                else:
                    # Si ocurre otro error, levantar la excepción
                    raise

        # Procesar el archivo descargado
        ecmwf = xr.open_dataset(output_file_path).tprate
        ecmwf=ecmwf.mean(dim='number')
        ecmwf = ecmwf.rename({'latitude': 'Y', 'longitude': 'X', 'forecast_reference_time': 'S'})
        ecmwf.values=ecmwf.values* 1000 * 86400 * 30
        ecmwf=ecmwf.sum(dim='forecastMonth')
        ecmwf['X'] = xr.where(ecmwf['X'] > 180, ecmwf['X'] - 360, ecmwf['X'])
        ecmwf = ecmwf.rio.set_spatial_dims('X', 'Y', inplace=True)
        ecmwf = ecmwf.rio.write_crs("EPSG:4326", inplace=True)
        ecmwf = ecmwf.sortby(['X', 'Y'])
        ecmwf['S']=ecmwf['S'].dt.year.values
        ecmwf = ecmwf.expand_dims(dim={'M': ['ECMWF']})
        ecmwf=xc.regrid(ecmwf, obs_f.X, obs_f.Y)

        print(f"El archivo ha sido procesado y guardado en: {output_file_path}")

    else:
        print("'ecmwf' no está en modelos. No se realiza ningún proceso.")
    if ecmwf is not None:
        ecmwf = ecmwf.isel(S=-1)

    # Regrid combined_data si no es None
    if combined_data is not None:
        combined_data = xc.regrid(combined_data, obs_f.X, obs_f.Y)
        combined_data = combined_data.isel(S=-1)

    # Si ambos ecmwf y combined_data no son None, concatenar y promediar en 'M'
    if ecmwf is not None and combined_data is not None:
        # Asegurarse de que ambos tienen la dimensión 'M'
        if 'M' not in ecmwf.dims:
            ecmwf = ecmwf.expand_dims(dim={'M': 1}, axis=0)
        if 'M' not in combined_data.dims:
            combined_data = combined_data.expand_dims(dim={'M': 1}, axis=0)

        # Concatenar y calcular el promedio en 'M'
        combined_data = xr.concat([ecmwf, combined_data], dim='M')
        fore = combined_data.mean(dim='M').expand_dims(dim={'M': 1}, axis=0)

    # Si solo combined_data está disponible, calcular el promedio en 'M'
    elif combined_data is not None:
        if 'M' not in combined_data.dims:
            combined_data = combined_data.expand_dims(dim={'M': 1}, axis=0)
        fore = combined_data.mean(dim='M').expand_dims(dim={'M': 1}, axis=0)

    # Si solo ecmwf está disponible, asegurar que tiene la dimensión 'M' y asignar a fore
    elif ecmwf is not None:
        if 'M' not in ecmwf.dims:
            ecmwf = ecmwf.expand_dims(dim={'M': 1}, axis=0)
        fore = ecmwf

    # Si ninguno está disponible, imprimir mensaje y asignar fore a None
    else:
        print("No hay data disponible para forecast")
        fore = None
    print(fore)
    print(ecmwf)
    print(combined_data)
    eof_x = params['modos_x']
    eof_y = params['modos_y']
    cca_modos = params['modos_cca']
    A, B, C = create_values(eof_x, eof_y, cca_modos)
    combinaciones = itertools.product(A, B, C)
    combinaciones_filtradas = [(a, b, c) for a, b, c in combinaciones if c <= min(a, b)]
    resultados = []
    step_difference = params['last_step'] - params['start_step'] + 1
    window = 3 if step_difference == 3 else 2
    S_value = 1 if step_difference == 3 else 0
    for combinacion in combinaciones_filtradas:
        est_kwargs = {'search_override': combinacion, 'latitude_weighting': True}
        ytest_list, preds_list = [], []
        cross_validator = xc.CrossValidator(model1_f, obs_f, window=3)
        for xtrain, ytrain, xtest, ytest in cross_validator:
            reg = xc.CCA(**est_kwargs)
            reg.fit(xtrain, ytrain)
            preds = reg.predict(xtest)
            preds_list.append(preds.isel(S=1))
            ytest_list.append(ytest.isel(S=1))

            hindcasts_det = xr.concat(preds_list, 'S').chunk(dict(S=-1))
            ytest_concat = xr.concat(ytest_list, 'S').chunk(dict(S=-1))
        kendall_dataarray = xr.apply_ufunc(
            kendall_tau,
            hindcasts_det,
            ytest_concat,
            input_core_dims=[['S'], ['S']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
            )
        tau_kendall = kendall_dataarray.mean().compute().item()
        resultados.append({'Combinacion': combinacion, 'Goodness Index': tau_kendall})

    resultados_df = pd.DataFrame(resultados).round(2)
    resultados_df.to_excel(os.path.join(files_path, 'cross_validation.xlsx'), index=False)

    max_tau = resultados_df['Goodness Index'].max()
    max_tau_df = resultados_df[resultados_df['Goodness Index'] == max_tau]
    max_tau_df['complexity'] = max_tau_df['Combinacion'].apply(lambda x: sum(x))
    least_complex_model = max_tau_df.loc[max_tau_df['complexity'].idxmin()]
    best_combination = least_complex_model['Combinacion']

    best_kwargs = {
        'search_override': best_combination,
        'latitude_weighting': True,
    }

    hindcasts_det, hindcasts_prob = [], []
    step_difference = params['last_step'] - params['start_step'] + 1
    window = 3 if step_difference == 3 else 2
    S_value = 1 if step_difference == 3 else 0
    for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model1_f, obs_f, window=window):
        reg = xc.CCA(**best_kwargs)
        reg.fit(xtrain, ytrain)
        preds = reg.predict(xtest)
        probs = reg.predict_proba(xtest)
        hindcasts_det.append(preds.isel(S=S_value))
        hindcasts_prob.append(probs.isel(S=S_value))

    hindcasts_det = xr.concat(hindcasts_det, 'S')
    hindcasts_prob = xr.concat(hindcasts_prob, 'S')
    
    hindcasts_det = xc.gaussian_smooth(hindcasts_det, kernel=9)
    hindcasts_prob = xc.gaussian_smooth(hindcasts_prob, kernel=9)
    pearson = xc.Pearson(hindcasts_det, obs_f)
    kendall_dataarray = xr.apply_ufunc(
        pearson_to_kendall,
        pearson,
        vectorize=True,
        dask="parallelized"
    )
    kendall_dataarray = kendall_dataarray.rename("Kendall")
    plt.figure()
    xc.view(kendall_dataarray, drymask=drymask, title='Kendall CC', cmap=plt.get_cmap('RdBu', 8), vmin=-1, vmax=1)
    plt.savefig(os.path.join(img_path, 'kendall.png'))
    plt.figure()
    xc.view(pearson, drymask=drymask, title='Pearson CC', cmap=plt.get_cmap('RdBu', 8), vmin=-1, vmax=1)
    plt.savefig(os.path.join(img_path, 'pearson.png'))
    ohc = xc.OneHotEncoder() 
    ohc.fit(obs)
    T = ohc.transform(obs_f)
    clim = xr.ones_like(T) * 0.333
    plt.figure()
    xc.view_roc(hindcasts_prob, T)
    plt.savefig(os.path.join(img_path, 'roc.png'))

    if 'M' not in fore.dims:
        fore = fore.expand_dims('M').assign_coords(M=[0])
    if 'S' not in fore.dims:
        fore = fore.expand_dims('S').assign_coords(S=[0])
    fore['M'] = obs_f['M']
    print(fore)
    reg = xc.CCA(**best_kwargs) 
    reg.fit(model1_f, obs_f)
    preds = reg.predict(fore)
    probs = reg.predict_proba(fore)

    forecasts_det_smooth = xc.gaussian_smooth(preds, kernel=9)
    forecasts_prob_smooth = xc.gaussian_smooth(probs, kernel=9)
    forecasts_det_smooth_anomaly = forecasts_det_smooth - hindcasts_det.mean('S')

    forecasts_det_smooth.to_netcdf(os.path.join(files_path, 'forecasts_deterministic.nc'))
    forecasts_det_smooth_anomaly.to_netcdf(os.path.join(files_path, 'forecasts_anomaly.nc'))
    forecasts_prob_smooth.to_netcdf(os.path.join(files_path, 'forecasts_tercile.nc'))
    forecsts_probs20 = reg.predict_proba(fore, quantile=0.2)
    forecsts_probs80 = 1 - reg.predict_proba(fore, quantile=0.8)
    forecasts_prob20_smooth = xc.gaussian_smooth(forecsts_probs20, kernel=9)
    forecasts_prob80_smooth = xc.gaussian_smooth(forecsts_probs80, kernel=9)
    forecasts_prob20_smooth.to_netcdf(os.path.join(files_path, 'forecasts_prob_20.nc'))
    forecasts_prob80_smooth.to_netcdf(os.path.join(files_path, 'forecasts_prob_80.nc'))
    groc=xc.GROCS(hindcasts_prob, T)
    plt.figure()
    pl=xc.view(groc,drymask=drymask,title='GROCS',cmap=plt.get_cmap('RdBu',8),vmin=0,vmax=1)
    plt.savefig(os.path.join(img_path, 'GROCS.png'))
    print(resultados_df)
    plt.figure()
    xc.view_probabilistic(forecasts_prob_smooth.isel(S=0), title='Probabilisitc forecast', drymask=drymask)
    plt.savefig(os.path.join(img_path, 'prob_forecast.png'))
    last_obs = obs_f.isel(S=-1)
    print("Data Observada ultimo ano:")
    print(last_obs['S'])
    BN=forecasts_prob_smooth.sel(M='BN').isel(S=-1)
    NN=forecasts_prob_smooth.sel(M='NN').isel(S=-1)
    AN=forecasts_prob_smooth.sel(M='AN').isel(S=-1)
    max_indices = forecasts_prob_smooth.isel(S=-1).fillna(-np.inf).argmax(dim='M')
    result = forecasts_prob_smooth.isel(S=-1).copy()
    result = xr.where(max_indices == 0, result, result + 0) 
    result = xr.where(max_indices == 1, result + 1, result)  
    result = xr.where(max_indices == 2, result + 2, result)  
    result=result.isel(M=0)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    honduras = world[world.name == 'Honduras']
    AN = AN.rename({'X': 'x', 'Y': 'y'})
    BN = BN.rename({'X': 'x', 'Y': 'y'})
    NN = NN.rename({'X': 'x', 'Y': 'y'})
    result = result.rename({'X': 'x', 'Y': 'y'})
    AN=AN.rio.write_crs("EPSG:4326").rio.clip(honduras.geometry, honduras.crs)
    BN=BN.rio.write_crs("EPSG:4326").rio.clip(honduras.geometry, honduras.crs)
    NN=NN.rio.write_crs("EPSG:4326").rio.clip(honduras.geometry, honduras.crs)
    result=result.rio.write_crs("EPSG:4326").rio.clip(honduras.geometry, honduras.crs)
    save_raster(AN,os.path.join(tif_path, 'above.tif'))
    save_raster(BN,os.path.join(tif_path, 'below.tif'))
    save_raster(NN,os.path.join(tif_path, 'normal.tif'))
    save_raster(result,os.path.join(tif_path, 'dominant.tif'))
    last_model1_f = model1_f.isel(S=-1)
    print("Datos del modelo 1 ultimo ano:")
    print(last_model1_f['S'])
    last_forecast = fore.isel(S=-1)
    print("Datos del pronóstico ultimo ano:")
    print(last_forecast['S'])
def create_ensemble(params, paths, precip_monthly):
    """
    Crea el ensamble a partir de los datos GCM descargados.
    """
    gcm = []
    start_step= params['start_step']
    directory_path = params['save_path']
    descarga_path = os.path.join(directory_path, 'descarga')
    if paths:  # Si hay paths
        for file_path in paths:
            try:
                print(file_path)
                ds = xr.open_dataset(file_path, decode_times=False, engine='netcdf4').aprod
                ds = ds.rename({'L': 'M'})
                units = ds['S'].attrs['units']
                origin = pd.Timestamp(units.split('since')[-1].strip())
                dates = [origin + pd.DateOffset(months=month) for month in ds["S"].values]
                date_plus = [date + pd.DateOffset(months=math.floor(params['last_step'])) for date in dates]
                years = [date.year for date in date_plus]
                ds_last = ds.assign_coords({'S': years}).copy()
                ds_regrid = xc.regrid(ds_last, precip_monthly.X, precip_monthly.Y)
                print(ds_regrid)
                gcm.append(ds_regrid)
            except Exception as e:
                print(f"Error al procesar archivo {file_path}: {e}")
    if gcm:
        try:
            Model_hindcast = xr.concat(gcm, 'M')
            Ensamble = Model_hindcast.mean(dim='M').rename('prec')
            Ensamble.to_netcdf(os.path.join(params['save_path'], 'ensamble', 'ensamble.nc'))
            print('Creación de ensamble')
        except Exception as e:
            print(f"Error al crear el ensamble: {e}")
            return None
    
    return Ensamble

def run_script(params, output_path):
    base_path = params.get('save_path', output_path)
    params['save_path'] = base_path
    
    if params['LT'] == 0:
        last_step, start_step = train_step_one()
    elif params['LT'] == 3: 
        last_step, start_step = train_step_two()
    
    params['start_step'] = start_step
    params['last_step'] = last_step
    print(params)
    print('Iniciando descarga de CHIRPS')
    download_data_chirps(params)
    print("Datos CHIRPS descargados exitosamente.")

    print('Descargando datos de GCM')
    paths = download_data_gcm(params)
    print(paths)
    now = datetime.datetime.now()
    start_time = (now).month
    start_month = int(start_time + params['start_step'] - 0.5)
    end_month = int(start_time + params['last_step'] - 0.5)

    if start_month > 12:
        start_month -= 12
    if end_month > 12:
        end_month -= 12

    months = [start_month, end_month]
    print(months)
    precip_monthly = calculate_winter_precip(os.path.join(params['save_path'], 'chirps', 'chirps_daily.nc'), months)
    if precip_monthly is None:
        return "Error en el cálculo de precipitación"

    precip_monthly = precip_monthly.rename({'year': 'S'})

    print('Creando ensamble de GCM')
    create_ensemble(params, paths, precip_monthly)

    run_forecast(params, precip_monthly)
    print('Forecast realizado')

    print(f'Salida guardada en {output_path}')
