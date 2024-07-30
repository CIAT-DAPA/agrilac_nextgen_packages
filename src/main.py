import os
import argparse

from forecast import run_script
from utils import csv_to_dict, process_params

def main():

    # Params

    parser = argparse.ArgumentParser(description="Resampling script")

    parser.add_argument("-i", "--inputs", help="Inputs path", required=True)
    parser.add_argument("-o", "--outputs", help="Outputs path", required=True)


    args = parser.parse_args()

    print("Cargando la configuración")
    print(args)

    input_path = args.inputs
    output_path = args.outputs

    csv_path = os.path.join(input_path, "config.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo {csv_path} no existe.")


    params_dict = csv_to_dict(csv_path)

    params = process_params(params_dict)

    print("Iniciando el proceso")
    
    run_script(params, output_path)

 


if __name__ == "__main__":
    main()
#Ejemplo de comando
#python main.py -i "D:\Code\next_gen\packages\agrilac_nextgen_packages\config" -o "D:\Code\next_gen\packages\agrilac_nextgen_packages"