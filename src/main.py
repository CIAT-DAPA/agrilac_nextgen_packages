import os
import argparse

from forecast import run_script
from utils import  process_dataframe,create_monthly_folders

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

    xlsx_path = os.path.join(input_path, "config.xlsx")

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"El archivo {xlsx_path} no existe.")



    params_one,params_two = process_dataframe(xlsx_path)
    created_folders = create_monthly_folders(output_path)
    run_script(params_one, created_folders[0])
    run_script(params_two, created_folders[1])
 


if __name__ == "__main__":
    main()
#Ejemplo de comando
#python main.py -i "D:\Code\next_gen\packages\agrilac_nextgen_packages\config" -o "D:\Code\next_gen\packages\agrilac_nextgen_packages"