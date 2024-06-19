import os
import csv
import sys
import random
import shutil
import numpy as np
import importlib.util

#from openpyxl import load_workbook, Workbook

import torch

def save_to_csv(file_path, row_data, header):
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Write the row to the CSV file
    with open(file_path, 'a', newline='\n') as csv_file:
        writer = csv.writer(csv_file)

        # Write header if the file is empty
        if not file_exists:
            #header = ["Column1", "Column2", "Column3"]  # Replace with your actual column names
            writer.writerow(header)

        # Write the row data
        writer.writerow(row_data)
        
def save_to_excel(file_path, row_data, header):
    
    # Confirm file exists. If not, create it, add headers, then append new data
    try:
        wb = load_workbook(file_path)
        ws = wb.worksheets[0]  # select first worksheet
    except Exception:
        wb = Workbook()
        ws = wb.active
        ws.append(header) #header row
    ws.append(row_data)
    wb.save(file_path)
    wb.close()
    
def copy_src(root_src_dir, root_dst_dir, overwrite=True):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            if 'cpython' in file:
                continue
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.exists(dst_file):
                if overwrite:
                    shutil.copy(src_file, dst_file)
            else:
                shutil.copy(src_file, dst_file)

def reload_module_from_directory(module_name, directory, verbose=0):
    
    # try:
    # need this one because the module_name is also importing other module 
    if directory not in sys.path:
        sys.path.insert(0, directory)
        if verbose:
            print(sys.path[:3])
        
    # Remove the existing module from memory
    if module_name in sys.modules:
        if verbose:
            print(f" Deleting {sys.modules[module_name]} from modules ... ")
            
        del sys.modules[module_name]

    # Construct the module path
    module_path = f"{directory}/{module_name.replace('.', '/')}.py"

    # Import the module from the new directory
    spec = importlib.util.spec_from_file_location(f"{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if verbose:
        print(f"{module} created")
    
    # add to sys modules
    sys.modules[module_name] = module
    if verbose:
        print(sys.modules[module_name])
        
    sys.path.pop(0)


def remove_module_from_path(module_name):
    # Remove the existing module from memory
    if "data.dataset" in sys.modules:
        del sys.modules["data.dataset"]    

def save_file(file_path, data):
    with open(file_path, 'w') as f:
        f.write(data)

def load_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def create_new_dir(new_dir, clean=False):
    if clean:
        shutil.rmtree(new_dir, ignore_errors=True)
    #if not os.path.exists(new_dir):
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

def listdir(dir_name, file_type=''):
    files = os.listdir(dir_name)
    if file_type is not None:
        files = [f for f in files if f.endswith(file_type)]
    return sorted(files)