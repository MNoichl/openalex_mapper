import pickle
import requests
#import umap
import umap.umap_ as umap 
from numba.typed import List
import torch
from sentence_transformers import SentenceTransformer
import time
from pathlib import Path

def check_resources(files_dict, basemap_path, mapper_params_path):
    """
    Check if all required resources are present.
    
    Args:
        files_dict (dict): Dictionary mapping filenames to their download URLs
        basemap_path (str): Path to the basemap pickle file
        mapper_params_path (str): Path to the UMAP mapper parameters pickle file
        
    Returns:
        bool: True if all resources are present, False otherwise
    """
    all_files_present = True
    
    # Check downloaded files
    for filename in files_dict.keys():
        if not Path(filename).exists():
            print(f"Missing file: {filename}")
            all_files_present = False
    
    # Check basemap
    if not Path(basemap_path).exists():
        print(f"Missing basemap file: {basemap_path}")
        all_files_present = False
        
    # Check mapper params
    if not Path(mapper_params_path).exists():
        print(f"Missing mapper params file: {mapper_params_path}")
        all_files_present = False
    
    return all_files_present

def download_required_files(files_dict):
    """
    Download required files from URLs only if they don't exist.
    
    Args:
        files_dict (dict): Dictionary mapping filenames to their download URLs
    """
    print(f"Checking required files: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    files_to_download = {
        filename: url 
        for filename, url in files_dict.items() 
        if not Path(filename).exists()
    }
    
    if not files_to_download:
        print("All files already present, skipping downloads")
        return
        
    print(f"Downloading missing files: {list(files_to_download.keys())}")
    for filename, url in files_to_download.items():
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

def setup_basemap_data(basemap_path):
    """
    Load and setup the base map data.
    
    Args:
        basemap_path (str): Path to the basemap pickle file
    """
    print(f"Getting basemap data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    basedata_df = pickle.load(open(basemap_path, 'rb'))
    return basedata_df

def setup_mapper(mapper_params_path):
    """
    Setup and configure the UMAP mapper.
    
    Args:
        mapper_params_path (str): Path to the UMAP mapper parameters pickle file
    """
    print(f"Getting Mapper: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    params_new = pickle.load(open(mapper_params_path, 'rb'))
    print("setting up mapper...")
    mapper = umap.UMAP()
    
    umap_params = {k: v for k, v in params_new.get('umap_params', {}).items() 
                  if k != 'target_backend'}
    mapper.set_params(**umap_params)
    
    for attr, value in params_new.get('umap_attributes', {}).items():
        if attr != 'embedding_':
            setattr(mapper, attr, value)
    
    if 'embedding_' in params_new.get('umap_attributes', {}):
        mapper.embedding_ = List(params_new['umap_attributes']['embedding_'])
    
    return mapper

def setup_embedding_model(model_name):
    """
    Setup the SentenceTransformer model.
    
    Args:
        model_name (str): Name or path of the SentenceTransformer model
    """
    print(f"Setting up language model: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = SentenceTransformer(model_name)
    return model 