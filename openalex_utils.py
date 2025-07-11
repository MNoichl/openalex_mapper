import numpy as np
from urllib.parse import urlparse, parse_qs
from pyalex import Works
import pandas as pd
import ast, json

def openalex_url_to_pyalex_query(url):
    """
    Convert an OpenAlex search URL to a pyalex query.
    
    Args:
    url (str): The OpenAlex search URL.
    
    Returns:
    tuple: (Works object, dict of parameters)
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Initialize the Works object
    query = Works()
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        for f in filters:
            if ':' in f:
                key, value = f.split(':', 1)
                if key == 'default.search':
                    query = query.search(value)
                else:
                    query = query.filter(**{key: value})
    
    # Handle sort - Fixed to properly handle field:direction format
    if 'sort' in query_params:
        sort_params = query_params['sort'][0].split(',')
        for s in sort_params:
            if ':' in s:  # Handle field:direction format
                field, direction = s.split(':')
                query = query.sort(**{field: direction})
            elif s.startswith('-'):  # Handle -field format
                query = query.sort(**{s[1:]: 'desc'})
            else:  # Handle field format
                query = query.sort(**{s: 'asc'})
    
    # Handle other parameters
    params = {}
    for key in ['page', 'per-page', 'sample', 'seed']:
        if key in query_params:
            params[key] = query_params[key][0]
    
    return query, params

def invert_abstract(inv_index):
    """Reconstruct abstract from OpenAlex' inverted-index.

    Handles dicts, JSON / repr strings, or missing values gracefully.
    """
    # Try to coerce a string into a Python object first
    if isinstance(inv_index, str):
        try:
            inv_index = json.loads(inv_index)          # double-quoted JSON
        except Exception:
            try:
                inv_index = ast.literal_eval(inv_index)  # single-quoted repr
            except Exception:
                inv_index = None

    if isinstance(inv_index, dict):
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(w for w, _ in sorted(l_inv, key=lambda x: x[1]))
    else:
        return " "  
    
    
def get_pub(x):
    """Extract publication name from record."""
    try: 
        source = x['source']['display_name']
        if source not in ['parsed_publication','Deleted Journal']:
            return source
        else: 
            return ' '
    except:
            return ' '

def get_field(x):
    """Extract academic field from record."""
    try:
        field = x['primary_topic']['subfield']['display_name']
        if field is not None:
            return field
        else:
            return np.nan
    except:
        return np.nan

def process_records_to_df(records):
    """
    Convert OpenAlex records to a pandas DataFrame with processed fields.
    Can handle either raw OpenAlex records or an existing DataFrame.
    
    Args:
    records (list or pd.DataFrame): List of OpenAlex record dictionaries or existing DataFrame
    
    Returns:
    pandas.DataFrame: Processed DataFrame with abstracts, publications, and titles
    """
    # If records is already a DataFrame, use it directly
    if isinstance(records, pd.DataFrame):
        records_df = records.copy()
        # Only process abstract_inverted_index and primary_location if they exist
        if 'abstract_inverted_index' in records_df.columns:
            records_df['abstract'] = [invert_abstract(t) for t in records_df['abstract_inverted_index']]
        if 'primary_location' in records_df.columns:
            records_df['parsed_publication'] = [get_pub(x) for x in records_df['primary_location']]
            records_df['parsed_publication'] = records_df['parsed_publication'].fillna(' ') # fill missing values with space, only if we have them.
        
    else:
        # Process raw records as before
        records_df = pd.DataFrame(records)
        records_df['abstract'] = [invert_abstract(t) for t in records_df['abstract_inverted_index']]
        records_df['parsed_publication'] = [get_pub(x) for x in records_df['primary_location']]
        records_df['parsed_publication'] = records_df['parsed_publication'].fillna(' ')
    
    # Fill missing values and deduplicate
    
    records_df['abstract'] = records_df['abstract'].fillna(' ')
    records_df['title'] = records_df['title'].fillna(' ')
    records_df = records_df.drop_duplicates(subset=['id']).reset_index(drop=True)
    
    return records_df

def openalex_url_to_filename(url):
    """
    Convert an OpenAlex URL to a filename-safe string with timestamp.
    
    Args:
    url (str): The OpenAlex search URL
    
    Returns:
    str: A filename-safe string with timestamp (without extension)
    """
    from datetime import datetime
    import re
    
    # First parse the URL into query and params
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Create parts of the filename
    parts = []
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        for f in filters:
            if ':' in f:
                key, value = f.split(':', 1)
                # Replace dots with underscores and clean the value
                key = key.replace('.', '_')
                # Clean the value to be filename-safe and add spaces around words
                clean_value = re.sub(r'[^\w\s-]', '', value)
                # Replace multiple spaces with single space and strip
                clean_value = ' '.join(clean_value.split())
                # Replace spaces with underscores for filename
                clean_value = clean_value.replace(' ', '_')
                
                if key == 'default_search':
                    parts.append(f"search_{clean_value}")
                else:
                    parts.append(f"{key}_{clean_value}")
    
    # Handle sort parameters
    if 'sort' in query_params:
        sort_params = query_params['sort'][0].split(',')
        for s in sort_params:
            if s.startswith('-'):
                parts.append(f"sort_{s[1:].replace('.', '_')}_desc")
            else:
                parts.append(f"sort_{s.replace('.', '_')}_asc")
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Combine all parts
    filename = '__'.join(parts) if parts else 'openalex_query'
    filename = f"{filename}__{timestamp}"
    
    # Ensure filename is not too long (max 255 chars is common filesystem limit)
    if len(filename) > 255:
        filename = filename[:251]  # leave room for potential extension
    
    return filename

def get_records_from_dois(doi_list, block_size=50):
    """
    Download OpenAlex records for a list of DOIs in blocks.
    Args:
        doi_list (list): List of DOIs (strings)
        block_size (int): Number of DOIs to fetch per request (default 50)
    Returns:
        pd.DataFrame: DataFrame of OpenAlex records
    """
    from pyalex import Works
    from tqdm import tqdm
    all_records = []
    for i in tqdm(range(0, len(doi_list), block_size)):
        sublist = doi_list[i:i+block_size]
        doi_str = "|".join(sublist)
        try:
            record_list = Works().filter(doi=doi_str).get(per_page=block_size)
            all_records.extend(record_list)
        except Exception as e:
            print(f"Error fetching DOIs {sublist}: {e}")
    return pd.DataFrame(all_records)