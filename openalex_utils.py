import numpy as np
from urllib.parse import urlparse, parse_qs
from pyalex import Works
import pandas as pd

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
    
    # Handle sort
    if 'sort' in query_params:
        sort_params = query_params['sort'][0].split(',')
        for s in sort_params:
            if s.startswith('-'):
                query = query.sort(**{s[1:]: 'desc'})
            else:
                query = query.sort(**{s: 'asc'})
    
    # Handle other parameters
    params = {}
    for key in ['page', 'per-page', 'sample', 'seed']:
        if key in query_params:
            params[key] = query_params[key][0]
    
    return query, params

def invert_abstract(inv_index):
    """Reconstruct abstract from inverted index."""
    if inv_index is not None:
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))
    else:
        return ' '

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
    
    Args:
    records (list): List of OpenAlex record dictionaries
    
    Returns:
    pandas.DataFrame: Processed DataFrame with abstracts, publications, and titles
    """
    records_df = pd.DataFrame(records)
    records_df['abstract'] = [invert_abstract(t) for t in records_df['abstract_inverted_index']]
    records_df['parsed_publication'] = [get_pub(x) for x in records_df['primary_location']]
    
    records_df['parsed_publication'] = records_df['parsed_publication'].fillna(' ')
    records_df['abstract'] = records_df['abstract'].fillna(' ')
    records_df['title'] = records_df['title'].fillna(' ')
    
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