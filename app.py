import time
print(f"Starting up: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Standard library imports
import os
from pathlib import Path
from datetime import datetime
from itertools import chain

# Third-party imports
import numpy as np
import pandas as pd
import torch
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import matplotlib.pyplot as plt
import tqdm
import colormaps
import matplotlib.colors as mcolors


import opinionated # for fonts
plt.style.use("opinionated_rc")

from sklearn.neighbors import NearestNeighbors


def is_running_in_hf_space():
    return "SPACE_ID" in os.environ

if is_running_in_hf_space():
    import spaces # necessary to run on Zero.

import datamapplot
import pyalex

# Local imports
from openalex_utils import (
    openalex_url_to_pyalex_query, 
    get_field,
    process_records_to_df,
    openalex_url_to_filename
)
from styles import DATAMAP_CUSTOM_CSS
from data_setup import (
    download_required_files,
    setup_basemap_data,
    setup_mapper,
    setup_embedding_model,
    
)

# Configure OpenAlex
pyalex.config.email = "maximilian.noichl@uni-bamberg.de"

print(f"Imports completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# FastAPI setup
app = FastAPI()
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Gradio configuration
gr.set_static_paths(paths=["static/"])

# Resource configuration
REQUIRED_FILES = {
    "100k_filtered_OA_sample_cluster_and_positions_supervised.pkl": 
        "https://huggingface.co/datasets/m7n/intermediate_sci_pickle/resolve/main/100k_filtered_OA_sample_cluster_and_positions_supervised.pkl",
    "umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl":
        "https://huggingface.co/datasets/m7n/intermediate_sci_pickle/resolve/main/umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl"
}
BASEMAP_PATH = "100k_filtered_OA_sample_cluster_and_positions_supervised.pkl"
MAPPER_PARAMS_PATH = "umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl"
MODEL_NAME = "m7n/discipline-tuned_specter_2_024"

# Initialize models and data
start_time = time.time()
print("Initializing resources...")

download_required_files(REQUIRED_FILES)
basedata_df = setup_basemap_data(BASEMAP_PATH)
mapper = setup_mapper(MAPPER_PARAMS_PATH)
model = setup_embedding_model(MODEL_NAME)

print(f"Resources initialized in {time.time() - start_time:.2f} seconds")



# Setting up decorators for embedding on HF-Zero:
def no_op_decorator(func):
    """A no-op (no operation) decorator that simply returns the function."""
    def wrapper(*args, **kwargs):
        # Do nothing special
        return func(*args, **kwargs)
    return wrapper

# Decide which decorator to use based on environment
decorator_to_use = spaces.GPU(duration=60) if is_running_in_hf_space() else no_op_decorator

    
@decorator_to_use
def create_embeddings(texts_to_embedd):
    """Create embeddings for the input texts using the loaded model."""
    return model.encode(texts_to_embedd, show_progress_bar=True, batch_size=192)

def predict(text_input, sample_size_slider, reduce_sample_checkbox, sample_reduction_method, 
           plot_time_checkbox, locally_approximate_publication_date_checkbox, 
           download_csv_checkbox, download_png_checkbox, progress=gr.Progress()):
    """
    Main prediction pipeline that processes OpenAlex queries and creates visualizations.
    
    Args:
        text_input (str): OpenAlex query URL
        sample_size_slider (int): Maximum number of samples to process
        reduce_sample_checkbox (bool): Whether to reduce sample size
        sample_reduction_method (str): Method for sample reduction ("Random" or "Order of Results")
        plot_time_checkbox (bool): Whether to color points by publication date
        locally_approximate_publication_date_checkbox (bool): Whether to approximate publication date locally before plotting.
        progress (gr.Progress): Gradio progress tracker
    
    Returns:
        tuple: (link to visualization, iframe HTML)
    """
    # Check if input is empty or whitespace
    print(f"Input: {text_input}")
    if not text_input or text_input.isspace():
        error_message = "Error: Please enter a valid OpenAlex URL in the 'OpenAlex-search URL'-field"
        return [
            error_message,  # iframe HTML
            gr.DownloadButton(label="Download Interactive Visualization", value='html_file_path', visible=False),  # html download
            gr.DownloadButton(label="Download CSV Data", value='csv_file_path', visible=False),  # csv download
            gr.DownloadButton(label="Download Static Plot", value='png_file_path', visible=False),  # png download
            gr.Button(visible=False)  # cancel button state
        ]

    
    # Check if the input is a valid OpenAlex URL

    
    
    start_time = time.time()
    print('Starting data projection pipeline')
    progress(0.1, desc="Starting...")

    # Query OpenAlex
    query_start = time.time()
    query, params = openalex_url_to_pyalex_query(text_input)
    
    filename = openalex_url_to_filename(text_input)
    print(f"Filename: {filename}")
    
    query_length = query.count()
    print(f'Requesting {query_length} entries...')
    
    records = []
    target_size = sample_size_slider if reduce_sample_checkbox and sample_reduction_method == "First n samples" else query_length
    
    
    should_break = False
    for page in query.paginate(per_page=200,n_max=None):
        for record in page:
            records.append(record)
            progress(0.1 + (0.2 * len(records) / target_size), desc="Getting queried data...")
           # print(len(records))
            if reduce_sample_checkbox and sample_reduction_method == "First n samples" and len(records) >= target_size:
                should_break = True
                break
        if should_break:
            break
    
    print(f"Query completed in {time.time() - query_start:.2f} seconds")

    # Process records
    processing_start = time.time()
    records_df = process_records_to_df(records)
    
    if reduce_sample_checkbox and sample_reduction_method != "All":
        sample_size = min(sample_size_slider, len(records_df))        
        if sample_reduction_method == "n random samples":
            records_df = records_df.sample(sample_size)
        elif sample_reduction_method == "First n samples":
            records_df = records_df.iloc[:sample_size]
    print(f"Records processed in {time.time() - processing_start:.2f} seconds")
    
    # Create embeddings
    embedding_start = time.time()
    progress(0.3, desc="Embedding Data...")
    texts_to_embedd = [f"{title} {abstract}" for title, abstract 
                      in zip(records_df['title'], records_df['abstract'])]
    embeddings = create_embeddings(texts_to_embedd)
    print(f"Embeddings created in {time.time() - embedding_start:.2f} seconds")

    # Project embeddings
    projection_start = time.time()
    progress(0.5, desc="Project into UMAP-embedding...")
    umap_embeddings = mapper.transform(embeddings)
    records_df[['x','y']] = umap_embeddings
    print(f"Projection completed in {time.time() - projection_start:.2f} seconds")

    # Prepare visualization data
    viz_prep_start = time.time()
    progress(0.6, desc="Preparing visualization data...")
    
    basedata_df['color'] = '#ced4d211'
    
    if not plot_time_checkbox:
        records_df['color'] = '#5e2784'
    else:
        cmap = colormaps.haline
        if not locally_approximate_publication_date_checkbox:
            # Create color mapping based on publication years
            years = pd.to_numeric(records_df['publication_year'])
            norm = mcolors.Normalize(vmin=years.min(), vmax=years.max())
            records_df['color'] = [mcolors.to_hex(cmap(norm(year))) for year in years]
            
        else:
            n_neighbors = 10  # Adjust this value to control smoothing
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(umap_embeddings)
            distances, indices = nn.kneighbors(umap_embeddings)

            # Calculate local average publication year for each point
            local_years = np.array([
                np.mean(records_df['publication_year'].iloc[idx])
                for idx in indices
            ])
            norm = mcolors.Normalize(vmin=local_years.min(), vmax=local_years.max())
            records_df['color'] = [mcolors.to_hex(cmap(norm(year))) for year in local_years]
                        
            

    stacked_df = pd.concat([basedata_df, records_df], axis=0, ignore_index=True)
    stacked_df = stacked_df.fillna("Unlabelled")
    stacked_df['parsed_field'] = [get_field(row) for ix, row in stacked_df.iterrows()]
    extra_data = pd.DataFrame(stacked_df['doi'])
    print(f"Visualization data prepared in {time.time() - viz_prep_start:.2f} seconds")
    
    # Create and save plot
    plot_start = time.time()
    progress(0.7, desc="Creating plot...")
    # Create a solid black colormap
    black_cmap = mcolors.LinearSegmentedColormap.from_list('black', ['#000000', '#000000'])
    
    
    plot = datamapplot.create_interactive_plot(
        stacked_df[['x','y']].values,
                                np.array(stacked_df['cluster_2_labels']),
        np.array(['Unlabelled' if pd.isna(x) else x for x in stacked_df['parsed_field']]),
        
        hover_text=[str(row['title']) for ix, row in stacked_df.iterrows()],
        marker_color_array=stacked_df['color'],
        use_medoids=True,
        width=1000,
        height=1000,
        point_radius_min_pixels=1,
        text_outline_width=5,
        point_hover_color='#5e2784',
        point_radius_max_pixels=7,
        cmap=black_cmap,
        #color_label_text=False,
        font_family="Roboto Condensed",
        font_weight=600,
        tooltip_font_weight=600,
        tooltip_font_family="Roboto Condensed",
        extra_point_data=extra_data,
        on_click="window.open(`{doi}`)",
        custom_css=DATAMAP_CUSTOM_CSS,
        initial_zoom_fraction=.8,
        enable_search=False,
        offline_mode=False
    )

    # Save plot
    html_file_name = f"{filename}.html"
    html_file_path = static_dir / html_file_name
    plot.save(html_file_path)
    print(f"Plot created and saved in {time.time() - plot_start:.2f} seconds")

    
   
    # Save additional files if requested
    csv_file_path = static_dir / f"{filename}.csv"
    png_file_path = static_dir / f"{filename}.png"
    
    if download_csv_checkbox:
        # Export relevant columns
        export_df = records_df[['title', 'abstract', 'doi', 'publication_year', 'x', 'y']]
        export_df.to_csv(csv_file_path, index=False)
        
    if download_png_checkbox:
        png_start_time = time.time()
        print("Starting PNG generation...")

        # Sample and prepare data
        sample_prep_start = time.time()
        sample_to_plot = basedata_df#.sample(20000)
        labels1 = np.array(sample_to_plot['cluster_2_labels'])
        labels2 = np.array(['Unlabelled' if pd.isna(x) else x for x in sample_to_plot['parsed_field']])
        
        ratio = 0.6
        mask = np.random.random(size=len(labels1)) < ratio
        combined_labels = np.where(mask, labels1, labels2)
        
        # Get the 30 most common labels
        unique_labels, counts = np.unique(combined_labels, return_counts=True)
        top_30_labels = set(unique_labels[np.argsort(counts)[-50:]])
        
        # Replace less common labels with 'Unlabelled'
        combined_labels = np.array(['Unlabelled' if label not in top_30_labels else label for label in combined_labels])
        
        colors_base = ['#536878' for _ in range(len(labels1))]
        print(f"Sample preparation completed in {time.time() - sample_prep_start:.2f} seconds")

        # Create main plot
        print(sample_to_plot[['x','y']].values)
        print(combined_labels)
        
        main_plot_start = time.time()
        fig, ax = datamapplot.create_plot(
            sample_to_plot[['x','y']].values,
            combined_labels,
            label_wrap_width=12,
            label_over_points=True,
            dynamic_label_size=True,
            use_medoids=True,
            point_size=2,
            marker_color_array=colors_base,
            force_matplotlib=True,
            max_font_size=12,
            min_font_size=4,
            min_font_weight=100,
            max_font_weight=300,
            font_family="Roboto Condensed",
            color_label_text=False, add_glow=False,
            highlight_labels=list(np.unique(labels1)),
            label_font_size=8,
            highlight_label_keywords={"fontsize": 12, "fontweight": "bold", "bbox":{"boxstyle":"circle", "pad":0.75,'alpha':0.}},
        )
        print(f"Main plot creation completed in {time.time() - main_plot_start:.2f} seconds")

        # Time-based visualization
        scatter_start = time.time()
        if plot_time_checkbox:
            if locally_approximate_publication_date_checkbox:
                scatter = plt.scatter(
                    umap_embeddings[:,0],
                    umap_embeddings[:,1],
                    c=local_years,
                    cmap=colormaps.haline,
                    alpha=0.8,
                    s=5
                )
            else:
                years = pd.to_numeric(records_df['publication_year'])
                scatter = plt.scatter(
                    umap_embeddings[:,0],
                    umap_embeddings[:,1],
                    c=years,
                    cmap=colormaps.haline,
                    alpha=0.8,
                    s=5
                )
            plt.colorbar(scatter, shrink=0.5, format='%d')
        else:
            scatter = plt.scatter(
                umap_embeddings[:,0],
                umap_embeddings[:,1],
                c=records_df['color'],
                alpha=0.8,
                s=5
            )
        print(f"Scatter plot creation completed in {time.time() - scatter_start:.2f} seconds")

        # Save plot
        save_start = time.time()
        plt.axis('off')
        png_file_path = static_dir / f"{filename}.png"
        plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saving completed in {time.time() - save_start:.2f} seconds")
        
        print(f"Total PNG generation completed in {time.time() - png_start_time:.2f} seconds")







    progress(1.0, desc="Done!")
    print(f"Total pipeline completed in {time.time() - start_time:.2f} seconds")
    
    iframe = f"""<iframe src="/static/{html_file_name}" width="100%" height="1000px"></iframe>"""
    
    # Return iframe and download buttons with appropriate visibility
    return [
        iframe,
        gr.DownloadButton(label="Download Interactive Visualization", value=html_file_path, visible=True, variant='secondary'),
        gr.DownloadButton(label="Download CSV Data", value=csv_file_path, visible=download_csv_checkbox, variant='secondary'),
        gr.DownloadButton(label="Download Static Plot", value=png_file_path, visible=download_png_checkbox, variant='secondary'),
        gr.Button(visible=False)  # Return hidden state for cancel button
    ]


theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Roboto Condensed"), "ui-sans-serif", "system-ui", "sans-serif"],
    text_size="lg",
).set(
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="#f3f4f6",
    button_secondary_border_color="black",
    button_secondary_text_color="black",
    button_border_width="2px",

)


# Gradio interface setup
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("""
    <div style="max-width: 100%; margin: 0 auto;">
    <br>
    
    # OpenAlex Mapper
    
    OpenAlex Mapper is a way of projecting search queries from the amazing OpenAlex database on a background map of randomly sampled papers from OpenAlex, which allows you to easily investigate interdisciplinary connections. OpenAlex Mapper was developed by Maximilian Noichl and Andrea Loettgers at the Possible Life project.

    To use OpenAlex Mapper, first head over to [OpenAlex](https://openalex.org/) and search for something that interests you. For example, you could search for all the papers that make use of the [Kuramoto model](https://openalex.org/works?page=1&filter=default.search%3A%22Kuramoto%20Model%22), for all the papers that were published by researchers at [Utrecht University in 2019](https://openalex.org/works?page=1&filter=authorships.institutions.lineage%3Ai193662353,publication_year%3A2019), or for all the papers that cite Wittgenstein's [Philosophical Investigations](https://openalex.org/works?page=1&filter=cites%3Aw4251395411). Then you copy the URL to that search query into the OpenAlex search URL box below and click "Run Query." It will take a moment to download all of these records from OpenAlex and embed them on our interactive map. After a little time, that map will appear and be available for you to interact with and download. You can find more explanations in the FAQs below.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                run_btn = gr.Button("Run Query", variant='primary')
                cancel_btn = gr.Button("Cancel", visible=False, variant='secondary')
            
            # Create separate download buttons
            html_download = gr.DownloadButton("Download Interactive Visualization", visible=False, variant='secondary')
            csv_download = gr.DownloadButton("Download CSV Data", visible=False, variant='secondary')
            png_download = gr.DownloadButton("Download Static Plot", visible=False, variant='secondary')

            text_input = gr.Textbox(label="OpenAlex-search URL",
                                    info="Enter the URL to an OpenAlex-search.")
            
            gr.Markdown("### Sample Settings")
            reduce_sample_checkbox = gr.Checkbox(
                label="Reduce Sample Size",
                value=True,
                info="Reduce sample size."
            )
            sample_reduction_method = gr.Dropdown(
                ["All", "First n samples", "n random samples"],
                label="Sample Selection Method",
                value="First n samples",
                info="How to choose the samples to keep."
            )
            sample_size_slider = gr.Slider(
                label="Sample Size",
                minimum=500,
                maximum=20000,
                step=10,
                value=1000,
                info="How many samples to keep.",
                visible=True
            )
            
            gr.Markdown("### Plot Settings")
            plot_time_checkbox = gr.Checkbox(
                label="Plot Time",
                value=True,
                info="Colour points by their publication date."
            )
            locally_approximate_publication_date_checkbox = gr.Checkbox(
                label="Locally Approximate Publication Date",
                value=True,
                info="Colour points by the average publicaion date in their area."
            )
            
            gr.Markdown("### Download Options")
            download_csv_checkbox = gr.Checkbox(
                label="Generate CSV Export",
                value=False,
                info="Export the data as CSV file"
            )
            download_png_checkbox = gr.Checkbox(
                label="Generate Static PNG Plot",
                value=False,
                info="Export a static PNG visualization. This will make things slower!"
            )
            
            
            
            
        with gr.Column(scale=2):
            html = gr.HTML(
                value='<div style="width: 100%; height: 1000px; display: flex; justify-content: center; align-items: center; border: 1px solid #ccc; background-color: #f8f9fa;"><p style="font-size: 1.2em; color: #666;">The visualization map will appear here after running a query</p></div>',
                label="Map", 
                show_label=True
            )
    gr.Markdown("""
    <div style="max-width: 100%; margin: 0 auto;">
    
    # FAQs
    
    ## Who made this?

    This project was developed by [Maximilian Noichl](https://maxnoichl.eu) (Utrecht University), in cooperation with Andrea Loettger and Tarja Knuuttila at the [Possible Life project](http://www.possiblelife.eu/), at the University of Vienna. If this project is useful in any way for your research, we would appreciate citation of **...**

    This project received funding from the European Research Council under the European Union's Horizon 2020 research and innovation programme (LIFEMODE project, grant agreement No. 818772).

    ## How does it work?

    The base map for this project is developed by randomly downloading 250,000 articles from OpenAlex, then embedding their abstracts using our [fine-tuned](https://huggingface.co/m7n/discipline-tuned_specter_2_024) version of the [specter-2](https://huggingface.co/allenai/specter2_aug2023refresh_base) language model, running these embeddings through [UMAP](https://umap-learn.readthedocs.io/en/latest/) to give us a two-dimensional representation, and displaying that in an interactive window using [datamapplot](https://datamapplot.readthedocs.io/en/latest/index.html). After the data for your query is downloaded from OpenAlex, it then undergoes the exact same process, but the pre-trained UMAP model from earlier is used to project your new data points onto this original map, showing where they would show up if they were included in the original sample. For more details, you can take a look at the method section of this paper: **...**

    ## I think I found a mistake in  the map.

    There are various considerations to take into account when working with this map:

    1. The language model we use is fine-tuned to separate disciplines from each other, but of course, disciplines are weird, partially subjective social categories, so what the model has learned might not always correspond perfectly to what you would expect to see.

    2. When pressing down a really high-dimensional space into a low-dimensional one, there will be trade-offs. For example, we see this big ring structure of the sciences on the map, but in the middle of the map there is a overly stretchedstring of bioinformaticsthat stretches from computer science at the bottom up to the life sciences clusters at the top. This is one of the areas where the UMAP algorithm had trouble pressing our high-dimensional dataset into a low-dimensional space. For more information on how to read a UMAP plot, I recommend looking into ["Understanding UMAP"](https://pair-code.github.io/understanding-umap/) by Andy Coenen & Adam Pearce.
    
    3. Finally, the labels we're using for the regions of this plot are created from OpenAlex's own labels of sub-disciplines. They give a rough indication of the papers that could be expected in this broad area of the map, but they are not necessarily the perfect label for the articles that are precisely below them. They are just located at the median point of a usually much larger, much broader, and fuzzier category, so they should always be taken with quite a big grain of salt.
    </div>
    """)

    def update_slider_visibility(method):
        return gr.Slider(visible=(method != "All"))

    sample_reduction_method.change(
        fn=update_slider_visibility,
        inputs=[sample_reduction_method],
        outputs=[sample_size_slider]
    )
    
    def show_cancel_button():
        return gr.Button(visible=True)
    
    def hide_cancel_button():
        return gr.Button(visible=False)

    # Update the run button click event
    run_event = run_btn.click(
        fn=show_cancel_button,
        outputs=cancel_btn,
        queue=False
    ).then(
        fn=predict,
        inputs=[text_input, sample_size_slider, reduce_sample_checkbox, 
                sample_reduction_method, plot_time_checkbox, 
                locally_approximate_publication_date_checkbox,
                download_csv_checkbox, download_png_checkbox],
        outputs=[html, html_download, csv_download, png_download, cancel_btn]
    )

    # Add cancel button click event
    cancel_btn.click(
        fn=hide_cancel_button,
        outputs=cancel_btn,
        cancels=[run_event],
        queue=False  # Important to make the button hide immediately
    )

# Mount and run app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)