---
layout: post
title: "Satellite imagery processing for DL application"
date: 2024-03-25 22:01:18
categories: Deep learning prediction
permalink: /posts/Cloudy Pixel Replacement
---
## Introduction
Remember the pain you go through working with very cloudy satellite images? Where lowering the cloud percentage would mean that you will have little or no images to work with? Moreso painful and frustrating when you build a very good deep-learning model only to realize that the inference component is extra challenging because the single image to predict with is almost always cloudy. This is even more prevalent when you are working with optical imageries such as Sentinel 2.

A recent project to develop a deep learning model to predict and estimate cropland area from freely available optical satellites further  highlighted the need to design a solution.  

In this blog post, I will share with you how to identify cloudy pixels and replace them with alternative satellite images.
Overall, the combination of high spatial resolution, multi-spectral capabilities, frequent coverage, open data access, and cloud cover monitoring makes Sentinel-2 a preferred choice for agriculture and land use applications, facilitating more informed decision-making and sustainable land management practices.
Despite the advantages of using Sentinel 2, this optical imagery suffers  in areas (locations) where there are a lot of cloud covers.




## Image acquisition 
For this tutorial, the ROI will be located in Rwanda. Rwanda's tropical climate, topography, proximity to the ITCZ, seasonal variation, and potential impacts of climate change contribute to the prevalence of cloud cover in the region making it ideal for this tutorial. We use Sentinelhub to access freely available Sentinel 2 and Landsat 8 and(or) 9. To register for Sentinelhub, head over [here](https://www.sentinel-hub.com), and create a client ID and client secret for your account.  

Now, in the code environment, import the necessary modules. 
```
from sentinelsat import SentinelAPI
from datetime import date

from shapely.ops import unary_union

import rasterio
import geopandas as gpd
from shapely.geometry import shape
from sentinelhub import BBox, CRS, DataCollection, SentinelHubRequest, MimeType, SHConfig
```

Set your Sentinel Hub credentials
```
config = SHConfig()
config.sh_client_id = 'Add your Sentinel Hub instance ID'
config.sh_client_secret = 'Add your Sentinel Hub client secret'
```
Next, we set the date range for which the images will be downloaded, and the location over which the image will be downloaded
```
Set the path to your shapefile
shapefile_path = "/path/to/shapefile.shp/"

#Set the date range for the Sentinel-2 image search
start_date = date(2024, 1, 1)

end_date = date(2024, 3, 30)
#image size to be downloaded
image_size = (512, 512)
# Define the mean and standard deviation values for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
Set up Sentinel Hub API including the bands to download
```
#Sentinel 2 bands to be downloaded
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "SCL", "B11", "B12","SCL"]

#Sentinel 1 bands to be downloaded
bands_s1 = ['VV', 'VH']
#Landsat 8,9 bands to be downloaded
l_band = ["B02", "B03", "B04","B05"]

api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12","SCL"],
        output: [
            { id: "B01", bands: 1, sampleType: SampleType.AUTO },
            { id: "B02", bands: 1, sampleType: SampleType.AUTO },
            { id: "B03", bands: 1, sampleType: SampleType.AUTO },
            { id: "B04", bands: 1, sampleType: SampleType.AUTO },
            { id: "B05", bands: 1, sampleType: SampleType.AUTO },
            { id: "B06", bands: 1, sampleType: SampleType.AUTO },
            { id: "B07", bands: 1, sampleType: SampleType.AUTO },
            { id: "B08", bands: 1, sampleType: SampleType.AUTO },
            { id: "B8A", bands: 1, sampleType: SampleType.AUTO },
            { id: "B09", bands: 1, sampleType: SampleType.AUTO },
            { id: "B11", bands: 1, sampleType: SampleType.AUTO },
            { id: "B12", bands: 1, sampleType: SampleType.AUTO },
            { id: "RGB", bands: 3, sampleType: SampleType.AUTO },
            { id: "RGBN", bands: 4, sampleType: SampleType.AUTO },
            { id: "TCI", bands: 3, sampleType: SampleType.AUTO },
            { id: "NDVI", bands: 1, sampleType: SampleType.FLOAT32 },  // NDVI band
            { id: "SAVI", bands: 3, sampleType: SampleType.FLOAT32 },
            { id: "SCL", bands: 3, sampleType: SampleType.FLOAT32 },  
           
        ]
    };
}
# Define any vegetation indices you are interested in
function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04);
    
    // Calculate SAVI
    L = 0.5;  // Soil brightness correction factor (adjust as needed)
    savi = ((samples.B08 - samples.B04) / (samples.B08 + samples.B04 + L)) * (1 + L);

    return {
        B01: [samples.B01],
        B02: [samples.B02],
        B03: [samples.B03],
        B04: [samples.B04],
        B05: [samples.B05],
        B06: [samples.B06],
        B07: [samples.B07],
        B08: [samples.B08],
        B8A: [samples.B8A],
        B09: [samples.B09],
        B11: [samples.B11],
        B12: [samples.B12],
        RGB: [2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02],
        RGBN: [samples.B04, samples.B03, samples.B02, samples.B08],
        TCI: [3*samples.B04, 3*samples.B03, 3*samples.B02],
        NDVI: [ndvi],  
        SAVI: [savi],
        SCL: [samples.SCL],
    };
}
"""
api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript_s1 = """
//VERSION=3
function setup() {
    return {
        input: ["VV", "VH"],
        output: [
            { id: "VV", bands: 1, sampleType: SampleType.AUTO },
            { id: "VH", bands: 1, sampleType: SampleType.AUTO },
            { id: "RGB", bands: 3, sampleType: SampleType.AUTO }
            
        ],
        visualization: {
            bands: ["VV", "VH"],
            min: [-25,-25], // Adjust these values based on your data distribution
            max: [5,5], // Adjust these values based on your data distribution
        }
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    // Adjust the coefficients for a natural color representation
    ratio = samples.VH-samples.VV
    rgb_ratio = samples.VH+ samples.VV+ratio
    red = samples.VH;
    green = samples.VV;
    blue = rgb_ratio;
    return {
        VH: [red],
        VV: [green],
        RGB: [red, green, blue] 
    };
}
"""

api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript_l8 = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04","B05"], // Bands for true color and NIR
        output: [
            { id: "rgb", bands: 3,  sampleType: SampleType.AUTO}, // True color RGB
            { id: "ndvi", bands: 3,  sampleType: SampleType.AUTO} // NDVI
        ]
        
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    // Calculate NDVI
        ndvi = (samples.B05 - samples.B04) / (samples.B05 + samples.B04);

    // Return true color RGB and NDVI values
    return {
        rgb: [2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02], // True color RGB
        ndvi: [ndvi] // NDVI
    };
}
```
Define python function to download the satellite images which makes a call to the Sentinel Hub APIs defined above

```
# Function to download Sentinel-2 images using sentinelhub
def download_sentinel_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')

    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area

    # Calculate the bounding box of the union of all geometries in the shapefile
    shapefile_union = unary_union(gdf['geometry'])
    bbox = BBox(bbox=shape(shapefile_union).bounds, crs=CRS(common_crs))

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']


        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'sentinel2'),
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25
                )
            ],
            responses=[
                SentinelHubRequest.output_response('RGB', MimeType.TIFF),
                SentinelHubRequest.output_response('SCL', MimeType.TIFF)
            ],
            bbox=bbox,
            size=image_size,
            config=config
        )

        try:
            request.save_data()


            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")
    return total_area_shapefile_hectares


def download_landsat_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')


    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area

    # Sum the areas to get the total area of the shapefile
    total_area_shapefile_m2 = gdf['area_m2'].sum()
    # Convert total area to hectares
    total_area_shapefile_hectares = total_area_shapefile_m2 / 10000

    # Convert total area to square kilometers
    total_area_shapefile_square_km = total_area_shapefile_hectares / 1000
    print("projection information")
    # Print the total area in square kilometers
    print(f"Total area of the shapefile: {total_area_shapefile_square_km:.2f} square kilometers")



    # Print the total area
    #print(f"Total area of the shapefile: {total_area_shapefile_m2:.2f} square meters")
    #print(f"Total area of the shapefile: {total_area_shapefile_hectares:.2f} square hectares")

    # Calculate the bounding box of the union of all geometries in the shapefile
    shapefile_union = unary_union(gdf['geometry'])
    bbox = BBox(bbox=shape(shapefile_union).bounds, crs=CRS(common_crs))

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']


        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'landsat'),
            evalscript=evalscript_l8,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.LANDSAT_OT_L1,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25

                )
            ],
            responses=[
                SentinelHubRequest.output_response('rgb', MimeType.TIFF),
                SentinelHubRequest.output_response('ndvi', MimeType.TIFF)
            ],
            bbox=bbox,
            size=image_size,
            config=config
        )

        try:
            request.save_data()


            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")
    return total_area_shapefile_hectares
    #def shapefile_wihtout_water():
      #  Land_without_water = total_area_shapefile_square_km - total_water_area
      #  return Land_without_water

def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_path)
```

We may want to create patches from very large shapes. To achieve this, the below code block is helpful

```
def preprocess_patch(patch):
    # Convert to RGB format
    patch_rgb = np.transpose(patch, (1, 2, 0))
    # Convert to PIL Image
    patch_image = Image.fromarray((patch_rgb * 255).astype(np.uint8))
    # Resize the image to match the model's input size
    patch_image = patch_image.resize((512, 512))
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # Convert to PyTorch tensor
    patch_tensor = transform(patch_image)
    # Add a batch dimension
    patch_tensor = patch_tensor.unsqueeze(0)
    return patch_tensor

def create_patches_from_single_image(image_path, patch_size=512):
    num_patches_total = 0
    total_predictions = np.zeros((1, 1, 512, 512))

    try:
        with rasterio.open(image_path) as src:
            num_patches = 0

            for i in range(0, src.width, patch_size):
                for j in range(0, src.height, patch_size):
                    window = rasterio.windows.Window(i, j, patch_size, patch_size)
                    patch = src.read(window=window)

                    # TODO: Process the patch as needed (e.g., save it to disk)
                    processed_patch = preprocess_patch(patch)

                    # Make predictions on the processed patch
                    with torch.no_grad():
                        predictions_patch = loaded_model(processed_patch.to(device))
                        predictions_patch = torch.sigmoid(predictions_patch)
                        predictions_patch = predictions_patch.cpu().numpy()

                        # Accumulate the predictions
                        total_predictions += predictions_patch

                    # For now, just print the patch information
                    print(f"Patch {num_patches + 1}: {window}")

                    num_patches += 1
                    num_patches_total += 1

            print(f"Number of patches created: {num_patches}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening the image at {image_path}: {e}")

    return num_patches_total

```

![Particle size](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/landast.jpg?raw=true)

## Pixel replacement
