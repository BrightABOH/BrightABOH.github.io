---
layout: post
title:  "Harmattan through satellite lense"
date:   2022-01-15 22:00:00
categories: harmattan through satellite lense
permalink: /posts/Data-Science-and-remote-sensing-3
image: "https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/aerosol1.png?raw=true"
---




## 1 Introduction 
Harmattan is  a north eastern (wind direction is mainly north east) trade dry and dusty winds that blows from the south of the Sahara towards the Gulf of Guinea.  These trade winds are considered to be seasonal as they blow from November till March . 
With such dry winds, the relative humidity usually hovers around <10% , temperatures usually fluctuates a lot during this period with cold night temperatures ranging from 9-20 Celsius. During the day however, temperatures could go as high as 30-39 Celsius depending on the location ! 

### 1.1 Harmattan haze
A well noticeable phenomena associated with harmattan is the presence of grey skies (hazy skies) which normally blocks the sun’s rays creating a  low visibility! This haze phenomenon is caused by the presence of huge dust particles  in the atmosphere!

### 1.2 Formation of harmattan haze and/or dust storm
When wind blows over the open desert area, it carries with it a lot of dust particles from the ground. The very small dust particles ranging from sizes 0.5 microns to 10 microns suspends in the air . These suspended particulate matter are transported over great distances by this wind . When the air movement is strong enough, it could lead to huge sand storms.  
During the harmattan season, there is a lot of dust in the atmosphere in many west African countries causing haze in the atmosphere among other effects. So which countries in Africa are mostly affected by this harmattan? These countries include; Niger,Nigeria Chad, Ghana, Burkina Faso, Benin, Mali Ivory Coast, Togo and Cameroon.
### 1.3 Effects of harmattan
Strong and  persistent harmattan can destroy crops due to the large dust layer on these leaves and lack of rainfall causing some famers to experience crop failures. 
The high amount of dust in the atmosphere limits visibility greatly in some West African countries. As a results, the aviation industry suffers some great loses as some have their flights  cancel or interrupted each year due to harmattan. 
Among the numerous effects of harmattan, public health implications of harmattan cannot be overemphasized. Some of the health concerns usually associated with this period are irritating eyes and lips, cracked lips. More serious concerns such as meningitis(Sahel meningitis), asthma and other respiratory problems are often associated with the harmattan season. Since the sizes of the particles are very small, they are easily absorbed into the respiratory truck and the lungs sometimes thereby causing very serious cardiovascular and other health disorders. For the purpose of further understanding these particles, they are further grouped based on their sizes with PM 2.5 (Particulate matter with diameter less than 2.5 micrometers) and PM 10 (Particulate matter with diameter less than 10 micrometers) generally considered fine inhalable particles and inhalable particles respectively. The above image provides some insight on the particles!
![Particle size](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/pm.jpeg?raw=true)


### 1.4 Other winds of Africa
It is worth mentioning that aside the harmattan winds, there are other major winds on the African continent . These include; Sirocco: Hot, dry wind that brings dust and from Tunisia and Algeria to the Mediterranean and Southen Europe. Ghibli: Hot, dry  south wind blowing from Libya towards Greece, khamsin, Haboob, Cape Doctor, Levanter etc.
           
## 2. Data description

In this tutorial, we will use the Sentinel -5Precursor Near Real Time (NRTI) UltraViolet Aerosol Index (UVAI) also known as the Aerosol Absorbing Index(AAI). The Sentinel-5Precursor is  part of the sentinel’s earth observation satellite mission launched by the European Space Agency collect remotely sensed atmospheric data mostly on air pollution

The AAI is defined as the difference between the ratio of measured reflectance at 354 and 388 nm and the ratio of simulated reflectances at these wavelength depending on the changes in Rayleigh wavelength scattering. 


Mathematically; 

<img src="https://latex.codecogs.com/svg.image?AAI&space;=&space;-100.\left&space;[&space;log_{10}\left&space;(&space;\frac{l_{354}}{l_{388}}&space;\right&space;)meas&space;-log_{10}\left&space;(&space;\frac{l_{354}}{l_{388}}&space;\right&space;)simu\right&space;]" title="AAI = -100.\left [ log_{10}\left ( \frac{l_{354}}{l_{388}} \right )meas -log_{10}\left ( \frac{l_{354}}{l_{388}} \right )simu\right ]" />

When the AAI is positive, it indicates the presence of UV-absorbing aerosols like dust and smoke. This index is useful for tracking the evolution of episodic aerosol plumes from dust outbreaks (such as harmattan), volcanic ash, etc.

For further reading, interested readers should contact the official documentation [here](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_AER_AI) 
## 3. Goal
 In this tutorial, we will monitor the progression of the trade winds responsible for the dust storms experienced during  harmattan period with the help of  satellite images.  In particular, we will monitor when these dust reaches Ghana from the Sahara desert (dust origin).
## 4. La crème de la crème
To start with, we import the necessary modules and packages!
```python
import ee
# Trigger the authentication flow.
ee.Authenticate()
# Initialize the library.
ee.Initialize()
#Import the neccesary packages
import geemap
import json
import os
import requests
```

Next be define the our area of interest. 
```python
AF_region = ee.Geometry.Polygon(
  [[[-18.698368046353494, 38.1446395611524],
    [-18.698368046353494, -36.16300755581617],
    [52.229366328646506, -36.16300755581617],
    [52.229366328646506, 38.1446395611524]]],
  None, False)
  
mask = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
africaBorder = mask.filter(ee.Filter.eq('wld_rgn', 'Africa'));
ghanaBorder = mask.filter(ee.Filter.eq('country_na', 'Ghana'))
```

```python
#Collect Sentinel 5P NRTI satellite images
dust = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
.select('absorbing_aerosol_index')\
.filterBounds(AF_region)\
#Save the images separately based on the filter date
.filterDate('2021-09-01','2021-09-30')
#.filterDate('2021-10-01','2021-10-31')
#.filterDate('2021-11-01','2021-11-30')
#.filterDate('2021-12-01','2021-12-31')
#.filterDate('2021-01-01','2022-01-15')

```
