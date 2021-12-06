---
layout: post
title:  "Data Science and remote sensing: Water detection Part 2"
date:   2021-12-02 11:15:17
categories: website blog jekyll github
permalink: /posts/Data-Science-and-remote-sensing-2
image: "https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/RGB_water.png?raw=true"
---

Welcome back! This is the continuation of the previous blog on 'Detection of water' using satellite images. Read [here](https://brightaboh.github.io/posts/Data-Science-and-remote-sensing) if you haven't done so already. As indicated in the previous blog, one will need Google Earth Engine(GEE) account in order to follow through the rest of this tutorial. Register for a free GEE account [here](https://earthengine.google.com) 

Next we load (import) all the necesary python modules into our jupyter notebook
![Dependent modules](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/modules.png?raw=true)

The next is to define the Region Of Interest(ROI),i.e the area we want to collect satellite images over. To define your own ROI, you can use this [tool](http://geojson.io/#map=2/20.0/0.0). In case, you have your own shapefiles, here is the good place to upload them. As seen from the block codes below,there are few things worth noting; 1) convertion of geojson files into coordinate system that is understood by GEE, 2) specifing the satellite images to use for our analysis 3) filtering this satellite images based on filters such as date, transmitter polarization. We use the Synthetic Aperture Rader images (such as Sentinel 1)  as it has some advantages over optical satellite images. Observing from the out in the cell, we can see that there are 121 images available for our ROI given the necessary filter
![Location](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/geojson.png?raw=true)
We look at the first image available in the entire collection (121). To do that we create a map of our ROI, display as a layer the first image 
The following block of codes creates a map using geemap of our ROI, then display the first image from the Sentinel 1 collection.
![Firstmap](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/water_map.png?raw=true)
The output of the above codes is shown bellow
![Firstimage](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/firstimage.png?raw=true)

A major challenge working with SAR images is the speckle noise effect appparent in these images which degrades in the quality of the images. Evident in the SAR image obtained  are very bright tiny dots when you zoomed into the water body for example in this tutorial. Due to the very bright colors that are associated with objects such land, SAR images turn to suffer from misclassifications when this speckle noise is not treated. To remove the speckle noise from our images and make them ready for classification, we implemented the mean focal filter. This looks at the pixel from the entire image collection and takes the mean. Other approaches that could be used includes the median filter, the Lee filter, refined Lee filter etc.
The implementation of the speckle filtering is shown below;
![Filtermean](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/speckle.png)
