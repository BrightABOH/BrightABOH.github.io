---
layout: post
title: "Data Science and remote sensing: Water detection Part 1"
date: 2021-10-31 22:01:18
categories: website blog jekyll github
permalink: /posts/Data-Science-and-remote-sensing
---
**Introduction**

Having worked in the intersection of this space(data science and remote sensing) for a while, I wish to bring to the limelight the amazing benefits shrouded in the unison of these two fields. I am writting series of blog posts focusing on the intersections of these feilds, real world applications, tutorials etc.
To start with, let’s begin with an understanding of the two fields separately; Remote sensing and Data Science. I will describe Remote sensing as the study of the physical constituents of nature without having to make any physical contact with the objects. Therefore the needed information about these objects is collected from a distance e.g from a satellite, from an airplane, etc. Some of the information collected on these Earth objects includes vegetation, air, ocean, open water, oil reservoirs, etc. Data Science is an interdisciplinary field comprising domain knowledge, programming skills, and knowledge of mathematics and statistics to extract and identify hidden patterns within data sets including those collected through remote sensing.
Now that we have a firm understanding of what remote sensing and data science are, we will explore the interplay of the two. Our focus will be on how to identify and groundwater from satellite images using Google Earth Engine and its Python API. For easy content absorption, this blog is divided into two parts; the first part is dedicated to understanding the underlying concepts and the second part is the practical session.   The data used in this blog is Copernicus Sentinel 1 Synthetic Aperture Radar(SAR) obtained through Google Earth Engine(GEE) platform; a platform for the analysis and visualization of a large volume of geospatial datasets. A major challenge usually encountered working with satellite images is the computation requirements to process such data. To leverage the computational powers and the readily available satellite data, we use the GEE platform. Next, we define the location to collect data on. Our interest is in the Volta region of Ghana, please feel free to use any region of interest. In addition, you will need to create an account with GEE to have access to the platform. 

**Data Description**

Launched in 2014, Sentinel 1 is equipped with a constellation of two polar-orbiting satellites, operating all day and all night irrespective of the weather to provide a large amount of data suitable for environmental and security monitoring globally.  The Radar imaging technology onboard the SAR Sentinel 1 allows data acquisitions even in very hazy regions due to its ability to penetrate clouds. 

**Tools**

Python 3.9 is used as the main programming language to process and extract the dataset. Google Earth Engine platform through its Python API is used for accessing satellite images and for further analysis. 
Kindly follow through with the concluding part.

