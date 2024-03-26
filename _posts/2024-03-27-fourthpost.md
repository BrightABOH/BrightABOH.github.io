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





## Image acquisition 

## Pixel replacement
