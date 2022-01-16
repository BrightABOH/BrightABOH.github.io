---
layout: post
title:  "Harmattan through satellite lense"
date:   2022-01-15 22:00:00
categories: harmattan through satellite lense
permalink: /posts/Data-Science-and-remote-sensing-3
image: "https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/aerosol1.png?raw=true"
---

## Goal

## Introduction 

## Data description

In this tutorial, we will use the Sentinel -5Precursor Near Real Time (NRTI) UltraViolet Aerosol Index (UVAI) also known as the Aerosol Absorbing Index(AAI). The Sentinel-5Precursor is  part of the sentinel’s earth observation satellite mission launched by the European Space Agency collect remotely sensed atmospheric data mostly on air pollution

The AAI is defined as the difference between the ratio of measured reflectance at 354 and 388 nm and the ratio of simulated reflectances at these wavelength depending on the changes in Rayleigh wavelength scattering. 


Mathematically; 

<img src="https://latex.codecogs.com/svg.image?AAI&space;=&space;-100.\left&space;[&space;log_{10}\left&space;(&space;\frac{l_{354}}{l_{388}}&space;\right&space;)meas&space;-log_{10}\left&space;(&space;\frac{l_{354}}{l_{388}}&space;\right&space;)simu\right&space;]" title="AAI = -100.\left [ log_{10}\left ( \frac{l_{354}}{l_{388}} \right )meas -log_{10}\left ( \frac{l_{354}}{l_{388}} \right )simu\right ]" />

When the AAI is positive, it indicates the presence of UV-absorbing aerosols like dust and smoke. This index is useful for tracking the evolution of episodic aerosol plumes from dust outbreaks(such as harmattan), volcanic ash, etc.


