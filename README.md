# :sound: Spatial Audio Generation

[![arXiv](https://img.shields.io/badge/arXiv-2301.12503-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2301.12503)

<!-- # [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer) -->

**Generate speech, sound effects, music and beyond.**

This repo currently support: 

- **Text-to-Audio Generation**: Generate audio given text input.
- **Audio-to-Audio Generation**: Given an audio, generate another audio that contain the same type of sound. 
- **Text-guided Audio-to-Audio Style Transfer**: Transfer the sound of an audio into another one using the text description.

<hr>

## Important tricks to make your generated audio sound better
1. Try to provide more hints to AudioLDM, such as using more adjectives to describe your sound (e.g., clearly, high quality) or make your target more specific (e.g., "water stream in a forest" instead of "stream"). This can make sure AudioLDM understand what you want. 
2. Try to use different random seeds, which can affect the generation quality significantly sometimes.
3. It's best to use general terms like 'man' or 'woman' instead of specific names for individuals or abstract objects that humans may not be familiar with.

# Hardware requirement
- GPU with 8GB of dedicated VRAM
- A system with a 64-bit operating system (Windows 7, 8.1 or 10, Ubuntu 16.04 or later, or macOS 10.13 or later) 16GB or more of system RAM

## Reference
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution. 

> https://github.com/LAION-AI/CLAP

> https://github.com/CompVis/stable-diffusion

> https://github.com/v-iashin/SpecVQGAN 

> https://github.com/toshas/torch-fidelity


We build the model with data from AudioSet, Freesound and BBC Sound Effect library. We share this demo based on the UK copyright exception of data for academic research. 

<!-- This code repo is strictly for research demo purpose only. For commercial use please contact us. -->
