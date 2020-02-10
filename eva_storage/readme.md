## Eva Storage Guide for cs8803

- This directory contains all the code related to compression and indexing


## Indexing
- Running eva_storage/evaluation/evaluate_umnet.ipynb will give a good idea of how the indexing pipeline works

Basically, the indexing pipeline is:
1. Load the video
2. Preprocess the video by generating reference segmentation masks using background subtraction algorithm
3. Train the network / Load frozen UNet for better segmentation mask training
4. Postprocess the video (cv filtering)
5. Create patches that are fed into another small autoencoder for additional dimensionality reduction (Irrelevant to your project)
6. Arrange them into a hash / dictionary format for retrieval (Irrelevant to your project)

## Compression (Irrelevant)
- Running eva_storage/evaluation/evaluate_compression.ipynb will give a good idea of how the things are supposed to run (But not sure if it works 11/7 because I moved it from a different directory)



