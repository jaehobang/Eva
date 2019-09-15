# Relevant Components
0. Input Formatting Module
1. Decompression Module
2. CAE Network
3. Cluster Module
4. Custom Compression Module
5. Video Storage Module


### Input Formatting Module - done
This module performs input check
As video input, we expect decompressed format or compressed format

Function names: save_video(), help()

Input: will be filename or folder name
We expect the input to be a filename if compressed, we expect it to be folder name if decompressed

Output: np.array where shape will be n_samples, height, width, channels

If decompressed format, we expect a file structure where image files are located in the lowest level
We will 'walk' through all the folders given to generate the video where each folder level will be sorted

If compressed format, we will call the Decompression Module

After we make everything into the decompressed format and load all the images into np array, we will perform various type checks
1. Since images, we need the matrix be of type int8
2. We need to make sure all the frame height, width matches (or at least we have to load them that way)


### Decompression Module - done
Decompresses the given video

__Function names__: run(), help()

Input will be full_path_to_video + video_name
Output will be np.array where shape will be n_samples, height, width, channels

Help function will give a manual of what this class does

Will do some checks to make sure this is a compressed format.
Also will need to output the possible video types this can take


### CAE Network
Resize the image matrix to fit the network parameters

save the features as features_

save the compressed formates as images_compressed_

save the decoded formats as images_decoded_


__Function names__: Train(), Evaluate()

Train Input: np.array where shape is n_samples, height, width, channels

Train Output: compressed_
This function will also need to do input shaping into the appropriate format that the network can take

Evaluate Input: np.array where shape is n_samples, height, width, channels
Evaluate Output: compressed_

### Cluster Module
This module performs clustering on the compressed formats

__Function Names__: run(), help()

run Input: np.array where shape is n_samples, compressed_features
run Output: cluster_labels_


### Custom Compression Module
We will not implement this module for the following reason.
1. Our original goal was to keep a compressed version only. However, because h.264 applies various optimizations along with I,B,P
it would be hard to get access to only the 'I' frames of the videos. 
This means we will have to decompress anyways when doing analysis.

2. If we were to apply only the I,B,P frame optimization, we have no theoretical guarantee that our newly compressed
videos will have a lower capacity compared to the traditional method. This means, it is harder for us to justify all the workflow of
going through to making a new compression method.

3. So the final method we thought of is to keep the original file as compressed using the traditional h.264 compression standard, while
keeping a table record of the important frames and the corresponding image file for them.
That way, we can caccess them for analysis whenever needed.

```
Pick out representative frames (do this randomly for now) from each cluster, Use them as I frames, other frames as B,P frames
Make a .mp4 output

__Function Names__: run(), help()

run Input: original_image_matrix, cluster_labels_
run Output: .mp4 video file
```


### Video Storage Module
This will save the keyframes and the corresponding xml file

Save videos, important frames, and retrieve videos, important frames when needed
Also, need to save and retrieve indexes (save them into a xml or txt file somehow)

__Function Names__: save(), get()

save Input: filename, actual_file, explanation(whether video, I frames, indexes)
save Output: success

get Input: filename, explanation(video, I frames, indexes)
get Output: the actual file (if video: the mp4 format, if I frames: np.array, if indexes: whatever format Ujjwal and Shreya defines)
