# What you see is what you hear (WYSIWYH)

## Goal

Transform an image to audio in real time. This means that you can literally **hear what you see**. The audio should contain as much of the information of the image as possible. 

## Example (recommended with audio turned on)

On the top left you see the original video. On the bottom there's the audio spectrum, which is the output of the *recursively structuring* encoder.  The audio spectrum goes from 1 Hz on the left to 5000 Hz on the right. The unit on the y axis is arbitrary. On the top right is the video reconstructed by the decoder using the audio spectrum of the encoder. 

https://user-images.githubusercontent.com/1943719/158484295-a07674c1-1d16-4552-96d9-8b39559aa0e4.mp4

## Idea

- An *autoencoder neural network* is trained on a video. It learns to compress each image into an audio spectrum. 
- After training, the finished autoencoder can be used to transform a video (live of from a file) into an audio sequence in real time. 
- The audio should contain as much information of the input image as possible.
- Different input images should result in different audio. Different audio sequences should sound sufficiently different from one another for humans. 

## Reason

- For the **visually impaired** it might be useful to use a camera and transform what the camera sees into an audio sequence. This could help for indoor navigation. 
- One can transform images, which are in **color spaces invisible to humans**, to audio using the proposed method. For example, one could transform infrared or ultraviolet images to audio. 

## Specifics

- An *autoencoder* is trained: Its input are images. It consists of two parts: the *encoder* and the *decoder*. The encoder compresses the input image into a small code vector. The code vector is then fed into the decoder which tries to reconstruct the input using the code. Ideally, the whole autoencoder learns to efficiently transform the input image to a good code vector.
- The code, which is the compression of the image which the autoencoder learns, can be interpreted as a frequency spectrum and thus played back as audio. This frequency spectrum is specifies how loud each frequency is (magnitude).
- The frequency spectrum has to be transformed from the frequency domain to the time domain so that it can be played back. For this, in addition to the magnitude of the frequency spectrum, phase information is needed for each frequency in the spectrogram. After some experiments the best solution I found was to use a different random phase for each frequency. 

<p align="center">
<img src="https://user-images.githubusercontent.com/1943719/158268456-720dd062-17fe-4321-a722-3c7b2220c87f.svg" width="70%">
</p>
    
## The autoencoder code

### Recursively structuring encoder

If a regular autoencoder is trained, the code of the encoder is, for example, a vector consisting of 100 real numbers. The problem is that the elements of the vector are completely independent of each other. The element at position 3 is not similar to the element at position 4 at all. This is problematic because if this vector would be interpreted as a frequency spectrum, it would be hard for humans to perceive. For example, let's assume the 3rd element of the vector maps to the frequency of 100 Hz and and 4th element maps to the frequency of 101 Hz. If the 3rd element would have a significantly larger magnitude than the 4th element, it would mean that 100 Hz is significantly louder than 101 Hz. However, for a human this would not be distinguishable since these frequencies are too close, since they are just 1 Hz apart. For humans, large differences in the input image have to result in large differences in the audio spectrum which is generated. 

To achieve this, I propose the **recursively structuring encoder**. The concept is the following: During training, parts of the code, which is output by the encoder, are randomly averaged. This is done by randomly choosing a level of averaging. As can be seen in the following example images, the code is averaged up to a certain level. In the most extreme case, the whole code is averaged. Then the only information the decoder has is one real number. One level less extreme is when the lower half of the code is averaged and the upper half is averaged. The code then consists of two real numbers. Even less extreme is when the code is averaged in four groups, eight groups, sixtreen groups etc. or not averaged at all. 

#### Levels of averaging during training, example 1
![levels 1 drawio-3](https://user-images.githubusercontent.com/1943719/158484909-0338dbc5-c6bb-4cac-8385-82343a5def40.png)
#### Levels of averaging during training, example 2
![levels 2 drawio-3](https://user-images.githubusercontent.com/1943719/158484939-346f98e4-8770-4436-b66b-0c8fec4b3857.png)

This averaging of the code leads to the decoder getting less information than intended by the encoder. For example, if the 1st element of the vector has value 10 and the 2nd one value 60, it could happen that these two values are averaged and the decoder gets the 1st element as the average 35 and the 2nd element also as the average 35. This would mean that if they are randomly averaged, the 1st and 2nd component cannot represent individual information, only their average matters. This results in the encoder realizing than it doesn't make sense to use adjacent elements of the vector to encode very important information since it could be averaged away. Thus, it will learn to encode the most important information in a way that it is still preserved even when it is averaged. For example, it could learn to encode important information as the difference between the lower and the upper half of the code since the likelihood of them being averaged is small. 

### Equal distribution of the code

Furthermore, the **code should be equally distributed** in a certain space: For example, the autoencoder could learn to only use values between 0.0001 and 0.0002 and encode all information there. This would work because computers can differentiate between tiny differences of numbers but humans cannot. Thus, during training I add a penalty which makes sure that all values in the code are uniformly distributed between 0 and 1. 

### Bright should be loud

Since the bright white light has the highest power, it should also be the loudest. Thus, I also encourage the training to maximize the correlation between image brightness and the average magnitude of the audio spectrum. 

## Training

For training, there needs to be a directory with training images: 

    python learn.py --mode train --data_dir <path_to_training_image_folder>
    
To generate the training images, you can use ffmpeg. It is optional to crop the images. However, I personally prefer 4:3 over 16:9 as the input.

    ffmpeg -i <path_to_video_file> -filter:v "crop=1440:1080:240:0" <path_to_training_image_folder>/image%06d.jpg -hide_banner

## Transform video file to audio

    python learn.py --mode live --video_source <path_to_video_file> --weights <path_to_output_of_training>/variables/variables
    
You can, for example, use the video demo.mp4 that is included in the repository. 

## Transform video stream to audio

    python learn.py --mode live --video_source 0 --weights <path_to_output_of_training>/variables/variables
