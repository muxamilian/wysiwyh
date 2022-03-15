# What you hear is what you see (WYHIWYS)

## Goal

Transform an image to audio in real time. This means that you can literally **hear what you see**. 

## Example (recommended with audio turned on)

On the top left you see the original video. On the bottom there's the audio spectrum, which is the output of the *recursively structured* encoder.  On the top right the video reconstructed by the decoder using the audio spectrum of the encoder. The audio spectrum goes from 1 Hz on the left to 5000 Hz on the right. The unit on the y axis is arbitrary. 

https://user-images.githubusercontent.com/1943719/158484295-a07674c1-1d16-4552-96d9-8b39559aa0e4.mp4

## Idea

- An *autoencoder neural network* is **trained on a video**. It learns to compress each image into an **audio spectrum**. 
- After training, the finished autoencoder can be used to transform a video (live of from a file) into an audio sequence in real time.  

## Reason

- For the **visually impaired** it might be useful to use a camera and transform what the camera sees into an audio sequence. This could help for indoor navigation. 
- One can transform images, which are in **color spaces invisible** to humans, to audio using the proposed method. For example, one could transform **infrared** or **radar** images to audio. 

## Specifics

- An *autoencoder* is trained: Its inputs are images. It consists of two parts: the *encoder* and the *decoder*. The encoder compresses the input image into a small code vector. The code vector is then fed into the decoder which tries to reconstruct the input from the code. Ideally, the whole autoencoder learns to efficiently transform the input image to a good code vector.
- The code, which is the compression of the image which the autoencoder learns, can be interpreted as a frequency spectrum and thus played back as audio. This frequency spectrum is interpreted as how loud each frequency is (magnitude).
- The frequency spectrum has to be transformed from the frequency domain to the time domain so that it can be played back. For this, in addition to the frequency spectrum, phase information is needed for each frequency in the spectrogram. I generate a different random phase for each frequency in the spectrogram for each input image. 

<p align="center">
<img src="https://user-images.githubusercontent.com/1943719/158268456-720dd062-17fe-4321-a722-3c7b2220c87f.svg" width="70%">
</p>
    
## The autoencoder code

### Recursively structuring encoder

If a regular autoencoder is trained, the code of the encoder is, for example, a vector consisting of 100 real numbers. The problem is that the elements of the vector are completely independent of each other. The element in position 3 is not similar to the element at position 4 at all. This is problematic because if this vector would be interpreted as a frequency spectrum, it would be hard for humans to perceive. For example, let's assume the 3rd element of the vector maps to the frequency of 100 Hz and and 4th element maps to the frequency of 101 Hz. If the 3rd element would have a significantly larger magnitude than the 4th element, it would bean that 100 Hz has a significantly larger magnitude than 101 Hz. However, for a human this would not be distinguishable since these frequencies are too close. For humans, **large differences** in the input **image** have to result in **large differences** in the **audio** spectrum, that is generated. 

To achieve this, I propose the **recursively structured encoder**. The concept is the following: During training, parts of the code, which is output by the encoder, are randomly averaged. This is done by randomly choosing a level of averaging. As can be seen in the following example images, the code is averaged up to a certain level. In the most extreme case, the whole code is averaged. Then the only information the decoder has is one real number. One level less extreme is when the lower half of the code is average and the upper half is averaged. The code then consists of two real numbers. Even less extreme is when the code is averaged in four groups, eight groups, sixtreen groups etc. or not averaged at all. 

#### Level example 1
![levels 1 drawio-3](https://user-images.githubusercontent.com/1943719/158484909-0338dbc5-c6bb-4cac-8385-82343a5def40.png)
#### Level example 2
![levels 2 drawio-3](https://user-images.githubusercontent.com/1943719/158484939-346f98e4-8770-4436-b66b-0c8fec4b3857.png)

This averaging of the code leads to the decoder getting less information than intended by the decoder. For example, if the 1st element of the vector has value 10 and the 2nd one value 60, it could happen that these two values are averaged and the decoder gets the 1st element as the average 35 and the 2nd element also as the average 35. This would mean that if the 1st and 2nd component cannot represent individual information, only their average matters, if they are randomly averaged. This results in the encoder realizing than it doesn't make sense to use adjacent elements of the vector to encode very important information since it could be averaged away. Thus, it will learn to encode the most important information in a way, that it is still preserved even when it is averaged. For example, it could learn to encode important information as the difference between the lower and the upper half of the code since the likelihood of them being averaged is small. 

Furthermore, the **code should be equally distributed** in a certain space: For example, the autoencoder could learn to only use values between 0.0001 and 0.0002 and encode all information there. This would work because computers can differentiate between tiny differences of numbers but humans cannot. Thus, during training I add a penalty which makes sure that all values in the code are uniformly distributed between 0 and 1. 

### Equal distribution of the code

To make sure that a video can be transformed in real time and a human does not notice a large lag, the video is processed with 10 frames per second (fps) and audio is created for each frame. Because a different random phase is chosen for each frequency spectrum created from each frame, the audio output is a smooth continuous colored noise.  

### Bright should be loud

Since the bright white light contains the most energy, it should also be the loudest. Thus, I also encourage the training to maximize the correlation between image brightness and the magnitude of the audio spectrum. 

## Training

For training, there needs to be a directory with training images: 

    python learn.py --mode train --data_dir <path_to_folder_containing_training_images>

## Transform video file to audio

    python learn.py --mode live --video_source <path_to_video_file> --weights <path_to_output_of_training>/variables/variables

## Transform video stream to audio

    python learn.py --mode live --video_source 0 --weights <path_to_output_of_training>/variables/variables
