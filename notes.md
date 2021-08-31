# What you hear is what you see (WYHIWYS)

## Goal

Transform a video to audio in real time

## Idea

- A set of images is collected. These images can, for example, be images of an apartment. 
- A machine learning model is trained to transform each image in this set to a distinct audio sequence. 
- This should work in real time: A video stream should be played back as audio continuously. 

## Specifics

- An *autoencoder* is trained: Its input is an image. It consists of two parts: the *encoder* and the *decoder*. The encoder compresses the input image to a small code vector. The code vector is then fed into the decoder which tries to reconstruct the input from the code. Ideally, the whole autoencoder learns to efficiently transform the input image to a good code vector so that from the code it can perfectly reconstruct the input image again. In summary, the autoencoder learns to compress images. 
- This compression makes the most sense if all input images have something in common: For example, if the autoencoder learns on a set of images which represent an apartment, it can efficiently compress the images to a code vector because all images look somewhat similar: In the apartment you'll see some white walls and some sofas but probably not a lot of water or fish. Thus the autoencoder can efficiently compress since the data images are not completely random but it is somewhat clear what to expect, if, for example, all images come from an apartment. 
- The code, which is the compression of the image which the autoencoder learns, can be interpreted as a frequency spectrum and thus played back as audio. This frequency spectrum is interpreted as how loud each frequency is (magnitude).
- The frequency spectrum has to be transformed from the frequency domain to the time domain so that it can be played back. For this, in addition to the frequency spectrum, phase information is needed for each frequency in the spectrogram. I generate a different random phase for each frequency in the spectrogram for each input image. 
- The code of the autoencoder should fulfill certain properties: 
    - Frequencies which are close to one another should contain similar information. For example, if one frequency encodes the color of an apartment wall, the frequency next to it should be related. I achieve this by smoothing the code output by the autoencoder with a triangle function. Thus, values next to each other are averaged. 
    - The code should be equally distributed in a certain space: For example the autoencoder could learn to only use values between 0.0001 and 0.0002 and encode all information there. This would work because computers can differentiate between tiny differences of numbers but humans cannot. Thus, during training I add a penalty which makes sure that the code is uniformly distributed between 0 and 1. 
    - Similar images should have similar codes. This means that an image which is just another image offset by one pixel should be almost the same as the original image. Actually the experiments showed that this property is true by default and nothing specific has to be done to achieve this. 
- After training, the autoencoder can create a code from each image and this code can then by transformed into audio. 
- To make sure that a video can be transformed in real time and a human does not notice a large lag, the video is processed with 10 frames per second (fps) and audio is created for each frame. Because a different random phase is chosen for each frequency spectrum created from each frame, the audio output is a smooth continuous colored noise.  

## Training

    python learn.py --mode train

## Transform video file to audio

    python learn.py --mode live --video_source home3.mov --weights logs/20210829-133633/weights.1799-0.00745/variables/variables

## Transform video stream to audio

    python learn.py --mode live --video_source 0 --weights logs/20210829-133633/weights.1799-0.00745/variables/variables