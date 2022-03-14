# Kettlebell-Swing-Counter

## Problem Description
This is a program that can be used to count the number of kettlebell swings that are being performed. The inspiration for this idea came from the fact that I would sometimes lose count of the number of swings I would be performing during a workout.

## Previous Work
In determining how to solve this problem, I looked at how similar problems, such as a pushup counter, were previously solved. I found a blog post from Towards Data Science that used the Dense Optical Flow Algorithm as well as a Convolutional Neural Network (CNN) to successfully count the number of pushups that were being performed [2]. Thus, I used a similar approach for my kettlebell counter.

## Approach
For the overall approach, I first created a dataset by taking a video of myself doing kettlebell swings, running it through a Dense Optical Flow Algorithm, and then labelling each of the frames. I then trained a CNN to learn which part of the swing I was in. Lastly, I came up with a basic heuristic to determine whether or not a kettlebell swing was completed.

## Dataset
I created the dataset by taking a video of myself performing kettlebell swings. I ran the video through a Dense Optical Flow Algorithm and condensed the frames to a size of 32 x 32. Each of the frames were labeled as either "UP", "DOWN", or "NOTHING".

## Dense Optical Flow
 A Dense Optical Flow algorithm from opencv was used to display the direction of motion of each frame [1].
 <br>
 <br>
 Green corresponded to an upwards swing.
 <br>
 Purple corresponded to a downwards swing.
 <br>
 An ambiguous color corresponded to a "nothing" swing.
 
 ![](analysis/up.jpg)
 ![](analysis/down.jpg)
 ![](analysis/nothing.jpg)
 
 ```python optical_flow.py``` can be run to create the images from the kettlebell video.
 
 ## Convolutional Neural Network
 Pytorch was used to train a CNN to classify the dense optical flow images.
 <br>
 <br>
 ```python train.py``` can be run to train the model.
 
 ![](analysis/test_losses_accuracies.png)
 
 ## Results
 With an accuracy of around 85%, I was able to use the CNN output to determine whether or not a kettlebell swing was completed. With regard to how this was done, I simply determined whether or not the upward motion of the kettlebell swing was completed by determining a minimum threshold for the upward motion. With the model that was trained, 8 was an adequate number. For a demo, refer to the video.
 
 ```python demo.py``` can also be run to view the demo.
 
 ## Demo Video
 
 [![](https://img.youtube.com/vi/X_aL2QOQwiw/0.jpg)](https://youtu.be/X_aL2QOQwiw "Video Explanation")
 
 ## Discussion
 
One problem I initially encountered was the fact that the CNN was taking too long to train. I fixed this by condensing the size of the images to 32 x 32. If I continued working on this project, I would need to label a more complete dataset because there is some ambiguity between the "NOTHING" frames and the "UP"/"DOWN" frames. This is because the "NOTHING" frames will sometimes have majority upward or downward movement despite the fact that a kettlebell swing is not being performed. Additionally, it would be cool to generalize this to other kettlebell movements such as the turkish getup, snatch, etc. (I really love kettlebells). One other approach I considered was using human keypoint detection and signal analysis [3]. I ultimately stuck with this approach because it seemed more straightforward. However, both approaches are similar in that they both have a 2-step process where they try to first condense information in the image before using some sort of technique to determine the pushup count.

 ## Resources:
 [1] https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/<br>
 [2] https://towardsdatascience.com/how-i-created-the-workout-movement-counting-app-using-deep-learning-and-optical-flow-89f9d2e087ac<br>
 [3] https://aicurious.io/posts/2021-02-15-build-a-pushup-counter/
 
