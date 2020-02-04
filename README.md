# road_detection
A U-Net based approach to detect roads from stellite images

# Steps to re-produce results
1. Download **Massachusetts Roads Dataset** from https://www.cs.toronto.edu/~vmnih/data/
    1. Convert images to .jpg
    1. Place images into data/train, data/test & data/val folders
1. Run train.py
    1. Trained model will be stored in saved_models directory
    2. The results provided in the samples directory are obtained by using the default parameters specified in train.py file
2. Run test.py to check Completeness & Correctness scores of trained model

## Sample
### Original Image
![Original Image](https://github.com/cskanani/road_detection/blob/master/samples/images/7.jpg)
### Original Mask
![Original Mask](https://github.com/cskanani/road_detection/blob/master/samples/labels/7.jpg)
### Predicted Mask
![Predicted Mask](https://github.com/cskanani/road_detection/blob/master/samples/predictions/7.jpg)
