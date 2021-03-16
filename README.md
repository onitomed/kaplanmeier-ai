#kaplanmeier-ai
An AI algorithm to generate data points from images of Kaplan Meier charts.
This algorithm works in multiple passes, cleaning up the image, and tries to generate data points.
Uses image processing, openCV, and multiple in-house algorithms.
Can generate multiple CSV files for multiple charts in a single image.

PATENT PENDING

Developed and tested on Python 3.7.4 (64-bit) on Windows

Requirements: cv2 (openCV), numpy, matplotlib

To run, run trial0.py
Various subfolders will save images as checkpoints
Kindly run this program in a folder that contains imageset0 of archive, since openCV does not create folders and will result in checkpoints not being reached
