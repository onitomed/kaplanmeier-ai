# Kaplan-Meier chart to numbers algorithm

An algorithm to generate data points from images of Kaplan Meier charts. This algorithm works in multiple passes, cleaning up the image, and tries to generate data points through simple normalization.

### Run
```bash
python trial0.py
```

In a Kaplan-Meier chart, the y-axis has a fixed range 0-100. This fixed range denotes percentage of survival for patient groups after a medical treatment. The x-axis denotes time. Curves in a Kaplan-Meier chart represent the difference of 2 patients' (or 2 groups of patients') survival over time. These 2 groups generally receive different treatmens (for example, placebo and medicine groups). Over time, the survival of patients is recorded. The percentage of a group that survives over time is plotted on the y-axis at discrete points in time. Hence, the y-axis is already normalized between 0 and 100 (or 0 and 1).

The KM algorithm works by cropping each image to keep only curves in each image. The image is converted to black and white (binary color scheme), such that curve points on the image are black and the rest of the image is white. The top-most and bottom-most dark pixels are noted across the y-axis of an image. This serves as y-reference. Similarly, right-most and left-most dark pixels are collected as x-reference.

All dark pixels in this image belong to the curve. X and Y coordinates of all pixels are normalized using X and Y references respectively. The Y reference is a percentile at each data point. This is perfect because this is the real data point encoded in the chart, because Kaplan-Meier charts encode percentile on the y-axis. The X-axis is a percentile of time. Currently, the time axis remains same across charts and is hardcoded but optical character recognition needs to be embedded to obtain X (time) axis scales because different Kaplan-Meier charts have different time axis scales.

KM algorithm can generate multiple CSV files for multiple charts in a single image.

Developed and tested on Python 3.7.4 (64-bit) on Windows 10

Dependencies: cv2, numpy, matplotlib

Various subfolders will save images as checkpoints
Run this program in a folder that contains imageset0 of archive, since OpenCV does not create folders and will result in checkpoints not being reached

### Next steps

1. Time ranges were hardcoded. Text extraction needs to be done on an Kaplan-Meier chart image to obtain scale for x-axis. This must be implemented before cropping the image.
2. Needs more polishing to become an embeddable library/microservice/API.

## Bots 4 U Consultancy

If you'd like a human coder to implement and maintain this project within your organization's codebase, we have yearly contracts. Contact us at manonthemoon13131@gmail.com.
