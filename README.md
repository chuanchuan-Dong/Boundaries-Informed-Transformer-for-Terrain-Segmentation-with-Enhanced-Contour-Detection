# Boundaries-Informed Transformer for Terrain Segmentation with Enhanced Contour Detection
This is Project for deep machine learning at Chalmers.

# Network Structure
![network structure Diagram](https://github.com/chuanchuan-Dong/Boundaries-Informed-Transformer-for-Terrain-Segmentation-with-Enhanced-Contour-Detection/blob/master/structure.jpg)  

# Result
![network structure Diagram](https://github.com/chuanchuan-Dong/Boundaries-Informed-Transformer-for-Terrain-Segmentation-with-Enhanced-Contour-Detection/blob/master/instances.jpg)  

| Model       | Params | BG   | Smooth | Rough | Bumpy | Forbidden | Obstacle | MIoU |
|-------------|--------|-------|--------|-------|-------|-----------|----------|------|
| Mit_B0      | 7.7M   | 92.7  | 57.6   | 72.6  | 10.5  | 47.9      | 51.5     | 55.4 |
| Mit_B0BI    | 8.4M   | 93.1  | 57.8   | 78.9  | 18.0  | 54.8      | 59.7     | 60.3 |
| Mit_B1      | 30.7M  | 93.3  | 63.2   | 81.4  | 16.1  | 52.8      | 49.7     | 59.5 |
| Mit_B1BI    | 33.4M  | 94.2  | 62.4   | 82.0  | 20.7  | 60.8      | 62.7     | 63.8 |

**Table**: Model performance comparison. (Note: Mit_B*BI stands for Mit_B*Boundary-Informed model, BG stands for background.)


### Check our poster and report!

[Poster](https://github.com/chuanchuan-Dong/Boundaries-Informed-Transformer-for-Terrain-Segmentation-with-Enhanced-Contour-Detection/blob/master/Poster.pdf)

[report](https://github.com/chuanchuan-Dong/Boundaries-Informed-Transformer-for-Terrain-Segmentation-with-Enhanced-Contour-Detection/blob/master/Boundaries-Informed%20Transformer%20for%20Terrain%20Segmentation.pdf)
