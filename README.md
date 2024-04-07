# Mo-Li3DeSTr: Robust LiDAR Sensor for 3D Object Detection in Rainy Weather

Mo-Li3DeSTr is a cutting-edge research project focused on enhancing 3D object detection capabilities of LiDAR sensors under adverse weather conditions, specifically during rain. Developed by Mohamed Elatfi, Haigen Min, Akram Al-Radaei, and Anass Taoussi at the School of Information Engineering, Chang’an University, China, this project presents a transformative model that integrates transformer networks with LiDAR technology to improve detection robustness.

## Introduction

Autonomous vehicles (AVs) rely heavily on LiDAR (Light Detection and Ranging) for precise 3D object detection. However, LiDAR's performance can significantly degrade in rainy conditions, affecting the safety and efficiency of AVs. Mo-Li3DeSTr addresses this challenge by introducing an innovative architecture that enhances the interpretability of noisy data, enabling reliable navigation in diverse weather scenarios.

## Key Features

- *Transformer Network Integration*: Utilizes transformer networks to process LiDAR data, focusing on relevant features while filtering out noise caused by rain.
- *Robust Detection Performance*: Demonstrates superior Average Precision (AP) scores for detecting cars and cyclists in rainy conditions, surpassing standard benchmarks.
- *Advanced Noise Filtering*: Employs noise filtering techniques for preprocessing LiDAR data, ensuring optimal input quality for the neural network.


## Technologies Used

- *Python* for core development.
- *PyTorch* as the primary deep learning framework.
- The *KITTI dataset* for training and testing.
- Advanced *transformer network models* for data processing and interpretation.

## Current Status

MO-Li3DeSTr is currently in an advanced stage of development, with ongoing efforts to further refine its capabilities and extend its application to other adverse weather conditions like fog and snow.

## PROPOSED METHODOLOGY
![PROPOSED METHODOLOGY](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Element%20of%20the%20Proposed%20Solution%20new.PNG)

## MODEL ARCHITECTURE (MO-Li3DeSTr)
![MODEL ARCHITECTURE (MO-Li3DeSTr)](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Model%201.PNG)

## TNET (TRANSFORMATION NETWORK)
![TNET (TRANSFORMATION NETWORK)](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/TNet%20Figure.PNG)

## ROBUST POINTNET EMBEDDING LAYER
![ROBUST POINTNET EMBEDDING LAYER](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Robust%20PointNet%20Figure.PNG)

## RAIN EFFECT LAYER
![RAIN EFFECT LAYER](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Rain%20Effect%20Figure.PNG)

## Mo-Li3DeSTr detection in light Rain
![Mo-Li3DeSTr detection in light Rain](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Mo-Li3DeSTr%20detection%20rain%20rate%20(Light%20Rain).jpeg)

## Mo-Li3DeSTr detection in Heavy Rain
![Mo-Li3DeSTr detection in Heavy Rain](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Mo-Li3DeSTr%20detection%20rain%20rate%20(Heavy%20Rain).jpeg)


## Acknowledgments
Special thanks to Mohamed Elatfi, Haigen Min, Akram Al-Radaei, Anass Taoussi, and all contributors. This project is supported by Chang’an University, China.

