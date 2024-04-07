# Mo-Li3DeSTr: Robust LiDAR Sensor for 3D Object Detection in Rainy Weather

Mo-Li3DeSTr is a cutting-edge research project focused on enhancing 3D object detection capabilities of LiDAR sensors under adverse weather conditions, specifically during rain. Developed by Mohamed Elatfi, Haigen Min, Akram Al-Radaei, and Anass Taoussi at the School of Information Engineering, Chang’an University, China, this project presents a transformative model that integrates transformer networks with LiDAR technology to improve detection robustness.

MO-Li3DeSTr's innovative architecture demonstrates improved interpretability of noisy data, paving the way for reliable AV navigation in diverse weather scenarios and setting a new benchmark for AV sensory technology in rain-affected environments. MO-Li3DeTr marks a considerable progression in the domain, particularly evidenced by its superior Average Precision (AP) scores across varying degrees of detection difficulty—surpassing standard benchmarks in rainy scenarios known for their propensity to distort sensor data. 

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
![PROPOSED METHODOLOGY](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Elements%20of%20Proposed%20Methodology.PNG)

## MODEL ARCHITECTURE (MO-Li3DeSTr)
![MODEL ARCHITECTURE (MO-Li3DeSTr)](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Model%201.PNG)

##  A qualitative Comparison in different Situations
![A qualitative Comparison in different Situations](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Agualitative%20Comparison%20in%20different%20situations.png)

## Mo-Li3DeSTr detection in light Rain
![Mo-Li3DeSTr detection in light Rain](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Mo-Li3DeSTr%20detection%20rain%20rate%20(Light%20Rain).jpeg)

## Mo-Li3DeSTr detection in Heavy Rain
![Mo-Li3DeSTr detection in Heavy Rain](https://github.com/MoCoder007/Mo-Li3DeSTr--Robust-LiDAR-Sensor-for-3D-Object-Detection-with-Transformers-Network-in-Rainy-Weather-/blob/main/Mo-Li3DeSTr%20detection%20rain%20rate%20(Heavy%20Rain).jpeg)
