# Source Code and Contributions of UCL Master's Dissertation

## Dissertation Title:
*Enhancing Deep Space Optical Navigation: Deep Learning Models for Mars and Asteroids Flyby Image Processing*

## Contributions

The work presented in this dissertation makes significant contributions to the fields of machine learning and autonomous optical navigation. The key contributions are as follows:

- **Generation of Mars and Itokawa Datasets:**  
  This study has successfully created comprehensive and robust datasets for Mars and the asteroid Itokawa. These datasets include images and corresponding ground truth data, which are crucial for training supervised models to accurately locate the center of brightness (CoB) and center of mass (CoM).

- **Development of the AstroNet Architecture:**  
  A novel neural network architecture, AstroNet, was developed. AstroNet is a hybrid architecture combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. This architecture addresses the unique challenges posed by deep space images, overcoming limitations of traditional methods.

- **Development of Models for Tracking CoB and CoM of Mars and Itokawa:**  
  The dissertation resulted in deep learning models that can accurately track both the center of brightness (CoB) and center of mass (CoM) for Mars and Itokawa. These models represent a substantial advancement over existing optical navigation methods, particularly in handling irregular shapes. They enhance spacecraft navigation autonomy and precision, supporting more reliable mission operations in deep space.

## Instructions:

- **Step 1: Dataset**
- Select the celestial object: Mars (Mars file), Itokawa (asteroids file).
- Select the model type: Static (generate_dataset.py), Dynamic (generate_datasetSQ.py). 
- Select the target: CoB(type=CoB), CoM(Type=CoM).
- Adjust the size of the dataset by selecting the number of runs (509 runs max for Mars, 373 runs max for Itokawa).
- **Step 2: training and testing**
- Use "Static_models.ipynb" to train and test static models.
- Use "Dynamic_models.ipynb" to train and test dynamic models.
