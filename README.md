# Classification Using Neural Networks and Deep Learning Project
This project is about implementing a Convolution Neural Network (CNN) for a visual classification task.

## Dataset
The dataset is a subset of MNIST, containing only 4 digit classes. These 4 classes are selected based on the ID provided in the code. The dataset is loaded using provided **init_subset()** function from **precode.py** which return:
* Training images and labels
* Testing images and labels
Each image is preprocessed and reshaped to a 28*28 grayscale format.

## Neural Network architecture:
* **Input**: 28*28 grayscale image
* **Convolution Layer**:  
  - 6 feature maps
  - Filter size: 5*5
  - Stride: 2
  - No padding 
* **Activation**: ReLU
* **Max pooling**:
  - Pool size: 2x2
  - Stride: 2
* **Flatten Layer**: Converts the 3D feature map into a 1D vector
* **Fully Connected Layer 1**: 216 -> 32 neurons with ReLU activation 
* **Fully Connected Layer 2**: 32 -> 4 neurons (suitable for 4-class classification task)
* **Output Layer**: SoftMax activation for class probability prediction

## Objective
The main objective is to evaluate the perfomance of the CNN using: 
* **Accuracy**: How many prediction are correct
* **Loss**: Cross-Entropy loss to evaluate how well the model predicts class probabilities

## Training Configuration
* **Learing rate**: 0.001
* **Batch size**: 100 for training, 1 for testing
* **Epochs**: 10
* **Loss Function**: Cross-Entropy
* **Activation Function**: Softmax

After each epoch, the network's performance is evaluated on both the training and testing datasets.

## Evaluate Method
An evaluate() method is used to assess model performance. This method:
* Performs a forward pass through the network
* Computes cross-entropy loss
* Compares predicted vs actual labels to calculate accuracy
* Returns average loss and accuracy over the datset

## Requirements
- Python 3.x
- Numpy
- Matplotlib
- Pandas
- Jupyter Notebook
 
You can install required libraries using pip:
    pip install numpy matplotlib pandas

## How to Run the Code
#### 1. Clone the respository
  Clone the respository to your local machine using the following command:
    
      git clone https://github.com/Sudha-MS-Projects/Classification_using_NN_and_DL_Project.git
#### 2. Naviagte to the project directory
      cd Classification_using_NN_and_DL_Project 
#### 3. Open Jupyter Notebook
      Jupyter Notebook 
  This will open Jupyter Notebook interface in the browser.
#### 4. Run the Notebook
  Find and Open **Classification_using_CNN.ipynb** file and run the cells.
 
### 3. Review the Result
After tarining completes, the notebook displays:
* Training and Testing Accuracy per epoch
* Training and Testing loss per epoch
* Accuracy and Loss plots to visualize learing process
