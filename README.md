# Feedforward-neural-network-for-topic-classification
## Project Description
This project develops a Feedforward Neural Network for topic classification. The primary goals include implementing text processing methods, constructing the network, training with Stochastic Gradient Descent (SGD), and hyperparameter tuning. The dataset used is a subset of the AG News Corpus.

## Implementation Details

### 1. Text Processing
- **Tokenization**: Transform raw text data into tokens.
- **Stop Words Removal**: Filter out common stop words.
- **Vocabulary Creation**: Construct a vocabulary of the top-N most frequent unigrams.

### 2. Network Architecture
- **Input Layer**: One-hot encoding to map words into an embedding weight matrix.
- **Hidden Layer**: 
  - Compute the mean embedding vector of all input words.
  - Apply ReLU activation function.
- **Output Layer**: Use a softmax activation function for classification.

### 3. Training Algorithm
- **SGD with Backpropagation**: Learn the weights of the neural network.
  - **Loss Function**: Minimize Categorical Cross-entropy loss.
  - **Forward Pass**: Compute intermediate outputs.
  - **Backward Pass**: Compute gradients and update weights.
  - **Regularization**: Apply Dropout after each hidden layer.

### 4. Hyperparameter Tuning
- Experiment with different learning rates, embedding sizes (e.g., 50, 300, 500), and dropout rates (e.g., 0.2, 0.5).
- Utilize tables and graphs to show training and validation performance for each combination.

### 5. Model Evaluation
- **Learning Process**: Plot training and validation loss for each epoch.
- **Performance Analysis**: Discuss whether the model overfits, underfits, or performs optimally.

### 6. Pre-trained Embeddings
- **GloVe Embeddings**: Re-train the network using pre-trained GloVe embeddings.
  - Initialize the embedding weights matrix with pre-trained weights and freeze them during training.
  - Perform hyperparameter tuning and compare performance with randomly initialized embeddings.

### 7. Network Extension
- **Additional Hidden Layers**: Extend the network by adding one or more hidden layers.
  - Analyze the effect on performance.
  - Perform hyperparameter tuning with a subset of all possible combinations.

## Results
- The project includes a detailed analysis of the results, including error analysis and discussion of misclassifications.

## Code and Documentation
- The code is well-documented and commented, providing explanations for all choices made.
- Efficient solutions are implemented using Numpy arrays to ensure the notebook executes within a reasonable time frame on standard hardware.

## Data
- The dataset used is a subset of the AG News Corpus:
  - `train.csv`: 2,400 news articles (800 per class) for training.
  - `dev.csv`: 150 news articles (50 per class) for hyperparameter tuning.
  - `test.csv`: 900 news articles (300 per class) for testing.

## Hyperparameter Tuning and Results
- A variety of hyperparameter combinations were tested, and the results were plotted to show training and validation performance for each combination.

## Conclusion
- The final model, including the use of pre-trained embeddings and additional hidden layers, showed significant improvements in performance. Detailed results and analysis are provided within the code and accompanying documentation.
