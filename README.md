# 📚 ML Lab

## Fundamentals

### Introduction to Machine Learning
- **Definition**: What is Machine Learning?
- **Types**: 
  - Supervised Learning
  - Unsupervised Learning
  - ...
- **Common Use Cases**: Classification, Regression, Clustering, Dimensionality Reduction

---

## Supervised Learning

### Linear Models
- **Linear Regression**: Simple linear models for regression tasks
  - Cost function and Gradient Descent
  - Regularization (Ridge, Lasso)
  - Multivariate Linear Regression
- **Logistic Regression**: Used for binary classification
  - Sigmoid function
  - Decision boundaries

### Support Vector Machines (SVM)
- **Concept**: Maximizing the margin between classes
- **Kernel Trick**: Linear and Non-linear separation

### Decision Trees
- **Concept**: Tree-based models for classification and regression
  - Gini Index, Entropy
  - Pruning
- **Challenges**: Overfitting

### Ensemble Methods
- **Bagging**: Aggregating predictions from multiple models
  - Random Forest
- **Boosting**: Sequential model improvement
  - AdaBoost
  - Gradient Boosting Machines (GBM)
  - XGBoost, LightGBM, CatBoost (advanced)

### k-Nearest Neighbors (k-NN)
- **Concept**: Lazy learning algorithm that bases predictions on the k closest data points
- **Applications**: Simple classifiers, regression tasks

### Naive Bayes
- **Concept**: Probabilistic classifier based on Bayes' Theorem
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes (for text)

---

## Unsupervised Learning

### Clustering Algorithms
- **k-Means Clustering**: Partition data into k distinct clusters
- **Hierarchical Clustering**: Agglomerative and divisive clustering
- **DBSCAN**: Density-based clustering for detecting outliers

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reducing feature dimensions while maintaining variance
- **t-SNE**: Non-linear dimensionality reduction for data visualization
- **Autoencoders**: Neural network-based unsupervised learning for representation learning

---

## Advanced Supervised Learning

### Regularization Techniques
- **L1/L2 Regularization**: Prevent overfitting by adding a penalty to large coefficients
- **Elastic Net**: A combination of both L1 and L2 regularization

### Model Evaluation and Selection
- **Cross-Validation**: K-fold validation, Leave-One-Out validation
- **Hyperparameter Tuning**: Grid search, Random search
- **Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC

### Ensemble Learning (Advanced)
- **Stacking**: Combining multiple models for improved performance
- **Blending**: Similar to stacking but with different data splits
- **Voting Classifier**: Combining different classifiers via majority voting

---

## Neural Networks

### Basics of Neural Networks
- **Artificial Neurons**: Activation functions (ReLU, Sigmoid, Tanh)
- **Feedforward Networks**: Architecture and backpropagation
- **Cost Functions**: MSE, Cross-Entropy

### Training Deep Neural Networks
- **Optimization Algorithms**: Stochastic Gradient Descent, Adam, RMSProp
- **Challenges**: Overfitting, Vanishing and Exploding Gradients
- **Regularization in Deep Learning**: Dropout, Batch Normalization, Weight Decay

---

## Convolutional Neural Networks (CNN)

### Convolution and Pooling
- **Convolution Layer**: Filters, Kernels, and Feature Maps
- **Pooling Layer**: Max pooling, Average pooling

### Common Architectures
- **LeNet, AlexNet, VGG**: Early breakthroughs in CNNs
- **ResNet**: Skip connections for deep networks
- **Inception Network**: Multi-scale feature extraction

### Applications of CNNs
- Image classification
- Object detection
- Image segmentation

---

## Recurrent Neural Networks (RNN) and Sequence Models

### RNN Basics
- **Concept**: Modeling sequence data (e.g., time series, language)
- **Challenges**: Vanishing gradients, short-term memory

### LSTMs and GRUs
- **LSTM**: Long Short-Term Memory units for handling long-term dependencies
- **GRU**: Gated Recurrent Units for simpler, faster training

### Applications of RNNs
- Text generation
- Machine translation
- Time series prediction

---

## Advanced Topics in Deep Learning

### Generative Models
- **Autoencoders**: Data compression and generation
- **Variational Autoencoders (VAE)**: Probabilistic generative models
- **Generative Adversarial Networks (GANs)**: Two networks (generator and discriminator) competing to create realistic data

### Attention Mechanisms and Transformers
- **Attention Mechanism**: Focusing on relevant parts of input sequences
- **Transformer Architecture**: Foundation for state-of-the-art NLP models
  - **Self-Attention**
  - **BERT, GPT**: Pretrained models for various NLP tasks

---

## Reinforcement Learning

### Basics of Reinforcement Learning
- **Agent, Environment, Rewards**: Learning from interactions
- **Exploration vs. Exploitation**: Balancing learning and maximizing rewards

### Key Algorithms
- **Q-Learning**: Learning action-value functions
- **Deep Q-Networks (DQN)**: Q-learning with deep neural networks
- **Policy Gradient Methods**: Learning policies directly
  - REINFORCE Algorithm
  - Actor-Critic Methods

### Applications of Reinforcement Learning
- Game playing (e.g., AlphaGo, Atari games)
- Robotics
- Autonomous vehicles

---

## Natural Language Processing (NLP)

### Text Preprocessing
- Tokenization
- Stopwords removal
- Stemming and Lemmatization
- Word Embeddings (Word2Vec, GloVe)

### Traditional NLP Models
- **Bag of Words (BoW)**: Simplified representation of text
- **TF-IDF**: Term Frequency-Inverse Document Frequency for importance weighting

### Sequence Models
- **Recurrent Neural Networks (RNNs)**: For sequence data like text
- **Long Short-Term Memory (LSTM)**: Overcoming the vanishing gradient problem in RNNs
- **Transformers**: For handling long-range dependencies in text

### Pretrained Language Models
- **BERT**: Bidirectional Encoder Representations from Transformers for context understanding
- **GPT**: Generative Pretrained Transformers for text generation

### Applications of NLP
- Sentiment analysis
- Text classification
- Machine translation
- Named Entity Recognition (NER)

---

## Time Series Analysis

### Introduction to Time Series Data
- **Components**: Trend, Seasonality, Noise
- **Stationarity**: Checking and transforming for stationarity

### Time Series Forecasting Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA for seasonal data
- **Prophet**: Robust forecasting method for time series

### Applications of Time Series
- Stock price prediction
- Sales forecasting
- Anomaly detection in sequences

---

## Anomaly Detection

### Introduction to Anomaly Detection
- **Concept**: Identifying outliers in data
- **Types**: Univariate, Multivariate anomalies

### Techniques for Anomaly Detection
- **Statistical Methods**: Z-Score, Tukey’s Fences
- **Machine Learning Methods**: Isolation Forest, One-Class SVM
- **Deep Learning Methods**: Autoencoders for anomaly detection

### Applications of Anomaly Detection
- Fraud detection
- Network security
- Fault detection in industrial systems

---

## Model Interpretability and Fairness

### Interpretability Techniques
- **Feature Importance**: Understanding the impact of features on predictions
- **SHAP & LIME**: Local explanations for complex models

### Fairness in ML
- **Bias and Fairness**: Detecting and mitigating bias in ML models
- **Fairness-aware Learning**: Strategies to ensure fairness in decision-making

---

## Final Step: Large Language Models (LLMs)

### Understanding Large Language Models
- **GPT Family**: Generative Pretrained Transformers for text generation
- **BERT**: Bidirectional Encoder Representations for understanding context
- **T5**: Text-to-text framework for multiple NLP tasks

### Fine-Tuning LLMs
- **Pre-training vs. Fine-tuning**: Adapting LLMs to specific tasks
- **Prompt Engineering**: Crafting inputs for optimal performance

### Applications of LLMs
- Text summarization
- Question answering
- Conversational agents

---

