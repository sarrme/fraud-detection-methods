# fraud-detection-methods

This project explores various **fraud detection techniques**, both supervised and unsupervised, commonly used in identifying anomalies or suspicious transactions in financial datasets.

---

## 🔍 Supervised Methods

- **Random Forest Classifier**  
  A powerful ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

---

## 🧠 Unsupervised Methods

### 📊 Statistical Methods
- **Z-Score**  
  Detects anomalies by measuring how many standard deviations a point is from the mean.

### 📏 Proximity / Distance-Based Methods
- **K-Nearest Neighbors (KNN)**  
  Flags outliers based on the distance from their neighbors.

### 🌐 Density-Based Methods
- **Local Outlier Factor (LOF)**  
  Identifies points that have a substantially lower density than their neighbors.

### 🌀 Clustering-Based Methods
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
  Finds clusters of varying shape and flags low-density points as outliers.

---

## 🔧 Other Techniques

- **Isolation Forest**  
  Isolates anomalies instead of profiling normal data points, which makes it efficient for high-dimensional datasets.

- **Autoencoder**  
  A type of neural network used for learning efficient data codings in an unsupervised manner. It detects fraud by reconstructing input and measuring reconstruction errors.

---

Stay tuned for implementation examples, evaluation metrics, and visualization of results in future updates.
