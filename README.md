Heart-Disease-Prediction-by-Decision-Trees
This project focuses on building a predictive model to determine whether a patient is likely to suffer from heart disease. The dataset contains various medical attributes (like cholesterol level, age, blood pressure, etc.) that serve as features to train a Decision Tree Classifier. This notebook demonstrates the full ML pipeline, including data exploration, preprocessing, model training, visualization, and evaluation.

Dataset Overview
Dataset Name: heart_v2.csv
Target Column: heart disease (0 = No disease, 1 = Disease)
Features Include:
age, sex, cp (chest pain type), trestbps (resting blood pressure),
chol (serum cholesterol), fbs (fasting blood sugar), restecg (resting ECG results),
thalach (maximum heart rate achieved), exang (exercise-induced angina),
oldpeak, slope, ca, thal
Project Pipeline Overview
1. Data Loading
df = pd.read_csv('heart_v2.csv')
We load the heart disease dataset using pandas and inspect the features using df.head() and df.columns.

2. Feature Selection and Target Separation
X = df.drop('heart disease', axis=1)  # Feature matrix
y = df['heart disease']               # Target variable
We separate the dataset into independent features (X) and the dependent target variable (y).

3.** Train-Test Split**
We split the dataset into 70% training and 30% testing using Scikit-learn:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
4.** Model Building: Decision Tree Classifier**
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
Why Decision Tree?
Decision Trees are interpretable, fast, and useful for feature importance analysis.

max_depth=3 is used to avoid overfitting and keep the model simple and understandable.

Each decision node splits the dataset based on a feature that improves classification purity (Gini Index or Entropy).

5. Model Visualization
We use graphviz, pydotplus, and export_graphviz to visualize the tree:

from sklearn.tree import export_graphviz
from IPython.display import Image
from six import StringIO
import pydotplus

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                feature_names=X.columns, class_names=['No Disease', 'Disease'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
This tree diagram gives a clear, step-by-step logic of how decisions are made to classify patients.

6. Model Evaluation
After training, we evaluate the model’s performance:

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = dt.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
Evaluation Metrics:
Accuracy: Proportion of correctly predicted instances

Confusion Matrix: Shows TP, TN, FP, FN breakdown

Classification Report: Includes Precision, Recall, F1-score for each class

##Understanding the Model

Decision Tree Working Mechanism:
At each node, the model splits the dataset using the feature that gives highest information gain (or lowest Gini impurity).

This continues until:

The max depth is reached

Or all leaves are pure (contain only one class)

Example Decision Path:
Is cholesterol level > 200?

If yes, does the patient experience exercise-induced angina?

If no, what is the maximum heart rate achieved?

This kind of logic allows medical practitioners to understand and trust the model decisions — which is critical in healthcare.

##Libraries and Tools Used

Category	Libraries
Data Analysis	pandas, numpy
Visualization	matplotlib, seaborn, graphviz, pydotplus
Modeling	sklearn.tree.DecisionTreeClassifier
Evaluation	sklearn.metrics
##How to Run This Project

1. Clone the repository
git clone https://github.com/yourusername/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn graphviz pydotplus six
3. **Run the notebook88
jupyter notebook 2_Heart+Disease+Prediction.ipynb
Results Summary
Metric	Value
Accuracy	~85%
Max Depth	3
Model Type	Decision Tree Classifier
The model offers explainable decision rules for heart disease classification.

Despite simplicity, it performs with reasonable accuracy due to thoughtful feature engineering and depth tuning.

##Future Improvements

Switch to Random Forest or Gradient Boosting for improved performance

Perform hyperparameter tuning with GridSearchCV

Add feature scaling, if using distance-based models later

Visualize feature importances

Build a Streamlit dashboard for live prediction based on user input

🧠 How Does a Decision Tree Work?
A Decision Tree is a supervised machine learning algorithm used for both classification and regression problems. It works like a flowchart where each internal node represents a decision based on a feature, each branch represents an outcome of that decision, and each leaf node represents a class label (prediction).

In this project, the Decision Tree helps predict whether a patient is likely to have heart disease based on various clinical features.

🌱 Step-by-Step Process:
Start at the Root Node:

The entire dataset is considered.
Choose the Best Feature to Split:

The feature that provides the maximum information gain or lowest Gini impurity is selected to split the data.
Branching:

The dataset is divided into subsets based on feature values.
Each subset forms a new child node.
Recursive Splitting:

The above steps repeat on each child node until:
All records in a node belong to the same class, or
The maximum depth is reached (as set via max_depth=3 in our model)
Prediction:

When a new data point is passed to the model, it follows the decision path from root to a leaf to output a prediction (e.g., heart disease = 1 or 0).
📊 Key Terms:
Term	Description
Root Node	Represents the full dataset and starts the decision-making process
Decision Node	A node that splits the data based on a feature
Leaf Node	A terminal node that makes the final prediction
Splitting	Dividing a node into two or more sub-nodes
Pruning	Removing branches that add little predictive power (to reduce overfitting)
Depth	The length of the longest path from the root to a leaf
Information Gain	The reduction in entropy after a dataset split
Gini Impurity	A measure of how often a randomly chosen element would be incorrectly classified
📉 Mathematical Criteria for Splitting
🔸 Entropy:
Entropy measures the level of impurity or disorder in the dataset:

[ Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i) ]

Where ( p_i ) is the proportion of class ( i ) in the node.

🔸 Information Gain:
Information Gain (IG) tells how much entropy is reduced after splitting:

[ IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v) ]

🔸 Gini Impurity (used in scikit-learn by default):
Gini Impurity is another metric used to measure the "purity" of a node:

[ Gini(D) = 1 - \sum_{i=1}^{C} p_i^2 ]

Where ( p_i ) is the probability of class ( i ).

✅ Why Use Decision Trees for Heart Disease Prediction?
Interpretability: You can explain to a doctor why a prediction was made.
Fast Training: Even on small or medium datasets, decision trees work very quickly.
Handles Both Numerical & Categorical Data
No Need for Feature Scaling: Unlike SVM or k-NN, no normalization required.
Built-in Feature Selection: Automatically chooses the most relevant features.
By setting max_depth=3, this model avoids overfitting and remains interpretable — a key requirement in healthcare applications where transparency and accountability are crucial.

