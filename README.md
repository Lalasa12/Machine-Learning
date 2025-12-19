What is Machine Learning?

Machine Learning is the subset(part) of artificial intelligence which allows us systems learn from data,identifying the patterns,make the decisions without being explicit programmatically.
Why we need ML?
We need ML because in real life problems are very complex,changing frequently so we did not write rules for everything it is very difficult. So ml enables the system to learn from the data and identifying its patterns automatically.

Examples:

1)Netflix recommendation movies.

2)House Price Prediction.

3)Email Spam Detection.

WorkFlow Of ML
1.Collect data

2.Clean data

3.Split data

4.Train data

5.Test data

6.Predict Output

1)Collect data:

Gather the Past information.

Example Data:

Study Hours	Result
1	Fail
2	Fail
4	Pass
6	Pass
This is called as dataset.

2) Clean data

Handling the missing values.

Study Hours	Result
2	Fail
❌ blank	Pass
-3	Fail
Problems:

1.missing values

2.Wrong values

3.Duplicate values

Cleaning Means:

1.Remove blanks

2.Fix wrong data

3.Remove duplicates

3)Split data

Dividing the data into 2 Parts:

1.Train data

2.Test data

80 % = Training data

20 % = Testing data

Human Example:

Study from book → Training

Write exam → Testing

Why splitting is important?

If you test with same data → cheating
Test with new data → real accuracy
4) Train Data

Model looks at data and learns patterns.

1.Model creates its rules internally.

2.But we do not see rules but model learns.

This Trained Brain = Model.

5)Test Data

Check how good the model is.

This is called as

1.Accuracy

2.Performance

3.Evaluation

6)Predict Output

Use model for real world prediction.

Example:

new student:

study hours = 5

Model says:

Pass

Same Flow:

Application	What we Predict
House price	Price
Email spam	Spam / Not
Loan system	Approved / Rejected
Netflix	Movie recommendation
Beginner thinks:

“I will write if-else rules”

ML Engineer thinks:

“I will give data, model will learn”

Types Of Machine Learning

1.Supervised Machine Learning

2.Unsupervised Machine Learning

3.Reinforcement Machine learning

1.Supervised Machine Learning:

This Model trained from the labelled data where the input is paired with the correct output.The main goal is to map the input with the output to find the new patterns and the data.

Eg:

Predicting house prices based on the size,no of bedrooms,location.

Common algorithms:

1.Linear Regression

2.Logistic Regression

3.Support Vector Machine

4.Random Forest

Trick : Data with answers

2.Unsupervised Learning

The model trains based on the unlabelled data.This model finds patterns or groups as its own.

Eg:Customer Segmentation,where customers are grouped based on the customer behaviour without predefined categories.

What it does:

1.Groups similar data

2.Find Hidden Patterns

3.Reduce data size

Common Algorithms:

1.K-Means Clustering

2.Hierarchical Clustering

3.PCA(Principal Component Analysis)

Trick:Data without answers.

3.Reinforcement learning

Reinforcement Learning means agent interacts with the environment and learns decisions by receiving the feedback in the form of rewards and penalties.

Example:

Training a Robot through the maze rewarding it for reaching the exit and penalty for hitting the wall.

Trick : Learn from rewards and Punishment.

Common Algorithms:

1.Q Learning

2.Deep Q networks

3.Policy Gradient Networks.

4.Semi Supervised Learning

It is the combination of labelled and unlabelled data.

5.Self Supervised Learning

1.Model creates its own labells data.

2.Used in the modern deep learning(text,images).

Comparison:

Type	Labels	Example
Supervised	Yes	House price prediction
Unsupervised	No	Customer grouping
Reinforcement	Reward-based	Robot navigation




