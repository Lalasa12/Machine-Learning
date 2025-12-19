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


Prediction

Guessing a future value using past day.

Example:

Yesterday's Temperature = 20 degree celsius

Today Temperature = 15 degree Celsius

Tomorrow Temperature = Prediction.

Regression:

Relationship between the dependent variable(target or response variable) and independent variable(features or predictors).

Example:

Problem	Output	Regression?
Predict salary	Number	✅
Predict house price	Number	✅
Predict marks	Number	✅
Predict pass/fail	Yes/No	❌
Predict spam/not spam	Category	❌
So,if the output is a number then it is regression.

Types Of Regression.

1.Linear Regression

Why We need Linear Regression?

Real life Situation:

Experience (years)	Salary (₹)
1	20,000

2	30,000

3	45,000

4	60,000

Now a new person comes

Experience = 5

How much salary should we give?

So we need a model to predict the salary.This is where Linear Regression is used.

What is Linear Regression?

The relationship between the single independent variable and dependent variable.

(or)

Used to predict the continuous numerical values by straight line relationship between the input and output.

input=experience output=salary Result=Number

why it is called Linear?

Linear = Straight Line

So,if we draw a graph

X-axis = Experience

Y-axis = salary

All points roughly through Straight Line.

Formula:

Y=mx+c

x = input(Experience)

y = output(Salary)

m = slope(How fast salary increases)

c = intercept(Starting Salary)

How Linear Regression Learns?

1.Takes Training Data

2.Draws a line

3.Calculates Error

4.Adjusts the Line

5.Repeat until the error is minimum

Daily Life Scenerio:

1.First throw → misses basket

2️. See how far you missed (error)

3️. Adjust your hand

4️. Throw again

5️. Practice until you score


from sklearn.linear_model import LinearRegression

X = [[1],[2],[3],[4],[5]]

y = [20000,50000,40000,54400,78000]

model = LinearRegression()

model.fit(X,y)

prediction = model.predict([[6]])

print(prediction)

[84600.]

Step 1:

Import the Algorithm:

sklearn = Ml Library

linear_module = Module for linear models

LinearRegression = Algorithm

Step 2:

Input Data = X

Step 3:

Output Data = y(Target/Label)

** Step 4:**

Create a Model

Step 5:

Train a Model

What fit() does:

Takes input X

Takes output y

Learns relationship between them.

Step 6:

Predict the output

Step 7:

Print the result

2.Multiple Linear Regression

The model tries to establish the relationship between the dependent variable and the multiple independent variable.

(or)

Multiple Linear Regression is a supervised regression algorithm used to predict a NUMBER using MORE THAN ONE input feature.

Why do we need Multiple Linear Regression?

In Linear Regression,

We use only one input like Experience --> Salary.

But Real life not like that.

Real-life example- House Price

House price depends on:

1.Size (sq ft)

2.Number of rooms

3.Location

4.Age of house

More than one input

So we need:

Multiple Linear Regression

Formula:

y = m1x1 + m2x2 + m3x3 + c

x1 → Experience

x2 → Education

x3 → Skills

y → Salary

Still a straight-line idea, just more inputs.


[3]
0s
from sklearn.linear_model import LinearRegression

X = [
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4]
]

y = [20000, 30000, 45000, 60000]

model = LinearRegression()

model.fit(X, y)

model.predict([[5, 5, 5]])

array([72500.])

Feature:

It is a input value used to make prediction.

Example:

Feature (Input)	Type

Experience	Feature

Education	Feature

Skills	Feature

Salary	❌ Output (NOT feature)

Features = X values output = y values

Example:

Column	Feature or Output?

Size	Feature

Rooms	Feature

Location	Feature

Price	Output

Note: Feature = Input column

1.feature → Line 📈

2.features → Plane 🟦

Many features → Multi-dimensional surface







