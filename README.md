```
CS434 Machine Learning and Data Mining – Homework 2
```
# Linear Models for Regression and Classification

Overview and Objectives.In this homework, we are going to do some exercises about alternative losses for linear
regression, practice recall and precision calculations, and implement a logistic regression model to predict whether a tu-
mor is malignant or benign. There is substantial skeleton code provided with this assignment to take care of some of the
details you already learned in the previous assignment such as cross-validation, data loading, and computing accuracies.

How to Do This Assignment.

- Each question that you need to respond to is in a blue "Task Box" with its corresponding point-value listed.
- We prefer typeset solutions (LATEX / Word) but will accept scanned written work if it is legible. If a TA can’t
    read your work, they can’t give you credit.
- Programming should be done in Python and numpy. If you don’t have Python installed, you can install it from
    here. This is also the link showing how to install numpy. You can also search through the internet for numpy
    tutorials if you haven’t used it before. Google and APIs are your friends!

You are NOT allowed to...

- Use machine learning package such assklearn.
- Use data analysis package such aspandaorseaborn.
- Discuss low-level details or share code / solutions with other students.

Advice.Start early. There are two sections to this assignment – one involving working with math (20% of grade) and
another focused more on programming (80% of the grade). Read the whole document before deciding where to start.

How to submit.Submit a zip file to Canvas. Inside, you will need to have all your working code andhw2-report.pdf.
You will also submit test set predictions to a class Kaggle. This is required to receive credit for Q8.

## 1 Written Exercises: Linear Regression and Precision/Recall [5pts]

```
I’ll take any opportunity to sneak in another probability question. It’s a small one.
```
## 1.1Least Absolute Error Regression

In lecture, we showed that the solution for least squares regression was equivalent to the maximum likelihood estimate
of the weight vector of a linear model with Gaussian noise. That is to say, our probabilistic model was

```
yi∼N(μ=wTxi,σ) −→ P(yi|xi,w) =
```
#### 1

```
σ
```
#### √

```
2 π
```
```
e−
```
```
(yi−wTxi)^2
σ^2 (1)
```
and we showed that the MLE estimate under this model also minimized the sum-of-squared-errors (SSE):

```
argmax
```
#### ∏N

```
i=
```
```
P(yi|xi,w)
︸ ︷︷ ︸
Likelihood
```
```
=argmin
```
#### ∑N

```
i=
```
```
(yi−wTxi)^2
︸ ︷︷ ︸
Sum of Squared Errors
```
#### (2)

However, we also demonstrated that least squares regression is very sensitive to outliers – large errors squared can
dominate the loss. One suggestion was to instead minimize the sum ofabsoluteerrors.

In this first question, you’ll show that changing the probabilistic model to assume Laplace error yields a least
absolute error regression objective. To be more precise, we will assume the following probabilistic model for howyiis
produced givenxi:

```
yi∼Laplace(μ=wTxi,b) −→ P(yi|xi,w) =
```
#### 1

```
2 b
```
```
e−
```
```
|yi−wTxi|
b (3)
```

```
IQ1 Linear Model with Laplace Error [2pts].Assuming the model described in Eq.3, show that the
MLE for this model also minimizes the sum of absolute errors (SAE):
```
```
SAE(w) =
```
#### ∑N

```
i=
```
```
|yi−wTxi| (4)
```
```
Note that you donotneed to solve for an expression for the actual MLE expression forwto do this prob-
lem. Simply showing that the likelihood is proportional to SAE is sufficient because they would then have
the same maximizingw.
```
### 1.2Recall and Precision

```
y P(y|x)
0 0.
0 0.
0 0.
1 0.
0 0.
0 0.
1 0.
0 0.
```
```
y P(y|x)
0 0.
1 0.
1 0.
0 0.
1 0.
1 0.
1 0.
1 1.
```
```
Beyond just calculating accuracy, we discussed recall and precision as two other
measures of a classifier’s abilities. Remember that we defined recall and precision
as in terms of true positives, false positives, true negatives, and false negatives:
```
```
Recall=
```
```
#TruePositives
#TruePositives+ #FalseNegatives
```
#### (5)

```
and
```
```
Precision=
```
```
#TruePositives
#TruePositives+ #FalsePositives
```
#### (6)

```
IQ2 Computing Recall and Precision [3pts].To get a feeling for recall and precision, consider the set
of true labels (y) and model predictionsP(y|x)shown in the tables above. We compute Recall and Precision
at a specific thresholdt– considering any point withP(y|x)> tas being predicted to be the positive class
(1) and≤tto be the negative class (0). Compute and report the recall and precision for thresholds t = 0,
0.2, 0.4, 0.6, 0.8, and 1.
```
## 2 Implementing Logistic Regression for Tumor Diagnosis [20pts]

In this section, we will implement a logistic regression model for predicting whether a tumor is malignant (cancerous)
or benign (non-cancerous). The dataset has eight attributes – clump thickness, uniformity of cell size, uniformity
of cell shape, marginal adhesion, single epithelial cell size, bland chromatin, nomral nucleoli, and mitoses – all rated
between 1 and 10. You will again be submitting your predictions on the test set via the class Kaggle. You’ll need to
download thetrain_cancer.csvandtest_cancer_pub.csvfiles from the Kaggle’s data page to run the code.

### 2.1Implementing Logistic Regression

Logistic Regression.Recall from lecture that the logistic regression algorithm is a binary classifier that learns a linear
decision boundary. Specifically, it predicts the probability of an examplex∈Rdto be class 1 as

```
P(yi= 1|xi) =σ(wTxi) =
```
#### 1

```
1 +e−wTxi
```
#### , (7)

wherew∈Rdis a weight vector that we want to learn from data. To estimate these parameters from a dataset ofn
input-output pairsD={xi,yi}ni=1, we assumedyi∼Bernoulli(θ=σ(wTxi))and wrote the negative log-likelihood:

```
−logP(D|w) =−
```
```
∑n
```
```
i=
```
```
logP(yi|xi,w) =−
```
```
∑n
```
```
i=
```
#### (

```
yilogσ(wTxi) + (1−yi) log(1−σ(wTxi))
```
#### )

#### (8)


```
IQ3 Negative Log Likelihood [2pt].When we train our logistic regression model, we will be trying
to minimize this negative log-likelihood of our data by changing our weight vector. To see how this value
changes as we change our weight, we need a function to actually calculate it! To do so, finish implementing
these functions:
```
```
1.logistic(z)
Given ann× 1 input vector z, return an× 1 vector such that i’th element of the output isσ(zi).
```
```
2.calculateNegativeLogLikelihood(X,y,w))
Given ann×dinput data matrixXwhere each row represents one datapoint, an× 1 label vectory,
andd× 1 weight vectorw, compute the negative log likelihood of a logistic regression model that
applieswon the data defined byXandy. This function should calculate Eq.8 and make use of
logistic(z).
```
```
Note that np.log and np.exp will apply the log or exponential function to each element of an input matrix.
When computing negative log-likelihoods, we recommend adding a very small constant inside any log opera-
tions to keep things from growing too massive when the probability approaches zero (e.g., 0.0000001).
```
Gradient Descent.We want t find optimal weightsw∗=argminw−logP(D|w). However, taking the gradient of
the negative log-likelihood yields the expression below which does not offer a closed-form solution.

```
∇w(−logP(D|w)) =
```
```
∑n
```
```
i=
```
```
(σ(wTxi)−yi)xi (9)
```
Instead, we opted to minimize−logP(D|w)by gradient descent. We’ve provided pseudocode in the lecture but to
review the basic procedure is written below (αis the stepsize).

1. Initializewto some initial vector (all zeros,random, etc)
2. Repeat until max iterations:

```
(a)w=w−α∗∇w(−logP(D|w))
```
For convex functions (and sufficiently small values of the stepsizeα), this will converge to the minima.

The gradient expression in Eq. 9 is the sum of vectors (xi) weighted by their errors(σ(wTxi)−yi). We can express
this as a product between a matrix (X) and a vector of these errors. Specifically, assuming the logistic functionσ(·)
is applied elementwise when given a vector, we could compute:

```
∇w(−logP(D|w)) =XT(σ(Xw)−y) (10)
```
Don’t believe me? As an initial check, we can just run through the dimensions to make sure things make sense.Xw
is an×dtimesd× 1 yielding an× 1. The logistic function is applied elementwise and y is alson× 1 soσ(Xw)−y
is alson× 1 .XTisd×n. ThereforeXT(σ(Xw)isd× 1 after the final multiplication. This matches the dimensions
ofw. As a first test, this makes sense.

Next we can consider what exactly is happening here by expanding outXT(σ(Xw)−y). XT is just a matrix
where columniis just the vectorxi.Xwis just a column vector with the valuejbeingxTjw=wTxi. We could
write the product betweenXTand(σ(Xw)−y)as:

```
XT(σ(Xw)−y) =
```
#### [

```
x 1 x 2 ··· xn
```
#### ]

```
d×n
```
#### 

#### 

#### 

#### 

```
σ(wTx 1 )−y 1
σ(wTx 2 )−y 2
..
.
σ(wTxn)−yn
```
#### 

#### 

#### 

#### 

```
n× 1
```
#### =

```
∑n
```
```
i=
```
```
(σ(wTxi)−yi)xi (11)
```
Now that we’ve proven to ourselves this is the correct expression, we can implement it to compute the gradient
efficiently in logistic regression! Note that for settings where all of the data cannot fit into memory at once, the
summation solution might be preferred (or more likely, the matrix solution added up in chunks).


```
IQ4 Gradient Descent for Logistic Regression [5pt].Finish implementing thetrainLogistic
function inlogreg.py. The function takes in an×dmatrixXof example features (each row is an example)
and an× 1 vector of labelsy. It returns the learnedd× 1 weight vector and a list containing the observed
negative log-likelihood after each epoch (usescalculateNegativeLogLikleihood). The skeleton code is
shown below.
```
```
1 def trainLogistic(X,y, max_iters =2000, step_size =0.0001):
2
3 # Initialize our weights with zeros
4 w = np.zeros( (X.shape [1],1) )
5
6 # Keep track of losses for plotting
7 losses = [calculateNegativeLogLikelihood(X,y,w)]
8
9 # Take up to max_iters steps of gradient descent
10 for i in range(max_iters):
11
12 # Compute the gradient over the dataset and store in w_grad
13
14 # Todo: Compute the gradient over the dataset and store in w_grad
15 #.
16 #. Implement equation 9.
17 #.
18
19 # This is here to make sure your gradient is the right shape
20 assert(w_grad.shape == (X.shape [1],1))
21
22 # Take the update step in gradient descent
23 w = w - step_size*w_grad
24
25 # Calculate the negative log -likelihood with w
26 losses.append(calculateNegativeLogLikelihood(X,y,w))
27
28 return w, losses
```
```
To complete this code, you’ll need to implement Eq.9 to compute the gradient of the negative log-likelihood
of the dataset with respect to the weightsw. If you’ve implemented this question correctly, running
logreg.pyshould print out the learned weight vector and training accuracy. You can expect something
around 86% for the train accuracy. Provide your weight vector and accuracy in your report.
```
```
Note that an approach that loops over the dataset as in Eq. 9 takes about 15x longer than the fully matrix
version shown in Eq. 10. Either solution is fine for this assignment if you’re patient.
```
### 2.2Playing with Logistic Regression on This Dataset

Adding a Bias. The model we trained in the previous section did not have a constant offset (called a bias) in the
model – computingwTxrather thanwTx+b. A simple way to include this in our model is to add an new column
toXthat has all ones in it. This way, the first weight in our weight vector will always be multiplied by 1 and added.

```
IQ5 Adding A Dummy Variable [1pt].Implement thedummyAugmentfunction inlogreg.pyto add a
column of 1’s to the left side of an input matrix and return the new matrix.
```
```
Once you’ve done this, running the code should produce the training accuracy for both the no-bias and this
updated model. Report the new weight vector and accuracy. Did it make a meaningful difference?
```
Observing Training Curves. After finishing the previous question, the code now also produces a plot showing the
negative log-likelihood for the bias and no-bias models over the course of training. If we change the learning rate (also
called the step size), we could see significant differences in how this plot behaves – and in our accuracies.


```
IQ6 Learning Rates / Step Sizes. [2pt]Gradient descent is sensitive to the learning rate (or step size)
hyperparameter and the number of iterations. Does it look like the gradient descent algorithm has converged
or does it look like the negative log-likelihood could continue to drop ifmax_iterswas set higher?
```
```
Different values of the step size will change the nature of the curves in the training curve plot. In the skele-
ton code, this is originally set to 0.0001. Change the step size to 1, 0.1, 0.01, and 0.00001. Provide the re-
sulting training curve plots and training accuracy. Discuss any trends you observe.
```
Cross Validation. The code will also now print out K-fold cross validation results (mean and standard deviation of
accuracy) for K = 2, 3, 4, 5, 10, 20, and 50. This part may be a bit slow, but you’ll see how the mean and standard
deviation change with larger K.

```
IQ7 Evaluating Cross Validation [2pt]Come back to this after making your Kaggle submission.
The point of cross-validation is to help us make good choices for model hyperparameters. For different val-
ues of K in K-fold cross validation, we got different estimates of the mean and standard deviation of our ac-
curacy. How well did these means and standard deviations capture your actual performance on the leader-
board? Discuss any trends you observe.
```
### 2.3Make Your Kaggle Submission

Great work getting here. In this section, you’ll submit the predictions of your best model to the class-wide Kaggle
competition. You are free to make any modification to your logistic regression algorithm to improve performance;
however, it must remain logistic regression! For example, you can change feature representation, adjust the learning
rate, and max_steps parameters.

```
IQ8 Kaggle Submission [8pt]. Submit a set of predictions to Kaggle that outperforms the baseline on
the public leaderboard. To make a valid submission, use the train set to build your logistic regression clas-
sifier and then apply it to the test instances intest_cancer_pub.csvavailable from Kaggle’s Data tab.
Format your output as a two-column CSV as below:
```
```
id,type
0,
1,
2,
3,
```
```
.
.
.
```
```
where the id is just the row index intest_cancer_pub.csv. You may submit up to 10 times a day. In your
report, tell us what modifications you made for your final submission.
```
Extra Credit and Bragging Rights [1.25pt Extra Credit].The TA has made a submission to the leaderboard. Any
submission outperforming the TA on theprivateleaderboard at the end of the homework period will receive 1.25 extra
credit points on this assignment. Further, the top 5 ranked submissions will “win HW2” and receive bragging rights.


## 3 Debriefing (required in your report)

1. Approximately how many hours did you spend on this assignment?
2. Would you rate it as easy, moderate, or difficult?
3. Did you work on it mostly alone or did you discuss the problems with others?
4. How deeply do you feel you understand the material it covers (0%–100%)?
5. Any other comments?


