# 21 Supervised Learning Algorithms II: SVMs, Random Forest, Naive Bayes

<br>

## Content 
- **Support Vector Machines**
    - **Support Vector Machines**
        - [Introduction SVMs](#Introduction-SVMs)
        - [Optimal Decision Boundaries](#Optimal-Decision-Boundaries)
        - [Support Vectors and Margins](#Support-Vectors-and-Margins)
        - [SVM in scikit-learn](#SVM-in-scikit-learn)
        - [Outliers](#Outliers)
        - [Kernels](#Kernels)
        - [Polynomial Kernel](#Polynomial-Kernel)
        - [Radial Basis Function Kernel](#Radial-Basis-Function-Kernel)
        - [Review: SVMs](#Review-SVMs)
- **Random Forests**
    - **Random Forests**
        - [Basics of a Random Forest](#Basics-of-a-Random-Forest)
        - [Bootstrapping](#Bootstrapping)
        - [Bagging](#Bagging)
        - [Random Feature Selection](#Random-Feature-Selection)
        - [Bagging in `scikit-learn`](#Bagging-in-scikit-learn)
        - [Train and Predict using `scikit-learn`](#Train-and-Predict-using-scikit-learn)
        - [Random Forest Regressor](#Random-Forest-Regressor)
        - [Review: Random Forests](#Review-Random-Forests)
- **Naive Bayes Classifier**
    - **Bayes' Theorem**
        - [Introduction to Bayes' Theorem](#Introduction-to-Bayes'-Theorem)
        - [Independent Events](#Independent-Events)
        - [Conditional Probability](#Conditional-Probability)
        - [Testing for a Rare Disease](#Testing-for-a-Rare-Disease)
        - [Bayes' Theorem](#Bayes'-Theorem)
        - [Spam Filters](#Spam-Filters)
    - **Naive Bayes Classifier**
        - [The Naive Bayes Classifier](#The-Naive-Bayes-Classifier)
        - [Investigate the Data](#Investigate-the-Data)
        - [Bayes Theorem I](#Bayes-Theorem-I)
        - [Bayes Theorem II](#Bayes-Theorem-II)
        - [Smoothing](#Smoothing)
        - [Classify](#Classify)
        - [Formatting the Data for `scikit-learn`](#Formatting-the-Data-for-`scikit-learn`)
        - [Using `scikit-learn`](#Using-`scikit-learn`)

<br>

## Introduction SVMs
- A **Support Vector Machine** (SVM) is a powerful supervised machine learning model used for classification. 
- An SVM makes classifications by defining a decision boundary and then seeing what side of the boundary an unclassified point falls on. 
- In the next few exercises, we’ll learn how these decision boundaries get defined, but for now, know that they’re defined by using a training set of classified points. 
- That’s why SVMs are *supervised* machine learning models.

<br>

- Decision boundaries are easiest to wrap your head around when the data has two features. 
- In this case, the decision boundary is a line. 
- Take a look at the example below.  
    <img src="Images/two_dimensions.webp" width="400">
- Note that if the labels on the figures in this lesson are too small to read, you can resize this pane to increase the size of the images.

<br>

- This SVM is using data about fictional games of Quidditch from the Harry Potter universe! 
- The classifier is trying to predict whether a team will make the playoffs or not. 
- Every point in the training set represents a “historical” Quidditch team. 
- Each point has two features — the average number of goals the team scores and the average number of minutes it takes the team to catch the Golden Snitch.

<br>

- After finding a decision boundary using the training set, you could give the SVM an unlabeled data point, and it will predict whether or not that team will make the playoffs.

<br>

- Decision boundaries exist even when your data has more than two features. 
- If there are three features, the decision boundary is now a plane rather than a line.  
    <img src="Images/three_dimensions.webp" width="400">
- As the number of dimensions grows past 3, it becomes very difficult to visualize these points in space.
- Nonetheless, SVMs can still find a decision boundary.
- However, rather than being a separating line, or a separating plane, the decision boundary is called a *separating hyperplane*.

<br>

## Optimal Decision Boundaries
- One problem that SVMs need to solve is figuring out what decision boundary to use. 
- After all, there could be an infinite number of decision boundaries that correctly separate the two classes. 
- Take a look at the image below:  
    <img src="Images/decision_boundaries.webp" width="400">
- There are so many valid decision boundaries, but which one is best? 
- In general, we want our decision boundary to be as far away from training points as possible.

<br>

- Maximizing the distance between the decision boundary and points in each class will decrease the chance of false classification. 
- Take graph C for example.  
    <img src="Images/graph_c.webp" width="400">
- The decision boundary is close to the blue class, so it is possible that a new point close to the blue cluster would fall on the red side of the line.
- Out of all the graphs shown here, graph F has the best decision boundary.

<br>

## Support Vectors and Margins
- We now know that we want our decision boundary to be as far away from our training points as possible. 
- Let’s introduce some new terms that can help explain this idea.

<br>

- The support vectors are the points in the training set closest to the decision boundary. 
- In fact, these vectors are what define the decision boundary. 
- But why are they called vectors? 
- Instead of thinking about the training data as points, we can think of them as vectors coming from the origin.  
    <img src="Images/vectors.webp" width="600">
- These vectors are crucial in defining the decision boundary — that’s where the “support” comes from. 
- If you are using `n` features, there are at least `n+1` support vectors.

<br>

- The distance between a support vector and the decision boundary is called the *margin*. 
- We want to make the margin as large as possible. 
- The support vectors are highlighted in the image below:  
    <img src="Images/margin.png" width="600">
- Because the support vectors are so critical in defining the decision boundary, many of the other training points can be ignored. 
- This is one of the advantages of SVMs. 
- Many supervised machine learning algorithms use every training point in order to make a prediction, even though many of those training points aren’t relevant. 
- SVMs are fast because they only use the support vectors!

<br>

## SVM in scikit-learn
- Now that we know the concepts behind SVMs we need to write the code that will find the decision boundary that maximizes the margin. 
- All of the code that we’ve written so far has been guessing and checking — we don’t actually know if we’ve found the best line. 
- Unfortunately, calculating the parameters of the best decision boundary is a fairly complex optimization problem. 
- Luckily, Python’s scikit-learn library has implemented an SVM that will do this for us.

<br>

- Note that while it is not important to understand how the optimal parameters are found, you should have a strong conceptual understanding of what the model is optimizing.

<br>

- To use scikit-learn’s SVM we first need to create an SVC object.
- It is called an SVC because scikit-learn is calling the model a Support Vector Classifier rather than a Support Vector Machine.
    ```python	
    classifier = SVC(kernel = 'linear')
    ```
- We’ll soon go into what the `kernel` parameter is doing, but for now, let’s use a `'linear'` kernel.

<br>

- Next, the model needs to be trained on a list of data points and a list of labels associated with those data points. 
- The labels are analogous to the color of the point — you can think of a `1` as a red point and a 0 as a blue point. 
- The training is done using the `.fit()` method:
    ```python
    training_points = [[1, 2], [1, 5], [2, 2], [7, 5], [9, 4], [8, 2]]
    labels = [1, 1, 1, 0, 0, 0]
    classifier.fit(training_points, labels) 
    ```
- The graph of this dataset would look like this:  
    <img src="Images/example_dataset.png" width="600">
- Calling `.fit()` creates the line between the points.

<br>

- Finally, the classifier predicts the label of new points using the `.predict()` method. 
- The `.predict()` method takes a list of points you want to classify. 
- Even if you only want to classify one point, make sure it is in a list:
    ```python
    print(classifier.predict([[3, 2]]))
    ```
- In the image below, you can see the unclassified point `[3, 2]` as a black dot. 
- It falls on the red side of the line, so the SVM would predict it is red.  
    <img src="Images/predict.webp" width="600">
- In addition to using the SVM to make predictions, you can inspect some of its attributes. 
- For example, if you can print `classifier.support_vectors_` to see which points from the training set are the support vectors.
- In this case, the support vectors look like this:
    ```python
    print(classifier.support_vectors_)
    
    # Output
    [[7, 5],
     [8, 2],
     [2, 2]]
    ```

<br>

## Outliers
- SVMs try to maximize the size of the margin while still correctly separating the points of each class. 
- As a result, outliers can be a problem. 
- Consider the image below.  
    <img src="Images/outliers.png" width="600">
- The size of the margin decreases when a single outlier is present, and as a result, the decision boundary changes as well. 
- However, if we allowed the decision boundary to have some error, we could still use the original line.

<br>

- SVMs have a parameter `C` that determines how much error the SVM will allow for. 
- If `C` is large, then the SVM has a hard margin — it won’t allow for many misclassifications, and as a result, the margin could be fairly small. 
- If `C` is too large, the model runs the risk of overfitting. 
- It relies too heavily on the training data, including the outliers.

<br>

- On the other hand, if `C` is small, the SVM has a soft margin. 
- Some points might fall on the wrong side of the line, but the margin will be large. 
- This is resistant to outliers, but if `C` gets too small, you run the risk of underfitting. 
- The SVM will allow for so much error that the training data won’t be represented.

<br>

- When using scikit-learn’s SVM, you can set the value of `C` when you create the object:
    ```python
    classifier = SVC(C = 0.01)
    ```
- The optimal value of `C` will depend on your data. 
- Don’t always maximize margin size at the expense of error.
- Don’t always minimize error at the expense of margin size. 
- The best strategy is to validate your model by testing many different values for `C`.

<br>

## Kernels
- Up to this point, we have been using data sets that are linearly separable. 
- This means that it’s possible to draw a straight decision boundary between the two classes. 
- However, what would happen if an SVM came along a dataset that wasn’t linearly separable?  
    <img src="Images/circles.webp" width="500">
- It’s impossible to draw a straight line to separate the red points from the blue points!

<br>

- Luckily, SVMs have a way of handling these data sets. 
- Remember when we set `kernel = 'linear'` when creating our SVM? 
- Kernels are the key to creating a decision boundary between data points that are not linearly separable.

<br>

- Note, that most machine learning models should allow for some error. 
- For example, the image below shows data that isn’t linearly separable. 
- However, it is not linearly separable due to a few outliers.
- We can still draw a straight line that, for the most part, separates the two classes. 
- You shouldn’t need to create a non-linear decision boundary just to fit some outliers. 
- Drawing a line that correctly separates every point would be drastically overfitting the model to the data.
    <img src="Images/outlier_example.webp" width="500">

<br>

## Polynomial Kernel
- That kernel seems pretty magical. 
- It is able to correctly classify every point! 
- Let’s take a deeper look at what it was really doing.
- We start with a group of non-linearly separable points that looked like this:
    <img src="Images/circles.webp" width="500">
- The kernel transforms the data in a clever way to make it linearly separable.
- We used a polynomial kernel which transforms every point in the following way:
$$ (x, y) \rightarrow (\sqrt{2} \cdot x \cdot y, x^2, y^2) $$
- The kernel has added a new dimension to each point! For example, the kernel transforms the point `[1, 2]` like this:
$$ (1, 2) \rightarrow (\sqrt{2} \cdot 1 \cdot 2, 1^2, 2^2) = (2\sqrt{2}, 1, 4) $$
- If we plot these new three dimensional points, we get the following graph:
    <img src="Images/projected_with_boundary.png" width="500">
- All of the blue points have scooted away from the red ones. 
- By projecting the data into a higher dimension, the two classes are now linearly separable by a plane. 
- We could visualize what this plane would look like in two dimensions to get the following decision boundary.
    <img src="Images/curved_boundary.png" width="500">

<br> 

## Radial Basis Function Kernel
- The most commonly used kernel in SVMs is a radial basis function (**rbf**) kernel.
-  This is the default kernel used in scikit-learn’s `SVC` object. 
- If you don’t specifically set the kernel to `"linear"`, `"poly"` the `SVC` object will use an rbf kernel. 
- If you want to be explicit, you can set `kernel = "rbf"`, although that is redundant.

<br>

- It is very tricky to visualize how an rbf kernel “transforms” the data. 
- The polynomial kernel we used transformed two-dimensional points into three-dimensional points. 
- An rbf kernel transforms two-dimensional points into points with an infinite number of dimensions!

<br>

- We won’t get into how the kernel does this — it involves some fairly complicated linear algebra. 
- However, it is important to know about the rbf kernel’s `gamma` parameter.
    ```python
    classifier = SVC(kernel = "rbf", gamma = 0.5, C = 2)
    ```
- `gamma` is similar to the `C` parameter. 
- You can essentially tune the model to be more or less sensitive to the training data. 
- A higher `gamma`, say `100`, will put more importance on the training data and could result in overfitting. 
- Conversely, A lower `gamma` like `0.01` makes the points in the training data less relevant and can result in underfitting.

<br>

## Review: SVMs
- SVMs are supervised machine learning models used for classification.
- An SVM uses support vectors to define a decision boundary. Classifications are made by comparing unlabeled points to that decision boundary.
- Support vectors are the points of each class closest to the decision boundary. The distance between the support vectors and the decision boundary is called the margin.
- SVMs attempt to create the largest margin possible while staying within an acceptable amount of error.
- The `C` parameter controls how much error is allowed. A large `C` allows for little error and creates a hard margin. A small `C` allows for more error and creates a soft margin.
- SVMs use kernels to classify points that aren’t linearly separable.
- Kernels transform points into higher dimensional space. A polynomial kernel transforms points into three dimensions while an rbf kernel transforms points into infinite dimensions.
- An rbf kernel has a `gamma` parameter. If `gamma` is large, the training data is more relevant, and as a result overfitting can occur.

<br>

--- 

<br>

## Basics of a Random Forest
- We’ve seen that [decision trees](019_Supervised_Learning_I_Regressors_Classifiers_and_Trees.md)** can be powerful supervised machine learning models. 
- However, they’re not without their weaknesses — decision trees are often prone to overfitting. 
- We’ve discussed some strategies to minimize this problem, like pruning, but sometimes that isn’t enough. 
- We need to find another way to generalize our trees. 
- This is where the concept of a random forest comes in handy.

<br>

- A random forest is an *ensemble machine learning technique*. 
- A random forest contains many decision trees that all work together to classify new points. 
- When a random forest is asked to classify a new point, the random forest gives that point to each of the decision trees. 
- Each of those trees reports their classification and the random forest returns the most popular classification. 
- It’s like every tree gets a vote, and the most popular classification wins.
- Some of the trees in the random forest may be overfit, but by making the prediction based on a large number of trees, overfitting will have less of an impact.

<br>

- ** *A prerequisite for this lesson is the understanding of decision trees*. Be sure to revisit the decision trees module if needed!

<img src="Images/random_forest.svg" width="600" style="display: block; margin: auto;">

<br>

## Bootstrapping
- You might be wondering how the trees in the random forest get created. 
- After all, right now, our algorithm for creating a decision tree is deterministic — given a training set, the same tree will be made every time. 
- To make a random forest, we use a technique called *bagging*, which is short for *bootstrap aggregating*. 
- This exercise will explain bootstrapping, which is a type of sampling method done with replacement.

<br>

- How it works is as follows: 
    - every time a decision tree is made, it is created using a different subset of the points in the training set. 
    - For example, if our training set had `1000` rows in it, we could make a decision tree by picking `100` of those rows at random to build the tree.
    -  This way, every tree is different, but all trees will still be created from a portion of the training data.

<br>

- In bootstrapping, we’re doing this process *with replacement*.
- Picture putting all `100` rows in a bag and reaching in and grabbing one row at random. 
- After writing down what row we picked, we put that row back in our bag.
-  This means that when we’re picking our `100` random rows, we could pick the same row more than once. 
- In fact, it’s very unlikely, but all `100` randomly picked rows could all be the same row! 
- Because we’re picking these rows with replacement, there’s no need to shrink our bagged training set from 1000 rows to `100`. 
- We can pick 1000 rows at random, and because we can get the same row more than once, we’ll still end up with a unique data set.

<br>

- We’ve loaded a dataset about cars here. 
- An important field within the dataset is the safety rating, which tells us how crash/rollover resistant a car is, in other words, how safe the car is.
- The `safety` variable can be either “low”, “med”, or “high.” 
- We’re going to implement bootstrapping and estimate the average safety rating across the different bootstrapped samples.

<br>

- Here is a list of the variables in the car evaluation dataset.
    | Variable | Description |
    | --- | --- |
    | safety | estimated safety of the car (low, med, or high) |
    | buying | buying price |
    | maint | price of the maintenance |
    | doors | number of doors |
    | persons | capacity in terms of persons to carry |
    | lug_boot | the size of luggage boot |
    | accep | evaulation level (unacceptable, acceptable, good, very good) |

<br>

## Bagging
- Random forests create different trees using a process known as bagging, which is short for bootstrapped aggregating. 
- As we already covered bootstrapping, the process starts with creating a single decision tree on a bootstrapped sample of data points in the training set. 
- Then after many trees have been made, the results are “aggregated” together. 
- In the case of a classification task, often the aggregation is taking the majority vote of the individual classifiers. 
- For regression tasks, often the aggregation is the average of the individual regressors.

<br>

- We will dive into this process for the cars dataset we used in the previous exercise.
- The dataset has six features:
    - `buying`: car price as a categorical variable: “vhigh”, “high”, “med”, or “low”
    - `maint`: cost of maintaining the car; can be “vhigh”, “high”, “med”, or “low”.
    - `doors`: number of doors; can be “2”, “3”, “4”, “5more”.
    - `persons`: number of people the car can hold; can be “2”, “4”, or “more”.
    - `lugboot`: size of the trunk; can be “small”, “med”, or “big”.
    - `safety`: safety rating of the car; can be “low”, “med”, or “high”

<br>

## Random Feature Selection
- In addition to using bootstrapped samples of our dataset, we can continue to add variety to the ways our trees are created by randomly selecting the features that are used.
- Recall that for our car data set, the original features were the following:
    - The price of the car which can be “vhigh”, “high”, “med”, or “low”.
    - The cost of maintaining the car which can be “vhigh”, “high”, “med”, or “low”.
    - The number of doors which can be “2”, “3”, “4”, “5more”.
    - The number of people the car can hold which can be “2”, “4”, or “more”.
    - The size of the trunk which can be “small”, “med”, or “big”.
    - The safety rating of the car which can be “low”, “med”, or “high”
- Our target variable for prediction is an acceptability rating, `accep`, that’s either `True` or `False`. 
- For our final features sets, `x_train` and `x_test`, the categorical features have been dummy encoded, giving us 15 features in total.

<br>

- When we use a decision tree, all the features are used and the split is chosen as the one that increases the information gain the most. 
- While it may seem counter-intuitive, selecting a random subset of features can help in the performance of an ensemble model. 
- In the following example, we will use a random selection of features prior to model building to add additional variance to the individual trees.
- While an individual tree may perform worse, sometimes the increases in variance can help model performance of the ensemble model as a whole.

<br>

## Bagging in `scikit-learn`
- The two steps we walked through above created trees on bootstrapped samples and randomly selecting features. 
- These can be combined together and implemented at the same time! 
- Combining them adds an additional variation to the base learners for the ensemble model. 
- This in turn increases the ability of the model to generalize to new and unseen data, i.e., it minimizes bias and increases variance. 
- Rather than re-doing this process manually, we will use `scikit-learn`‘s bagging implementation, `BaggingClassifier()`, to do so.

<br>

- Much like other models we have used in `scikit-learn`, we instantiate a instance of `BaggingClassifier()` and specify the parameters. 
- The first parameter, `base_estimator` refers to the machine learning model *that is being bagged*. 
- In the case of random forests, the *base estimator* would be a *decision tree*. 
- We are going to use a decision tree classifier WITH a `max_depth` of 5, this will be instantiated with `BaggingClassifier(DecisionTreeClassifier(max_depth=5))`.

<br>

- After the model has been defined, methods `.fit()`, `.predict()`, `.score()` can be used as expected. 
- Additional hyperparameters specific to bagging include the number of estimators (`n_estimators`) we want to use and the maximum number of features we’d like to keep (`max_features`).

<br>

- *Note*: While we have focused on decision tree classifiers (as this is the base learner for a random forest classifier), this procedure of bagging is not specific to decision trees, and in fact can be used for any base classifier or regression model. 
- The scikit-learn implementation is generalizable and can be used for other base models!

<br>

## Train and Predict using `scikit-learn`
- Now that we have covered two major ways to combine trees, both in terms of samples and features, we are ready to get to the implementation of random forests! 
- This will be similar to what we covered in the previous exercises, but the random forest algorithm has a slightly different way of randomly choosing features. 
- Rather than choosing a single random set at the onset, each split chooses a different random set.

<br>

- For example, when finding which feature to split the data on the first time, we might randomly choose to only consider the price of the car, the number of doors, and the safety rating. 
- After splitting the data on the best feature from that subset, we’ll likely want to split again. 
- For this next split, we’ll randomly select three features again to consider. 
- This time those features might be the cost of maintenance, the number of doors, and the size of the trunk.
-  We’ll continue this process until the tree is complete.

<br>

- One question to consider is how to choose the number of features to randomly select. 
- Why did we choose 3 in this example? A good rule of thumb is select as many features as *the square root of the total number of features*. 
- Our car dataset doesn’t have a lot of features, so in this example, it’s difficult to follow this rule. 
- But if we had a dataset with 25 features, we’d want to randomly select 5 features to consider at every split point.

<br>

- You now have the ability to make a random forest using your own decision trees. 
- However, `scikit-learn` has a `RandomForestClassifier()` class that will do all of this work for you! 
- `RandomForestClassifier` is in the `sklearn.ensemble module`.

<br>

- `RandomForestClassifier()` works almost identically to `DecisionTreeClassifier()` — the `.fit()`, `.predict()`, and `.score()` methods work in the exact same way.

<br>

## Random Forest Regressor
- Just like in decision trees, we can use random forests for regression as well! 
- It is important to know when to use regression or classification — this usually comes down to what type of variable your target is. 
- Previously, we were using a binary categorical variable (acceptable versus not), so a classification model was used.

<br>

- We will now consider a hypothetical new target variable, price, for this data set, which is a continuous variable. 
- We’ve generated some fake prices in the dataset so that we have numerical values instead of the previous categorical variables. 
- (Please note that these are not reflective of the previous categories of high and low prices - we just wanted some numeric values so we can perform regression! :) )

<br>

- Now, instead of a classification task, we will use `scikit-learn`‘s `RandomForestRegressor()` to carry out a regression task.

<br>

- Note: Recall that the default evaluation score for regressors in `scikit-learn` is the R-squared score.

<br>

## Review: Random Forests
- Here are some of the major takeaways about random forests:
    - A random forest is an ensemble machine learning model. It makes a classification by aggregating the classifications of many decision trees.
    - Random forests are used to avoid overfitting. By aggregating the classification of multiple trees, having overfitted trees in a random forest is less impactful.
    - Every decision tree in a random forest is created by using a different subset of data points from the training set. Those data points are chosen at random with replacement, which means a single data point can be chosen more than once. This process is known as bagging.
    - When creating a tree in a random forest, a randomly selected subset of features are considered as candidates for the best splitting feature. If your dataset has n features, it is common practice to randomly select the square root of n features.

<img src="Images/random_forest.svg" width="700" style="display: block; margin: auto;">

<br>

--- 

<br>

## Introduction to Bayes' Theorem
- In this lesson, we’ll learn about Bayes’ Theorem. 
- Bayes’ Theorem is the basis of a branch of statistics called Bayesian Statistics, where we take prior knowledge into account before calculating new probabilities.
- This allows us to find narrow solutions from a huge universe of possibilities. 
- British mathematician Alan Turing used it to crack the German Enigma code during WWII. And now it is used in:
    - Machine Learning
    - Statistical Modeling
    - A/B Testing
    - Robotics
- By the end of this lesson, you’ll be able to solve simple problems involving prior knowledge.
$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $$
- In this formula:
    - `P(A|B)` is the probability of `A` given `B`.
    - `P(B|A)` is the probability of `B` given `A`.
    - `P(A)` and `P(B)` are the probabilities of `A` and `B` independently.

<br>

## Independent Events
- The ability to determine whether two events are *independent* is an important skill for statistics.
- If two events are **independent**, then the occurrence of one event does not affect the probability of the other event. 
- Here are some examples of independent events:
    - I wear a blue shirt; my coworker wears a blue shirt
    - I take the subway to work; I eat sushi for lunch
    - The NY Giants win their football game; the NY Rangers win their hockey game
- If two events are **dependent**, then when one event occurs, the probability of the other event occurring changes in a predictable way.
- Here are some examples of dependent events:
    - It rains on Tuesday; I carry an umbrella on Tuesday
    - I eat spaghetti; I have a red stain on my shirt
    - I wear sunglasses; I go to the beach

<br>

## Conditional Probability
- *Conditional probability* is the probability that two events happen. 
- It’s easiest to calculate conditional probability when the two events are independent.
- **Note**: For the rest of this lesson, we’ll be using the statistical convention that the probability of an event is written as `P(event)`.
- If the probability of event `A` is `P(A)` and the probability of event `B` is `P(B)` and the two events are independent, then the probability of both events occurring is the product of the probabilities:
$$ P(A \cap B) = P(A) \times P(B) $$
- The symbol $\cap$ just means “and”, so `P(A ∩ B)` means the probability that both `A` and `B` happen.
- For instance, suppose we are rolling a pair of dice, and want to know the probability of rolling two sixes.  
    <img src="Images/dice.png" width="100">
    <img src="Images/dice.png" width="100">
- Each die has six sides, so the probability of rolling a six is 1/6. 
- Each die is independent (i.e., rolling one six does not increase or decrease our chance of rolling a second six), so:
$$ P(6 \cap 6) = P(6) \times P(6) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36} $$

<br>

## Testing for a Rare Disease
- Suppose you are a doctor and you need to test if a patient has a certain rare disease. 
- The test is very accurate: 
    - it’s correct 99% of the time. 
    - The disease is very rare: only 1 in 100,000 patients have it.

<br>

- You administer the test and it comes back positive, so your patient must have the disease, right?

<br>

- Not necessarily. If we just consider the test, there is only a 1% chance that it is wrong, but we actually have more information: 
    - we know how rare the disease is.

<br>

- Given that the test came back positive, there are two possibilities:
    1. The patient had the disease, and the test correctly diagnosed the disease.
    2. The patient didn’t have the disease and the test incorrectly diagnosed that they had the disease.

<br>

- In this context, a “positive result” means the test indicates the presence of the disease, regardless of whether the result is correct or not.
    1. **True Positive**: The patient has the disease, and the test correctly identifies it. This is a positive result because the test indicates the disease is present.
    2. **False Positive**: The patient does not have the disease, but the test incorrectly indicates they do. This is also considered a positive result because the test suggests the disease is present, even though it’s incorrect.
- Both scenarios result in the test showing a positive outcome.

<br>

- In medical testing, a “positive” result means the test indicates the presence of the disease. It doesn’t refer to the accuracy of the test.
    - True Positive: The test correctly indicates the disease is present.
    - False Positive: The test incorrectly indicates the disease is present when it is not.
- A “negative” result means the test indicates the absence of the disease, regardless of accuracy.

<br>

## Bayes' Theorem
- In the previous exercise, we determined two probabilities:
    1. The patient had the disease, and the test correctly diagnosed the disease ≈ 0.00001
    2. The patient didn’t have the disease and the test incorrectly diagnosed that they had the disease ≈ 0.01
- Both events are rare, but we can see that it was about 1,000 times more likely that the test was incorrect than that the patient had this rare disease.

<br>

- We’re able to come to this conclusion because we had more information than just the accuracy of the test; we also knew the prevalence of this disease.

<br>

- In statistics, if we have two events (`A` and `B`), we write the probability that event `A` will happen, given that event `B` already happened as` P(A|B)`. 
- In our example, we want to find `P(rare disease | positive result)`. 
- In other words, we want to find the probability that the patient has the disease given the test came back positive.
- We can calculate `P(A|B)` using **Bayes’ Theorem**, which states:
$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $$
- So in this case, we´d say:
$$ P(\text{rare disease} | \text{positive result}) = \frac{P(\text{positive result} | \text{rare disease}) \cdot P(\text{rare disease})}{P(\text{positive result})} $$
- It is important to note that on the right side of the equation, we have the term `P(B|A)`. 
- This is the probability that event `B` will happen given that event `A` has already happened.
- This is very different from `P(A|B)`, which is the probability we are trying to solve for. 
- **The order matters!**

<br>

## Spam Filters
- Let’s explore a different example. 
- Email spam filters use Bayes’ Theorem to determine if certain words indicate that an email is spam.
- Let’s take a word that often appears in spam: “enhancement”.
- With just 3 facts, we can make some preliminary steps towards a good spam filter:
    1. “enhancement” appears in just 0.1% of non-spam emails
    2. “enhancement” appears in 5% of spam emails
    3. Spam emails make up about 20% of total emails
- Given that an email contains “enhancement”, what is the probability that the email is spam?

<br>

--- 

<br>

## The Naive Bayes Classifier
- A Naive Bayes classifier is a supervised machine learning algorithm that leverages Bayes’ Theorem to make predictions and classifications. 
- Recall Bayes’ Theorem:
$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $$
- This equation is finding the probability of `A` given `B`. 
- This can be turned into a classifier if we replace `B` with a data point and `A` with a class. 
- For example, let’s say we’re trying to classify an `email` as either `spam` or `not spam`. 
- We could calculate `P(spam | email)` and `P(not spam | email)`. 
- Whichever probability is higher will be the classifier’s prediction.
- Naive Bayes classifiers are often used for text classification.

<br>

- So why is this a supervised machine learning algorithm? 
- In order to compute the probabilities used in Bayes’ theorem, we need previous data points. 
- For example, in the spam example, we’ll need to compute `P(spam)`.
- This can be found by looking at a tagged dataset of emails and finding the ratio of spam to non-spam emails.

<img src="Images/spam_classifier.gif" width="600" style="display: block; margin: auto;">

<br>

## Investigate the Data
- In this lesson, we are going to create a Naive Bayes classifier that can predict whether a review for a product is positive or negative. 
- This type of classifier could be extremely helpful for a company that is curious about the public reaction to a new product. 
- Rather than reading thousands of reviews or tweets about the product, you could feed those documents into the Naive Bayes classifier and instantly find out how many are positive and how many are negative.

<br>

- The dataset we will be using for this lesson contains Amazon product reviews for baby products. 
- The original dataset contained many different features including the reviewer’s name, the date the review was made, and the overall score. 
- We’ve removed many of those features; the only features that we’re interested in are the text of the review and whether the review was “positive” or “negative”. 
- We labeled all reviews with a score less than 4 as a negative review.

<br>

## Bayes Theorem I
- For the rest of this lesson, we’re going to write a classifier that can predict whether the review “This crib was amazing” is a positive or negative review. 
- We want to compute both `P(positive | review)` and `P(negative | review)` and find which probability is larger.
- To do this, we’ll be using Bayes’ Theorem. 
- Let’s look at Bayes’ Theorem for `P(positive | review)`.
$$ P(\text{positive} | \text{review}) = \frac{P(\text{review} | \text{positive}) \cdot P(\text{positive})}{P(\text{review})} $$
- The first part of Bayes’ Theorem that we are going to tackle is `P(positive)`. 
- This is the probability that any review is positive. 
- To find this, we need to look at all of our reviews in our dataset - both positive and negative - and find the percentage of reviews that are positive.
- We’ve bolded the part of Bayes’ Theorem we’re working on.
$$ P(\text{positive} | \text{review}) = \frac{P(\text{review} | \text{positive}) \cdot \textbf{P(\text{positive})}}{P(\text{review})} $$

<br>

## Bayes Theorem II
- Let’s continue to try to classify the review “This crib was amazing”.
- The second part of Bayes’ Theorem is a bit more extensive. 
- We now want to compute `P(review | positive)`.
$$ P(\text{positive} | \text{review}) = \frac{\textbf{P(\text{review} | \text{positive})} \cdot P(\text{positive})}{P(\text{review})} $$
- In other words, if we assume that the review is positive, what is the probability that the words “This”, “crib”, “was”, and “amazing” are the only words in the review?

<br>

- To find this, we have to assume that each word is conditionally independent. 
- This means that one word appearing doesn’t affect the probability of another word from showing up. This is a pretty big assumption!
- We now have this equation:
$$ P(\text{"This crib was amazing"} | \text{positive}) = P(\text{"This"} | \text{positive}) \cdot P(\text{"crib"} | \text{positive}) \cdot P(\text{"was"} | \text{positive}) \cdot P(\text{"amazing"} | \text{positive}) $$
- Let’s break this down even further by looking at one of these terms. 
- `P("crib"|positive)` is the probability that the word “crib” appears in a positive review. 
- To find this, we need to count up the total number of times “crib” appeared in our dataset of positive reviews. 
- If we take that number and divide it by the total number of words in our positive review dataset, we will end up with the probability of “crib” appearing in a positive review.
$$ P(\text{"crib"} | \text{positive}) = \frac{\text{\# of "crib" in positive}}{\text{\# of words in positive}} $$
- If we do this for every word in our review and multiply the results together, we have `P(review | positive)`.

<br>

## Smoothing
- In the last exercise, one of the probabilities that we computed was the following:
$$ P(\text{"crib"} | \text{positive}) = \frac{\text{\# of "crib" in positive}}{\text{\# of words in positive}} $$
- But what happens if “crib” was never in any of the positive reviews in our dataset? 
- This fraction would then be 0, and since everything is multiplied together, the entire probability `P(review | positive)` would become 0.

<br>

- This is especially problematic if there are typos in the review we are trying to classify. 
- If the unclassified review has a typo in it, it is very unlikely that that same exact typo will be in the dataset, and the entire probability will be 0. 
- To solve this problem, we will use a technique called *smoothing*.

<br>

- In this case, we smooth by adding 1 to the numerator of each probability and `N` to the denominator of each probability. 
- `N` is the number of unique words in our review dataset.
- For example, `P("crib" | positive)` goes from this:
$$ P(\text{"crib"} | \text{positive}) = \frac{\text{\# of "crib" in positive}}{\text{\# of words in positive}} $$
- To this:
$$ P(\text{"crib"} | \text{positive}) = \frac{\text{\# of "crib" in positive} + 1}{\text{\# of words in positive} + N} $$

<br>

## Classify
- If we look back to Bayes’ Theorem, we’ve now completed both parts of the numerator. 
- We now need to multiply them together.
$$ P(\text{positive} | \text{review}) = \frac{\textbf{P(\text{review} | \text{positive})} \cdot \textbf{P(\text{positive})}}{P(\text{review})} $$
- Let’s now consider the denominator `P(review)`. 
- In our small example, this is the probability that “This”, “crib”, “was”, and “amazing” are the only words in the review. 
- Notice that this is extremely similar to `P(review | positive)`. 
- The only difference is that we don’t assume that the review is positive.

<br>

- However, before we start to compute the denominator, let’s think about what our ultimate question is. 
- We want to predict whether the review “This crib was amazing” is a positive or negative review. 
- In other words, we’re asking whether `P(positive | review)` is greater than `P(negative | review)`. 
- If we expand those two probabilities, we end up with the following equations.
$$ P(\text{positive} | \text{review}) = \frac{P(\text{review} | \text{positive}) \cdot P(\text{positive})}{P(\text{review})} $$
$$ P(\text{negative} | \text{review}) = \frac{P(\text{review} | \text{negative}) \cdot P(\text{negative})}{P(\text{review})} $$
- Notice that `P(review)` is in the denominator of each. 
- That value will be the same in both cases! 
- Since we’re only interested in comparing these two probabilities, there’s no reason why we need to divide them by the same value. 
- We can completely ignore the denominator!

<br>

## Formatting the Data for `scikit-learn`
- Congratulations! You’ve made your own Naive Bayes text classifier. 
- If you have a dataset of text that has been tagged with different classes, you can give your classifier a brand new document and it will predict what class it belongs to.
- We’re now going to look at how Python’s scikit-learn library can do all of that work for us!

<br>

- In order to use scikit-learn’s Naive Bayes classifier, we need to first transform our data into a format that scikit-learn can use. 
- To do so, we’re going to use scikit-learn’s `CountVectorizer` object.
- To begin, we need to create a `CountVectorizer` and teach it the vocabulary of the training set. 
- This is done by calling the `.fit()` method.

<br>

- For example, in the code below, we’ve created a `CountVectorizer` that has been trained on the vocabulary `"Training"`, `"review"`, `"one"`, and `"Second"`.
    ```python
    vectorizer = CountVectorizer()

    vectorizer.fit(["Training review one", "Second review"])
    ```
- After fitting the vectorizer, we can now call its `.transform()` method. 
- The `.transform()` method takes a list of strings and will transform those strings into counts of the trained words. 
- Take a look at the code below.
    ```python
    counts = vectorizer.transform(["one review two review"])
    ```
- `counts` now stores the array `[[1 2 0 0]]`. 
- The word `"review"` appeared twice, the word `"one"` appeared once, and neither `"Training"` nor `"Second"` appeared at all.
- But how did we know that the `2` corresponded to `review`?
- You can print `vectorizer.vocabulary_` to see the index that each word corresponds to. 
- It might look something like this:
    ```python
    {'Training': 3, 'review': 1, 'one': 0, 'Second': 2}
    ```
- Finally, notice that even though the word `"two"` was in our new review, there wasn’t an index for it in the vocabulary.
-  This is because `"two"` wasn’t in any of the strings used in the `.fit()` method.
- We can now use `counts` as input to our Naive Bayes Classifier.

<br>

## Using `scikit-learn`
- Now that we’ve formatted our data correctly, we can use it using scikit-learn’s `MultinomialNB` classifier.
- This classifier can be trained using the `.fit()` method. 
- `.fit()` takes two parameters: 
    - The array of data points (which we just made) and 
    - an array of labels corresponding to each data point.

<br>

- Finally, once the model has been trained, we can use the `.predict()` method to predict the labels of new points. 
- `.predict()` takes a list of points that you want to classify and it returns the predicted labels of those points.

<br>

- Finally, `.predict_proba()` will return the probability of each label given a point. 
- Instead of just returning whether the review was good or bad, it will return the likelihood of a good or bad review.