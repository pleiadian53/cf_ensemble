# Meta-learning via Latent-Factor-Based Collaborative Filtering

Ensemble learning is a family of meta-algorithms that combine several base models into one predictive model in order to decrease variance (e.g. bagging) and generalization errors (e.g. stacking). Heterogeneous ensembles, in particular, make the algorithmic diversity explicit by introducing base models that generate decision boundaries with varying properties that, when combined, can be empirically shown to adapt better to complex data types, such as biological data, in terms of the predictive performance and reliability. In general, the ensemble learning process can be divided into two main stages: i) ensemble generation, where multiple base models are generated, and ii) ensemble integration, where base-level predictions are pruned and combined, leading to a multitude of methodologies – ensemble selections, stacked generalization, and Bayesian model combination, being among the prime examples. In this work, we will introduce an intermediate stage, **ensemble transformation for classification**, that precedes the ensemble integration stage. This is an additional layer of meta-learning through transforming the probability predictions at the base level toward increasing the reliability of their class conditional probability estimates, which then potentially benefit the ensemble integration stage as well as providing additional features such as model interpretation, often desirable in the context of biomedical data analytics.  

## Introduction
Classification problems in biomedical domains are challenging due to the complexity inherent in the relationship between biological concepts and explanatory variables as well as the lack of consensus in best classifiers, being problem-dependent. From data science perspective, the contributing factors to the modeling difficulties largely arise from prevalence of class imbalance, missing values, imprecise measurements introducing noises, and far-from-ideal predictor variables due to incomplete knowledge of the underlying processes per se. 

The ensemble classifier learning approach attempts to establish a balance between diversity and accuracy of the base classifiers. Heterogeneous ensembles in particular make the algorithmic diversity explicit by introducing classifiers that generate decision boundaries with varying properties that, when combined, can be empirically shown to adapt better to the biological datasets in terms of the predictive performance. 

Generally, the ensemble learning process can be divided into two main stages: i) **ensemble generation**, where multiple base models are generated, and ii) **ensemble integration**, where base-level predictions are pruned and combined, leading to a multitude of methodologies – ensemble selections, stacked generalization (aka stacking), and Bayesian model combination, being among the prime examples. In this work, we will introduce an intermediate stage, **ensemble transformation** (for classification), that precedes the ensemble integration stage. 

More precisely, this is an additional layer of [meta-learning](https://en.wikipedia.org/wiki/Meta_learning_(computer_science)) through transforming the probability predictions at the base level toward increasing the reliability of their class conditional probability estimates, which then potentially benefit the ensemble integration stage as well as providing additional features such as model interpretation, often desirable in the context of biomedical data analytics. For instance, identifying group structures in the training data in contexts of how they are being classified by the ensemble helps to identify the homogeneous subsets that share similar biomedical properties as well as training instances that would potentially pose challenges to the ensemble. To achieve both predictive performance improvements and model interpretability, here we will demonstrate a **latent factor-based collaborative filtering (CF) approach** to realizing the ensemble transformation stage. 

For more details, please go through [this introductory document](CF-EnsembleLearning-Intro.pdf). 

Optionally, you could also go through the [slides](https://www.slideshare.net/pleiadian53/metalearning-via-latentfactorbased-collaborative-filtering-252872052). 

This work is currently still under development. To see prototypes and examples, please go through the notebook series 1-5: 

1. [Optimization via alternating least sequare (ALS)](Demo-Part1-CF_with_ALS.ipynb)
2. [The role of loss function in CF ensemble learning](Demo-Part2-The_Role_of_Loss_Function_in_CF_Ensemble.ipynb)
3. [CF ensemble with K nearest neighbors](Demo-Part3-CF_Ensemble_with_kNNs.ipynb)
4. [CF ensemble for stacked generalization](Demo-Part4-CF_Stacker.ipynb)
5. [CF ensemble with probability filtering and sequence models](Demo-Part5b-Probability_Filtering_via_Custom_Loss.ipynb)


## Basic Workflow

<img width="328" alt="image" src="https://user-images.githubusercontent.com/1761957/188764919-f2217d9f-c451-4c51-9b34-cde9f8cdc7b4.png">


## Optimization 

In CF-based meta learning, base models play the role of “users” _{u}_ while data points play the role of “items” _{i}_, where we have borrowed a convention from recommender systems by using _u_ to denote a classifier (as a **u**ser) and _i_ to denote a data point (as an **i**tem). To draw an analogy between **binary classification** and **recommender system**, we can think of a classifier, when making predictions, as assigning "ratings" on data items in the sense that the rating tells us how likely a data point is positive (_y=1_) in terms of a conditional probability score (_P(y=1|x)_). If the classifier is very confident that a data item (x) should be classified as positive, then we would expect to observe a high rating (close or equal to 1); by contrast, the rating is expected to be relatively low (close or equal to 0) for data items associated with the negative class. To realize an effective ensemble transformation (see [this](CF-EnsembleLearning-Intro.pdf)), we seek to solve a family of optimization problems, $argmin_{X,Y} J(X, Y)$, where _J(X,Y)_ is a **cost function**, on latent matrices X and Y, that takes the following form:

![image](https://user-images.githubusercontent.com/1761957/188937553-e74e9837-51cf-4c7e-8ef9-66146ceb8d95.png)

Please refer to [this document](CFEnsembleLearning-optimization.pdf) for more details on the notation and the interpretation. 

<br />... to be continued

