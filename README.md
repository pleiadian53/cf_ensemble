# Collaborative Filtering-Enhanced Ensemble Learning

Classification problems in biomedical domains are challenging due to the complexity inherent in the relationship between biological concepts and explanatory variables as well as the lack of consensus in best classifiers, being problem-dependent. From data science perspective, the contributing factors to the modeling difficulties largely arise from prevalence of class imbalance, missing values, imprecise measurements introducing noises, and far-from-ideal predictor variables due to incomplete knowledge of the underlying processes per se. 

Ensemble classifiers are meta-algorithms that combine several classification algorithms into one predictive model in order to decrease variance (e.g. bagging) and generalization errors (e.g. stacking). Heterogeneous ensembles in particular make the algorithmic diversity explicit by introducing classifiers that generate decision boundaries with varying properties that, when combined, can be empirically shown to adapt better to the biological datasets in terms of the predictive performance. 

Without loss of generality, the ensemble learning process can be divided into two main stages: i) ensemble generation, where multiple base models are generated, and ii) ensemble integration, where base-level predictions are pruned and combined, leading to a multitude of methodologies – ensemble selections, stacked generalization (aka stacking), and Bayesian model combination, being among the prime examples. In this repos, we will introduce an intermediate stage, ensemble transformation, that precedes the ensemble integration stage. 

More precisely, this is an additional layer of meta-learning through transforming the probability predictions at the base level toward increasing the reliability of their class conditional probability estimates, which then potentially benefit the ensemble integration stage as well as providing additional features such as model interpretation, often desirable in the context of biomedical data analytics. For instance, identifying group structures in the training data in contexts of how they are being classified by the ensemble helps to identify the homogeneous subsets that share similar biomedical properties as well as training instances that would potentially pose challenges to the ensemble. To achieve both predictive performance improvements and model interpretability, here we will demonstrate a **latent factor-based collaborative filtering (CF) approach** to realizing the ensemble transformation stage. 


For more details and example prototypes, please go through Demo 1-5 (Juypter notebooks) starting from `Demo-Part1-CF_with_ALS`. 

... to be continued

