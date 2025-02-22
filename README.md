# 基于机器学习的缺失抽样调查数据回归插补法研究

![image](https://github.com/user-attachments/assets/ceccca52-df5a-468f-9823-cee7d0ec9e80)


## 本文研究方法：

### 1. 抽样方法

- 简单随机抽样（SRS）：包括不放回抽样和有放回抽样，用于比较其他复杂抽样方法的效果。
    
- 分层随机抽样：以“班级”为分层变量，结合Neyman最优分配公式，提高估计精度。
    
- 不等概率抽样：

 - PPS抽样（与辅助变量成比例的概率抽样）。
 - πPS抽样（严格不等概率抽样，采用Brewer方法）。
   
- 整群抽样（两阶段抽样）：以“教师”为群，第一阶段采用不等概率抽样，第二阶段采用简单随机抽样。

- 系统抽样：基于辅助变量的排序（如Test7成绩的校排名），采用圆形等距抽样和改进方法（如对称系统抽样）。

### 2. 回归插补方法

- 比估计和回归估计：利用辅助变量（如Test7成绩）进行比估计和回归估计。
  
- Lasso回归：处理多重共线性问题，用于回归插补。
  
- LightGBM回归：基于机器学习的回归插补方法，允许自变量缺失，适用于复杂数据结构。

## 本文研究结构：

### 数据集介绍与分析

- 数据来源：某高中高三年级的数学成绩数据集，包含高考成绩和7次考试成绩，由于数据敏感性暂不公开数据源及相关代码。
  
- 数据特性：纵向数据，存在缺失值，目标变量为高考数学成绩。
  
- 数据分析：通过可视化和统计分析，了解数据分布和辅助变量与目标变量的关系。
  
### 经典抽样方法回顾与设置

详细介绍和实现上述各种抽样方法，并通过随机模拟实验比较其精度。

- 抽样方法对比：统一设置样本量，比较不同抽样方法的估计精度（通过deff值）。

- 回归插补模型建立与随机模拟实验：分析缺失数据类型，讨论辅助变量和目标变量的缺失处理方法，构建Lasso回归和LightGBM回归模型，评估插补后的抽样估计精度和方差。


回顾文中和本学期学习的大多结论，都能体悟到各种抽样方法构造的**巧妙**之处。我们往往可以发现，有效的抽样方法本质上都是如何高效利用总体的“**先验信息**”的方法，有的是通过先验信息施行**更精细化的“抽样步骤”**（如分层抽样、系统抽样等），有的是通过先验信息实现对“统计推断”的**纠偏修正**（如比估计、回归估计、不等概率抽样等），从而降低最终估计的方差。

 作为一篇较为匆忙的大三课程设计，本文为了涵盖了大多本学期学到的抽样方法对各类方法的应用总觉有“刻意而为”、“勉强能用”的不自然，构造的方法步骤也许并不精确和严谨，只希望能够提高知识的掌握程度，并锻炼一下实践能力。文末最后提及的机器学习预测方法也只是一个简单的应用和对比，由于时间因素未能进行细致的调整和在不同样本量、不同情况下综合对比，**望读者海涵**。

 抽样调查这门课程学习的目标最终还是导向现实生活中我们如何构造方法的实践任务，对于纯理论的推导和习题零零散散的例子不能很好地让初学者理解各类方法的优劣比较，只是笼统地了解某些方法“似乎”很好，也难以直观理解各类方法的缺陷与使用限制。
 
 具体来说，当我们实际要考虑“如何抽样”时，才会开始思考诸多方法的便捷性和限制性，许多本在习题中指定应用某类方法、或者提供好某类辅助变量的情况在现实中并不常见，人们往往需要更深刻地了解要抽样的总体具有什么样的“特性”，才能构造真正合适的抽样方法。在本文限制为同一数据集下、全方位地使用各类抽样方法来对比实验中，采用若干方法极其显著地提高估计精度、降低方差时，这种收获的喜悦让人清晰明确地体会到“抽样方法”的好与坏，也明白这些统计思想背后蕴含的实际效益，“设计”一个优秀的抽样调查方法不仅是一个学术性的研究，更在某种程度上是一种“**对随机性理解的精妙艺术**”。

---

## Machine Learning Based Regression Interpolation Method for Missing Sample Survey Data

### This paper studies the methodology:

### 1. Sampling method

- Simple Random Sampling (SRS): including non-return sampling and return sampling, used to compare the effect of other complex sampling methods.
    
- Stratified Random Sampling (SRS): using “class” as the stratified variable, combined with Neyman's optimal allocation formula to improve the estimation accuracy.
    
- Unequal probability sampling:

 - PPS sampling (probability sampling proportional to auxiliary variables).
 - πPS sampling (strictly unequal probability sampling with Brewer's method).
   
- Cluster sampling (two-stage sampling): using “teachers” as a cluster, unequal probability sampling is used in the first stage, and simple random sampling is used in the second stage.

- Systematic sampling: based on the ranking of auxiliary variables (e.g., school ranking of Test 7 scores), using circular equidistant sampling and modified methods (e.g., symmetric systematic sampling).

### 2. Regression interpolation methods

- Ratio and regression estimation: ratio and regression estimation using auxiliary variables (e.g., Test7 scores).
  
- Lasso regression: deals with multicollinearity problems and is used for regression interpolation.
  
- LightGBM regression: machine learning-based regression interpolation method that allows for missing independent variables for complex data structures.

### This paper studies the structure:

### Data set introduction and analysis

- Data source: a high school senior math performance dataset, including the results of the college entrance examination and seven exams, due to the sensitivity of the data not to disclose the data source and the relevant code for the time being.
  
- Data Characteristics: Longitudinal data, with missing values, and the target variable is high school math scores.
  
- Data analysis: visualization and statistical analysis to understand the data distribution and the relationship between auxiliary variables and target variables.
  
### Review and setup of classical sampling methods

Detailed introduction and implementation of the various sampling methods mentioned above, and compare their accuracy through random simulation experiments.

- Comparison of sampling methods: set up sample sizes in a uniform manner and compare the estimation accuracy (via deff values) of different sampling methods.

- Regression interpolation model building and stochastic simulation experiments: analyze the types of missing data, discuss the treatment of missing auxiliary and target variables, construct Lasso regression and LightGBM regression models, and evaluate the sampling estimation accuracy and variance after interpolation.


Reviewing the text and most of the conclusions from this semester's study, it is possible to appreciate the **clever** aspects of the construction of various sampling methods. We often find that effective sampling methods are essentially ways to efficiently utilize the “**prior information**” of the population, either through the implementation of **more refined "sampling steps"** (e.g., stratified, systematic, etc.) or through the implementation of **more refined ’sampling steps ”** (e.g., stratified, systematic, etc.) with the a priori information. ), or through a priori information to realize the “statistical inference” **correction correction** (such as ratio estimation, regression estimation, unequal probability sampling, etc.), so as to reduce the variance of the final estimate.

 As a relatively hasty junior course design, this paper in order to cover most of the sampling methods learned this semester on the application of various methods always feel “deliberate”, “barely able to use” is not natural, the construction of the method of step may not be precise and rigorous, only I hope to improve my knowledge and practice my practical skills. The machine learning prediction method mentioned at the end of the article is just a simple application and comparison, due to time factors can not be carefully adjusted and in different sample sizes, under different circumstances, comprehensive comparison, **hope that the reader connotation**.

 Sample survey of this course of study is ultimately oriented to real life how we construct methods of practical tasks, for purely theoretical derivations and exercises scattered examples can not be very good for beginners to understand the advantages and disadvantages of various types of methods of comparison, just a general understanding of some of the methods “seem” to be very good, and it is difficult to intuitively understand the method of each type of It is also difficult to intuitively understand the shortcomings and limitations of each method.
 
 Specifically, when we actually have to consider “how to sample”, we will begin to think about the convenience and limitations of many methods, many of the exercises in the specified application of a certain type of method, or provide a good auxiliary variables of a certain type of situation in reality is not common, people often need to understand more deeply to sample the overall what kind of “characteristics”. “characteristics” in order to construct a truly appropriate sampling method. In this paper, we restrict ourselves to using a full range of sampling methods under the same data set to compare experiments, and when several methods are used to improve the estimation accuracy and reduce the variance in a very significant way, the joy of harvesting makes people clearly realize the good and bad of the “sampling methods”, and also understand the practical benefits behind these statistical ideas, and “designing” a “sampling method” is not always easy. The “design” of a good sampling method is not only an academic study, but in a way a “**subtle art of understanding randomness**”.

