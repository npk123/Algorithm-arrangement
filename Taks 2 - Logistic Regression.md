[1.逻辑回归与线性回归的联系与区别](#1)

　[1.1联系](#2)
  
　[1.2区别](#3)
  
[2.逻辑回归的原理](#4)

[3.逻辑回归损失函数推导及优化](#5)

[4.正则化与模型评估指标](#6)

　[4.1 正则化](#7)
 
　[4.2 模型评估指标](#8)
  
[5.逻辑回归的优缺点](#9)

　[5.1 优点](#10)
 
　[5.2 缺点](#11)
  
[6.样本不均衡问题解决办法](#12)

[7.sklearn参数](#13)

[8.LR与SVM](#14)







<h2 id='1'> 1. 逻辑回归与线性回归的联系与区别 </h2>

<h3 id='2'> 1.1 联系 </h3>

线性回归：根据几组已知数据和拟合函数训练其中未知参数，使得拟合损失达到最小。然后用所得的拟合函数进行预测。 

逻辑回归：和拟合函数训练其中未知参数使得对数似然函数最大。然后用所得的拟合函数进行二分类。 

两者都是回归，步骤和原理看起来相似。但线性回归的应用场合大多是回归分析，一般不用在分类问题上。原因可以概括为以下两个：

    1.回归模型是连续型模型，即预测出的值都是连续值（实数值），非离散值；
    2.预测结果受样本噪声的影响比较大。

<h4 id='3'> 1.2 区别 </h4>

如下图所示
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture.JPG)

1. 拟合函数和预测函数的关系，其实就是将拟合函数做了一个逻辑函数的转换。 

2. 最小二乘和最大似然估计不可以相互替代。原理：最大似然估计是计算使得数据出现的可能性最大的参数，依仗的自然
是Probability。而最小二乘是计算误差损失。

区别：

    1.线性回归要求变量服从正态分布，logistic回归对变量分布没有要求。
    2.线性回归要求因变量是连续性数值变量，而logistic回归要求因变量是分类型变量。
    3.线性回归要求自变量和因变量呈线性关系，而logistic回归不要求自变量和因变量呈线性关系
    4.logistic回归是分析因变量取某个值的概率与自变量的关系，而线性回归是直接分析因变量与自变量的关系

总之, logistic回归与线性回归实际上有很多相同之处，最大的区别就在于他们的因变量不同，其他的基本都差不多，
正是因为如此，这两种回归可以归于同一个家族，即广义线性模型（generalized linear model）。这一家族中的模
型形式基本上都差不多，不同的就是因变量不同，如果是连续的，就是多重线性回归，如果是二项分布，就是logistic回归。
logistic回归的因变量可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释。所以实际中最为常用的就是二分类的logistic回归。

<h2 id='4'> 2. 逻辑回归的原理 </h2>

逻辑回归是应用非常广泛的一个分类机器学习算法，它将数据拟合到一个logit函数(或者叫做logistic函数)中，从而能够完成对事件发生的概率进行预测。

它的核心思想是，如果线性回归的结果输出是一个连续值，而值的范围是无法限定的，那我们有没有办法把这个结果值映射为可以帮助我们判断的结果呢。而如果输出结果是 (0,1) 的一个概率值，这个问题就很清楚了。数学上使用便是sigmoid函数(如下)：

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/sigmoid%20function.jpg)
  
它的输入范围为−∞→+∞，而值域刚好为(0,1)，正好满足概率分布为(0,1)的要求。用概率去描述分类器，自然要比阈值要来的方便。
而且它是一个单调上升的函数，具有良好的连续性，不存在不连续点。
  
  其求导后为
  
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/sigmoid'.jpg)

<h2 id='5'> 3. 逻辑回归损失函数推导及优化 </h2>

损失函数

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/logistic-loss%20function.jpg)
  
简化过后的损失函数如下
  
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/simplified%20logistic%20loss%20function.png)
  
从数学上理解，我们为了找到最小值点，就应该朝着下降速度最快的方向(导函数/偏导方向)迈进，每次迈进一小步，再看看此时的下降最快方向是哪，再朝着这个方向迈进，直至最低点。

用迭代公式表示出来的最小化J(θ)的梯度下降算法如下：
  
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/%E4%B8%8B%E8%BD%BD1.png)
  
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/%E4%B8%8B%E8%BD%BD2.png)
  
<h2 id='6'> 4. 正则化与模型评估指标 </h2>

<h3 id='7'> 4.1 正则化 </h3>

当模型的参数过多时，很容易遇到过拟合的问题。这时就需要有一种方法来控制模型的复杂度，典型的做法在优化目标中加入正则项，
通过惩罚过大的参数来防止过拟合.
 
 ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Regularization-logistic%20regression.JPG)

一般情况下，取p=1或p=2，分别对应L1，L2正则化，两者的区别可以从下图中看出来，L1正则化（左图）倾向于使参数变为0，因此能产生稀疏解。

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Regularization%20result%201-logistic%20regression.png)
  
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Regularization%20result%202-logistic%20regression.png)

实际应用时，由于我们数据的维度可能非常高，L1正则化因为能产生稀疏解，使用的更为广泛一些。

<h3 id='8'> 4.2 模型评估指标 </h3>

混淆矩阵

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Confusion%20Matrix.png)

    真正例（True Positive，TP）：真实情况是"买"，预测结果也是"买"，TP表示正例预测成功次数。
    假正例（False Positive，FP）：真实情况是"不买"，预测结果也是"买"，FP表示负例预测错误次数。
    假负例（False Negative，FN）：真实情况是"买"，预测结果也是"不买"，FN表示正例预测错误次数。
    真负例（True Negative，TN）：真实情况是"不买"，预测结果也是"不买"，TN表示负例预测成功次数。

常用的衍生指标。

    模型精度（Accuracy）：Accuracy = (TP + TN) / (TP + FP + TN + FN)
    一般情况下，模型的精度越高，说明模型的效果越好。
    准确率，又称查准率（Precision，P）：Precision = TP / (TP + FP)
    一般情况下，查准率越高，说明模型的效果越好。
    召回率，又称查全率（Recall，R）：Recall = TP / (TP + FN)
    一般情况下，Recall越高，说明有更多的正类样本被模型预测正确，模型的效果越好。
    F1值（F1-Score）：F1 Score = P*R/2(P+R)，其中P和R分别为 precision 和 recall
    一般情况下，F1-Score越高，说明模型的效果越好。

 另有以下几个图形指标

    真阳性率（True Positive Rate，TPR )：TPR = TP / (TP+FN)
    等于Recall，一般情况下，TPR 越高，说明有更多的正类样本被模型预测正确，模型的效果越好。
    假阳性率（False Positive Rate，FPR ）：FPR= FP / (FP+TN)
    一般情况下，FPR 越低，说明有更多的负类样本被模型预测正确，模型的效果越好。
    
    ROC（Receiver Operating Characteristic）：ROC曲线的纵坐标为 TPR，横坐标为 FPR
    AUC（Area Under Curve）：ROC曲线下的面积，很明显，AUC 的结果不会超过 1，通常 ROC 曲线都在 y = x 这条直线上面，所以，AUC 的值一般在 0.5 ~ 1 之间，面积越大模型效果越好。
    
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/ROC-AUC.png)

<h2 id='9'> 5. 逻辑回归的优缺点 </h2>

<h3 id='10'> 5.1优点 </h3>

    1.LR是以概率的形式输出结果，不只是0和1的判定； 
    2.LR的可解释强，可控性高； 
    3.训练快，feature engineering之后效果赞； 
    4.因为结果是概率，可以做ranking model； 
    5.添加feature简单。 
    
<h3 id='11'> 5.2缺点 </h3>

    1.容易欠拟合，分类精度不高;
    2.数据特征有缺失或者特征空间很大时表现效果并不好。
   
LR的应用场景很多哈： 

    1.CTR预估、推荐系统的learning to rank； 
    2.一些电商搜索排序基线； 
    3.一些电商的购物搭配推荐； 
    4.新闻app排序基线。

<h2 id='12'> 6. 样本不均衡问题解决办法 </h2>

样本太大怎么处理？ 

    1.对特征离散化，离散化后用one-hot编码处理成0,1值，再用LR处理会较快收敛； 
    2.如果一定要用连续值的话，可以做scaling； 
    3.工具的话有 spark Mllib，它损失了一小部分的准确度达到速度的提升； 
    4.如果没有并行化平台，想做大数据就试试采样。需要注意采样数据，最好不要随机取，可以按照日期/用户/行为，来分层抽样。 

怎么使样本平衡？ 

    1.如果样本不均衡，样本充足的情况下可以做下采样——抽样，样本不足的情况下做上采样——对样本少的做重复； 
    2.修改损失函数，给不同权重。比如负样本少，就可以给负样本大一点的权重； 
    3.采样后的predict结果，用作判定请还原。

<h2 id='13'> 7. sklearn参数 </h2>

    penalty:正则化选择参数，参数可选值为l1和l2，分别对应l1正则化和l2正则化，默认是l2正则化。
    dual:用来指明是否将原问题改成他的对偶问题，对偶问题可以理解成相反问题，比如原问题是求解最大值的线性规划，那么他的对偶问题就是转化为求解最小值的线性规划，适用于样本较小的数据集，因样本小时，计算复杂度较低。
    tol:残差收敛条件，默认是0.0001，也就是只需要收敛的时候两步只差＜0.0001就停止，可以设置更大或更小。(逻辑回归模型的损失函数是残差平方和)
    C:正则化系数，正则化强度的导数，必须是一个正数，值越小，正则化强度越大，即防止过拟合的程度更大。
    fit_intercept:是否将截距/方差加入到决策模型中，默认为True。
    class_weight:class_weight是很重要的一个参数，是用来调节正负样本比例的，默认是值为None，也就是正负样本的权重是一样的，你可以以dict的形式给模型传入任意你认为合适的权重比，也可以直接指定一个值“balanced”，模型会根据正负样本的绝对数量比来设定模型最后结果的权重比。
    random_state:随机种子的设置，默认是None,如果设置了随机种子，那么每次使用的训练集和测试集都是一样的，这样不管你运行多少次，最后的准确率都是一样的；如果没有设置，那么每次都是不同的训练集和测试集，最后得出的准确率也是不一样的。
    solver:用来指明损失函数的优化方法，默认是‘liblinear’方法，sklearn自带了如下几种：
    
newton-cg, lbfgs和sag这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear对L1正则化和L2正则化都适用。同时，因sag每次仅仅使用了部分样本进行梯度迭代，所以当数据量较少时不宜选用，而当数据量很大时，为了速度，sag是第一选择。

    max_iter:算法收敛的最大迭代次数，即求取损失函数最小值的迭代次数，默认是100，multi_class:分类方法参数选择，‘ovr’和‘multinomial’两个值可以选择，默认值为‘ovr’，如果分类问题是二分类问题，那么这两个参数的效果是一样的，主要体现在多分类问题上。

对于多分类问题，"ovr"分类方法是：针对每一类别进行判断时，都会把这个分类问题简化为是/非两类问题；而‘multinomial’是从众多类别中选出两个类别，对这两个类别进行判断，待判断完成后，再从剩下的类别中再选出两类进行判断，直至最后判断完成。

    verbose:英文意思是"冗余"，就是会输出一些模型运算过程中的东西（任务进程），默认是False，也就是不需要输出一些不重要的计算过程。
    n_jobs:并行运算数量(核的数量)，默认为1，如果设置为-1，则表示将电脑的cpu全部用上。

模型对象

    coef_:返回各特征的系数,绝对值大小可以理解成特征重要性
    intercept_:返回模型的截距n_iter_:模型迭代次数
    
模型方法

    decision_function(X):返回决策函数值（比如svm中的决策距离）predict_proba(X):返回每个类别的概率值（有几类就返回几列值）predict_log_proba(X):返回概率值的log值（即将概率取对数）
    predict(X)：返回预测结果值（0/1）
    score(X, y=None):返回函数
    get_params(deep=True):返回估计器的参数
    set_params(**params):为估计器设置参数

<h2 id='14'> 8. LR与SVM </h2>

两种方法都是常见的分类算法，从目标函数来看，区别在于逻辑回归采用的是logistical loss，svm采用的是hinge loss。这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。SVM的处理方法是只考虑support vectors，也就是和分类最相关的少数点，去学习分类器。而逻辑回归通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。两者的根本目的都是一样的。此外，根据需要，两个方法都可以增加不同的正则化项，如l1,l2等等。所以在很多实验中，两种算法的结果是很接近的。但是逻辑回归相对来说模型更简单，好理解，实现起来，特别是大规模线性分类时比较方便。而SVM的理解和优化相对来说复杂一些。但是SVM的理论基础更加牢固，有一套结构化风险最小化的理论基础，虽然一般使用的人不太会去关注。还有很重要的一点，SVM转化为对偶问题后，分类只需要计算与少数几个支持向量的距离，这个在进行复杂核函数计算时优势很明显，能够大大简化模型和计算量。

两者对异常的敏感度也不一样。同样的线性分类情况下，如果异常点较多的话，无法剔除，首先LR，LR中每个样本都是有贡献的，最大似然后会自动压制异常的贡献，SVM+软间隔对异常还是比较敏感，因为其训练只需要支持向量，有效样本本来就不高，一旦被干扰，预测结果难以预料。
