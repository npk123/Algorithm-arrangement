1. 信息论基础（熵 联合熵 条件熵 信息增益 基尼不纯度）

1.1 熵 （entropy）

在信息论与概率统计中，熵表示随机变量不确定性的度量。设X是一个取有限个值得离散随机变量，其概率分布为

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture.JPG)

则随机变量X的熵定义为

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture1.JPG)

若pi等于0，定义0log0=0，熵的单位为比特或者纳特。

1.2 联合熵

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture4.JPG)

举个例子，更容易理解一些，比如天气是晴天还是阴天，和我穿短袖还是长袖这两个事件其可以组成联合信息熵H(X,Y)H(X,Y)，而对于H(x)H(x)就是天气单独一个事件的信息熵，因为两个事件组合起来的信息量肯定是大于单一一个事件的信息量的。 

而今天天气和我今天穿衣服这两个随机概率事件并不是独立分布的，所以如果已知今天天气的情况下，我的穿衣与天气的联合信息量/不确定程度是减少了的，也就相当于两者联合信息量已知了今天下雨，那么H(x)H(x)的信息量就应该被减去，得到当前的新联合信息量，也相当于条件信息量。 

所以当已知H(x)H(x)这个信息量的时候，联合分布H(X,Y)H(X,Y)剩余的信息量就是条件熵。

Ref：https://blog.csdn.net/gangyin5071/article/details/82228827#5%E8%81%94%E5%90%88%E4%BF%A1%E6%81%AF%E7%86%B5%E5%92%8C%E6%9D%A1%E4%BB%B6%E4%BF%A1%E6%81%AF%E7%86%B5

1.3 条件熵 (Conditional Entropy)

H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性定义为X给定条件下Y的条件概率分布的熵对X的数学期望

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture2.JPG)

经验熵和经验条件熵：当熵和条件熵中的概率由数据估计（特别是极大似然估计）得到时，所对应的熵与条件熵分别称为经验熵和条件经验熵。

1.4 信息增益（Information Gain）

信息增益表示得知特征X的信息而使得类Y的信息的不确定性减少的程度。特征A对训练数据集D的信息增益g(D,A)，定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差，即

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture3.JPG)

一般地，熵H(Y)与条件熵H(Y|X)之差称为互信息。决策树学习中的信息增益等价于训练数据集中类与特征的互信息。

于是我们可以应用信息增益准则来选择特征，信息增益表示由于特征A而使得对数据集D的分类的不确定性减少的程度。对数据集D而言，信息增益依赖于特征，不同的特征往往具有不同的信息增益。信息增益大的特征具有更强的分类能力。

1.5 基尼不纯度 (Gini impurity)

将来自集合中的某种结果随机应用于集合中某一数据项的预期误差率。

在CART(Classification and Regression Tree)算法中利用基尼不纯度构造二叉决策树。 

假设这个数据集里有k种不同标签，第i个标签所占的比重为pi，那么Gini impurity为

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture5.JPG)

它描述了一个数据集中标签分布的不纯度，类似于entropy。

2.决策树的不同分类算法（ID3算法、C4.5、CART分类树）的原理及应用场景

2.1 ID3算法 原理及应用场景

核心思想：以信息增益为度量，选择分裂后信息增益最大的特征进行分裂

遍历所有特征，对于特征A：

    1.计算特征A对数据集D的经验条件熵H(D|A)
    2.计算特征A的信息增益：
    g(D,A)=H(D) – H(D|A)
    3.选择信息增益最大的特征作为当前的分裂特征
    4.迭代

ID3优点是理论清晰、方法简单、学习能力较强，但也在应用中存在限制：

    （1）只能处理分类属性的数据，不能处理连续的数据；
    （2）划分过程会由于子集规模过小而造成统计特征不充分而停止；
    （3）ID3算法在选择根节点和各内部节点中的分支属性时，采用信息增益作为评价标准。信息增益的缺点是倾向于选择取值较多的属性，在有些情况下这类属性可能不会提供太多有价值的信息。

Ref：https://blog.csdn.net/u010089444/article/details/53241218

2.2 C4.5 原理及应用场景

C4.5算法是ID3算法的改进，区别有：

    1. C4.5算法中使用信息增益比率来作为选择分支的准则。信息增益比率通过引入一个被称作分裂信息的项来惩罚取值较多的特征，克服了用信息增益选择属性时偏向选择取值多的属性的不足。
    2. 在树构造过程中进行剪枝。合并相邻的无法产生大量信息增益的叶节点，消除过渡匹配问题。
    3. C4.5算法既能处理标称型数据，又能连续型数据。能够处理具有缺失属性值的训练数据。

缺点：
    
    1. 算法低效，在构造树的过程中，需要对数据集进行多次的顺序扫描和排序，因而导致算法的低效
    2. 内存受限，只适合于能够驻留于内存的数据集，当训练集大得无法在内存容纳时程序无法运行。

2.3 CART分类树 原理及应用场景

CART（Classification And Regression Tree）算法既可以用于创建分类树，也可以用于创建回归树。CART算法的重要特点包含以下三个方面：

    1. 二分(Binary Split)：在每次判断过程中，都是对样本数据进行二分。CART算法是一种二分递归分割技术，把当前样本划分为两个子样本，使得生成的每个非叶子结点都有两个分支，因此CART算法生成的决策树是结构简洁的二叉树。由于CART算法构成的是一个二叉树，它在每一步的决策时只能是“是”或者“否”，即使一个feature有多个取值，也是把数据分为两部分
    2. 单变量分割(Split Based on One Variable)：每次最优划分都是针对单个变量。
    3. 剪枝策略：CART算法的关键点，也是整个Tree-Based算法的关键步骤。剪枝过程特别重要，所以在最优决策树生成过程中占有重要地位。有研究表明，剪枝过程的重要性要比树生成过程更为重要，对于不同的划分标准生成的最大树(Maximum Tree)，在剪枝之后都能够保留最重要的属性划分，差别不大。反而是剪枝方法对于最优树的生成更为关键。

Ref：https://blog.csdn.net/u010089444/article/details/53241218

3. 回归树原理


4. 决策树防止过拟合手段

产生过度拟合问题的原因：

    样本问题
    构建决策树的方法问题
    
如何解决过度拟合数据问题的发生：

    合理、有效地抽样，用相对能够反映业务逻辑的训练集去产生决策树
    剪枝：提前停止树的增长或者对已经生成的树按照一定的规则进行后剪枝。
    
剪枝是一个简化过拟合决策树的过程。有两种常用的剪枝方法：

    先剪枝（prepruning）：通过提前停止树的构建而对树“剪枝”，一旦停止，节点就成为树叶。该树叶可以持有子集元组中最频繁的类
    后剪枝（postpruning）：它首先构造完整的决策树，允许树过度拟合训练数据，然后对那些置信度不够的结点子树用叶子结点来代替，该叶子的类标号用该结点子树中最频繁的类标记。后剪枝的剪枝过程是删除一些子树，然后用其叶子节点代替，这个叶子节点所标识的类别通过大多数原则(majority class criterion)确定。

5. 模型评估


6. sklearn参数详解，Python绘制决策树

6.1 sklearn参数详解

DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

参数含义：

criterion:string类型，可选（默认为"gini"）

    衡量分类的质量。支持的标准有"gini"代表的是Gini impurity(不纯度)与"entropy"代表的是information gain（信息增益）。

splitter:string类型，可选（默认为"best"）

    一种用来在节点中选择分类的策略。支持的策略有"best"，选择最好的分类，"random"选择最好的随机分类。

max_features:int,float,string or None 可选（默认为None）

在进行分类时需要考虑的特征数。

    1.如果是int，在每次分类是都要考虑max_features个特征。
    2.如果是float,那么max_features是一个百分率并且分类时需要考虑的特征数是int(max_features*n_features,其中n_features是训练完成时发特征数)。
    3.如果是auto,max_features=sqrt(n_features)
    4.如果是sqrt,max_features=sqrt(n_features)
    5.如果是log2,max_features=log2(n_features)
    6.如果是None，max_features=n_features
    
注意：至少找到一个样本点有效的被分类时，搜索分类才会停止。

max_depth:int or None,可选（默认为"None"）

    表示树的最大深度。如果是"None",则节点会一直扩展直到所有的叶子都是纯的或者所有的叶子节点都包含少于min_samples_split个样本点。忽视max_leaf_nodes是不是为None。

min_samples_split:int,float,可选（默认为2）

    区分一个内部节点需要的最少的样本数。
    1.如果是int，将其最为最小的样本数。
    2.如果是float，min_samples_split是一个百分率并且ceil(min_samples_split*n_samples)是每个分类需要的样本数。ceil是取大于或等于指定表达式的最小整数。

min_samples_leaf:int,float,可选（默认为1）

    一个叶节点所需要的最小样本数：
    1.如果是int，则其为最小样本数
    2.如果是float，则它是一个百分率并且ceil(min_samples_leaf*n_samples)是每个节点所需的样本数。

min_weight_fraction_leaf:float,可选（默认为0）

    一个叶节点的输入样本所需要的最小的加权分数。

max_leaf_nodes:int,None 可选（默认为None）

    在最优方法中使用max_leaf_nodes构建一个树。最好的节点是在杂质相对减少。如果是None则对叶节点的数目没有限制。如果不是None则不考虑max_depth.

class_weight:dict,list of dicts,“Banlanced” or None,可选（默认为None）

    表示在表{class_label:weight}中的类的关联权值。如果没有指定，所有类的权值都为1。对于多输出问题，一列字典的顺序可以与一列y的次序相同。
    "balanced"模型使用y的值去自动适应权值，并且是以输入数据中类的频率的反比例。如：n_samples/(n_classes*np.bincount(y))。
    对于多输出，每列y的权值都会想乘。
    如果sample_weight已经指定了，这些权值将于samples以合适的方法相乘。

random_state:int,RandomState instance or None

    如果是int,random_state 是随机数字发生器的种子；如果是RandomState，random_state是随机数字发生器，如果是None，随机数字发生器是np.random使用的RandomState instance.

persort:bool,可选（默认为False）

    是否预分类数据以加速训练时最好分类的查找。在有大数据集的决策树中，如果设为true可能会减慢训练的过程。当使用一个小数据集或者一个深度受限的决策树中，可以减速训练的过程。

属性：

feature_importances_ : array of shape = [n_features]
    
    特征重要性。该值越高，该特征越重要。
 
    特征的重要性为该特征导致的评价准则的（标准化的）总减少量。它也被称为基尼的重要性
    
max_feature_:int

    max_features推断值。
    
n_features_：int

    执行fit的时候，特征的数量。

n_outputs_ : int

    执行fit的时候，输出的数量。

tree_ : 底层的Tree对象

Ref：官方文档：http://scikit-learn.org/stable/modules/tree.html
