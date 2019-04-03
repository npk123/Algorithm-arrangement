1. 信息论基础（熵 联合熵 条件熵 信息增益 基尼不纯度）

1.1 熵

在信息论与概率统计中，熵表示随机变量不确定性的度量。设X是一个取有限个值得离散随机变量，其概率分布为
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture.JPG)

则随机变量X的熵定义为
  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture1.JPG)

若pi等于0，定义0log0=0，熵的单位为比特或者纳特。

1.2 联合熵

1.3 条件熵

H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性定义为X给定条件下Y的条件概率分布的熵对X的数学期望

  ![equation1](https://github.com/npk123/Algorithm-datawhale/blob/master/images/Capture2.JPG)

经验熵和经验条件熵：当熵和条件熵中的概率由数据估计（特别是极大似然估计）得到时，所对应的熵与条件熵分别称为经验熵和条件经验熵。

1.4 信息增益

1.5 基尼不纯度

2.决策树的不同分类算法（ID3算法、C4.5、CART分类树）的原理及应用场景

2.1 ID3算法 原理及应用场景

2.2 C4.5 原理及应用场景

2.3 CART分类树 原理及应用场景

3. 回归树原理


4. 决策树防止过拟合手段


5. 模型评估


6. sklearn参数详解，Python绘制决策树
6.1 
