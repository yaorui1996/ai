# Logistic Regression

- **定义**：一种统计方法，用于分析一个或多个独立变量与一个二分类因变量之间的关系。

- **用途**：常用于分类问题，如信用评分、疾病预测等。

- **公式**：

  $\hat{y}=\sigma\left(w^T x+b\right), \text { where } \sigma(z)=\frac{1}{1+e^{-z}}$
  
  Loss Function：$\mathcal{L}(\hat{y}, y)=-(y \log \hat{y}+(1-y) \log (1-\hat{y}))$
  
  Cost Function：$J(\omega, b)=\frac{1}{m} \sum_{i=1}^m \mathcal{L}\left(\hat{y}^{(i)}, y^{(i)}\right)$

- Loss Function 选择依据：$\frac{d L}{d z}=\hat{y}-y$

- Python Broadcast
  - (m,n) [+-\*/] (m,1) → (m,n) [+-\*/] (m,n)
  - (m,n) [+-\*/] (1,n) → (m,n) [+-\*/] (m,n)



# Neural Network

- 表示法，2 Layer NN
  - Input Layer (0) : $a^{[0]}=x$
  - Hidden Layer (1) : $a^{[1]}$
  - Output Layer (2) : $\hat{y}=a^{[2]}$

