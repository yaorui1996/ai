# Logistic Regression

- **定义**：一种统计方法，用于分析一个或多个独立变量与一个二分类因变量之间的关系。

- **用途**：常用于分类问题，如信用评分、疾病预测等。

- **公式**：

  $\hat{y}=\sigma\left(w^T x+b\right), \text { where } \sigma(z)=\frac{1}{1+e^{-z}}$
  
  Loss Function: $\mathcal{L}(\hat{y}, y)=-(y \log \hat{y}+(1-y) \log (1-\hat{y}))$
  
  Cost Function: $J(\omega, b)=\frac{1}{m} \sum_{i=1}^m \mathcal{L}\left(\hat{y}^{(i)}, y^{(i)}\right)$

- Loss Function 选择依据：$\frac{d L}{d z}=\hat{y}-y$

- Python Broadcast
  - (m,n) [+-\*/] (m,1) → (m,n) [+-\*/] (m,n)
  - (m,n) [+-\*/] (1,n) → (m,n) [+-\*/] (m,n)



# Neural Network

- 表示法，2 Layer NN
  - Input Layer (0) : $a^{[0]}=x,\ dim=n^{[0]}$
  - Hidden Layer (1) : $a^{[1]},\ dim=n^{[1]}$
  - Output Layer (2) : $\hat{y}=a^{[2]},\ dim=n^{[2]}=1$
- Activation Function: $g(z)$

    | Name           | Function                        | Derivative                          |
    | -------------- | ------------------------------- | ----------------------------------- |
    | $\sigma(z)$    | $\frac{1}{1+e^{-z}}$            | $\sigma(z)(1-\sigma(z))$            |
    | $\rm{tanh}(z)$ | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $1-\rm{tanh}^2(z)$                  |
    | ReLU           | $\rm{max}(0,z)$                 | $\frac{1}{2}(1+\rm{sgn}(z))$        |
    | Leaky ReLU     | $\rm{max}(0.01z,z)$             | $\frac{1}{2}(1.01+0.99\rm{sgn}(z))$ |

- Gradient Descent

  | Parameters          | Dimensions          |
  | ------------------- | ------------------- |
  | $\omega^{[1]}$      | $(n^{[1]},n^{[0]})$ |
  | $b^{[1]}$           | $(n^{[1]},1)$       |
  | $\omega^{[2]}$      | $(n^{[2]},n^{[1]})$ |
  | $b^{[2]}$           | $(n^{[2]},1)$       |
  | $X=A^{[0]}$         | $(n^{[0]},m)$       |
  | $Z^{[1]},A^{[1]}$   | $(n^{[1]},m)$       |
  | $Z^{[2]},Y=A^{[2]}$ | $(n^{[2]},m)$       |

  - Cost Function

    $\rm{J}(\omega^{[1]},b^{[1]},\omega^{[2]},b^{[2]})=\frac{1}{m}\sum{L(\hat{y}=a^{[2]},y)}$

  - Forward Propagation

    $\begin{aligned} & Z^{[1]}=W^{[1]} A^{[0]}+b^{[1]} \\ & A^{[1]}=g^{[1]}\left(Z^{[1]}\right) \\ & Z^{[2]}=W^{[2]} A^{[1]}+b^{[2]} \\ & A^{[2]}=g^{[2]}\left(Z^{[2]}\right)\end{aligned}$

  - Backward Propagation

    $\begin{aligned} & dZ^{[2]}=A^{[2]}-Y \\ & dW^{[2]}=\frac{1}{m} dZ^{[2]} A^{[1]T} \\ & db^{[2]}=\frac{1}{m} \rm{np.sum}(dZ^{[2]},axis=1,keepdims=True) \\ & dZ^{[1]}=W^{[2]T} dZ^{[2]} *_{element} g^{[1]\prime}\left(Z^{[1]}\right) \\ & dW^{[1]}=\frac{1}{m} dZ^{[1]} A^{[0]T} \\ & db^{[1]}=\frac{1}{m} \rm{np.sum}(dZ^{[1]},axis=1,keepdims=True)\end{aligned}$

- Random Initialization
  - 消除隐藏层节点对称性



# Deep Neural Network

- Notation

  | Notation                   | Meaning                |
  | -------------------------- | ---------------------- |
  | $L$                        | # layers               |
  | $n^{[l]}$                  | # units in layer l     |
  | $a^{[l]}=g^{[l]}(z^{[l]})$ | activations in layer l |

- Forward Propagation

  $\begin{aligned} & Z^{[l]}=W^{[l]} A^{[l-1]}+b^{[l]} \\ & A^{[l]}=g^{[l]}\left(Z^{[l]}\right)\end{aligned}$

  | Parameters        | Dimensions            |
  | ----------------- | --------------------- |
  | $W^{[l]}$         | $(n^{[l]},n^{[l-1]})$ |
  | $b^{[l]}$         | $(n^{[l]},1)$         |
  | $Z^{[l]},A^{[l]}$ | $(n^{[l]},m)$         |

- Backward Propagation

  $\begin{aligned} & dZ^{[l]}=dA^{[l]} *_{element} g^{[l]\prime}\left(Z^{[l]}\right) \\ & dW^{[l]}=\frac{1}{m} dZ^{[l]} A^{[l-1]T} \\ & db^{[l]}=\frac{1}{m} \rm{np.sum}(dZ^{[l]},axis=1,keepdims=True) \\ & dA^{[l-1]}=W^{[l]T} dZ^{[l]}\end{aligned}$

- Parameters: $W^{[l]}, b^{[l]}$

- Hyperparameters

  - learning rate, $\alpha$
  - \# iterations

  - \# hidden layers, $L$
  - \# hidden units, $n^{[l]}$
  - choice of activation function

- Build

  ```python
  def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
      parameters = initialize_parameters_deep(layers_dims)
      for i in range(0, num_iterations):
          AL, caches = L_model_forward(X, parameters)
          cost = compute_cost(AL, Y)
          grads = L_model_backward(AL, Y, caches)
          parameters = update_parameters(parameters, grads, learning_rate)
  	return parameters, costs
  ```
