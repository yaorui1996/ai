# Train/Dev/Test Sets

- Previous: 70/30, 60/20/20
- Big Data: 98/1/1
- Mismatched Train/Test Destribution
- Make sure Dev and Test come from same distribution



# Bias and Variance

1. high bias (train sets performance)
   - Bigger network
   - Train longer
   - NN architecture search
2. high variance (dev sets performance)
   - More data
   - Regularization
   - NN architecture search
3. Done



# L2 Regularization

$J(\omega, b)=\frac{1}{m} \sum_{i=1}^m L\left(\hat{y}^{(i)}, y^{(i)}\right)+\frac{\lambda}{2 m}\|\omega\|_2^2$

$J\left(\omega^{[1]}, b^{[1]}, \dots, \omega^{[L]}, b^{[L]}\right)=\frac{1}{m} \sum_{i=1}^m L\left(\hat{y}^{(i)}, y^{(i)}\right)+\frac{\lambda}{2 m} \sum_{l=1}^L\left\|\omega^{[l]}\right\|_{F}^2$

- Frobenius Norm: $\left\|\omega\right\|_F^2=\rm{Tr}(\omega^T\omega)$

$\begin{aligned} & d \omega^{[l]}=(\text {from backprop})+\frac{\lambda}{m} \omega^{[l]} \\ & \omega^{[l]}:=\omega^{[l]}-\alpha \cdot d \omega^{[l]}=(1-\frac{\alpha\lambda}{m})\omega^{[l]}-\alpha(\text {from backprop})\end{aligned}$



# Dropout Regularization

- Illustrate with layer 3, keep-prob = 0.8
- D3 = np.random.rand(A3.shape[0], A3.shape[1]) < keep-prob
- A3 *= D3
- A3 /= keep-prob



# Normalizing Training Sets

$x -= \mu,\  x /= \sigma$



# Weight Initialization

$z=\omega_1 x_1+\omega_2 x_2+...+\omega_n x_n+b$

$Var(\omega)=\frac{1}{n} \ or\  \frac{2}{n}$

$\omega^{[l]}=np.random.rand(shape)*np.sqrt(\frac{2}{n^{[l-1]}})$

- ReLU: $\sqrt{\frac{2}{n^{[l-1]}}}$

- tanh: $\sqrt{\frac{1}{n^{[l-1]}}}$

- Xavier initialization:  $\sqrt{\frac{2}{n^{[l-1]}+n^{[l]}}}$



# Numerical Approximation of Gradients

$\frac{f(\theta+\epsilon)-f(\theta+\epsilon)}{2\epsilon} \approx g(\theta)$



# Gradient Checking

Check $\frac{\left\|d \theta_{\text {approx }}-d \theta\right\|_2}{\left\|d \theta_{\text {approx }}\right\|_2+\|d \theta\|_2}$: $10^{-7}\ great,\ 10^{-5},\ 10^{-3}\ worry$

- Don't use in training
- If algorithm fails grad check, look at components and try to identify bug
- Remember regularization
- Doesn't work with dropout
- Run at random initialization perhaps again after some training

