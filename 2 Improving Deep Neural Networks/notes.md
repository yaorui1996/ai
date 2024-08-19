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



# Regularization

$J(\omega, b)=\frac{1}{m} \sum_{i=1}^m L\left(\hat{y}^{(i)}, y^{(i)}\right)+\frac{\lambda}{2 m}\|\omega\|_2^2$

$J\left(\omega^{[1]}, b^{[1]}, \dots, \omega^{[L]}, b^{[L]}\right)=\frac{1}{m} \sum_{i=1}^m L\left(\hat{y}^{(i)}, y^{(i)}\right)+\frac{\lambda}{2 m} \sum_{l=1}^L\left\|\omega^{[l]}\right\|_{F}^2$

- Frobenius Norm: $\left\|\omega\right\|_F^2=\rm{Tr}(\omega^T\omega)$

$\begin{aligned} & d \omega^{[l]}=(\text {from backprop})+\frac{\lambda}{m} \omega^{[l]} \\ & \omega^{[l]}:=\omega^{[l]}-\alpha \cdot d \omega^{[l]}=(1-\frac{\alpha\lambda}{m})\omega^{[l]}-\alpha(\text {from backprop})\end{aligned}$
