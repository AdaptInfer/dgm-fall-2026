---
layout: distill
title: "Lecture Review: Midterm"
description: "A condensed, stand-alone review of the first half of the course: the core idea of each lecture, the formulas with the shape of every object, and the mistakes that are easiest to make."
date: 2026-10-13

authors:
  - name: Oran Jiang
    url: "#"

editors:
  - name: None
    url: "#"

abstract: >
  The midterm covers Lectures 1 to 11. These notes condense them into one reference: what machine learning is, the history and the math we build on, single-layer networks and gradient descent, autograd, multilayer perceptrons and backpropagation, CNNs, regularization, normalization and initialization, and optimizers.
---

---

No new material was taught for the midterm session, so instead of scribing one lecture, these notes collect the essentials of Lectures 1 to 11, which the midterm covers, drawing on the slides and the readings <d-cite key="goodfellow2016deep"></d-cite>.

## How to Read These Notes

Each section gives the main idea of one lecture, the formulas with the shape of every object, and the easiest mistakes to make.

**Notation.** As on the slides, <d-math>n</d-math> is the number of examples and <d-math>m</d-math> the number of features, so the design matrix <d-math>\mathbf{X}\in\mathbb{R}^{n\times m}</d-math> has one example per row. A superscript <d-math>[i]</d-math> indexes examples, not powers. In Lecture 7, <d-math>n_l</d-math> is the width of layer <d-math>l</d-math>. "L7 s.22" means slide 22 of Lecture 7.

## Lecture 1: What Is Machine Learning?

In traditional programming a person writes the program; in machine learning the computer gets inputs and outputs and produces the program, a function with learned parameters. In Mitchell's definition, a program **learns** from experience <d-math>\mathcal{E}</d-math> on task <d-math>\mathcal{T}</d-math> if its performance <d-math>\mathcal{P}</d-math> on <d-math>\mathcal{T}</d-math> improves with <d-math>\mathcal{E}</d-math>, where <d-math>\mathcal{P}</d-math> is measured on unseen data. On the slides, ML and AI only partly overlap (Deep Blue's search is AI without ML), while deep learning lies inside ML: many-layered networks that learn their own representations.

**Three fundamental questions.** For <d-math>n</d-math> binary variables, *representation* by a full joint distribution needs <d-math>2^n</d-math> states and <d-math>2^n-1</d-math> free parameters (256 states for <d-math>n=8</d-math>). *Inference* such as <d-math>P(X_8\mid X_1)=P(X_1,X_8)/P(X_1)</d-math> needs a sum over the <d-math>2^6</d-math> settings of <d-math>X_2,\ldots,X_7</d-math> (marginalization). *Learning* picks <d-math>M=\mathrm{argmax}_{M\in\mathcal{H}}F(D;M)</d-math>, the model in the hypothesis space with the best score; the argmax returns the model, not the score. A graphical model factorizes the joint as <d-math>\prod_i P(X_i\mid \mathrm{Pa}(X_i))</d-math>, which needs <d-math>\sum_i 2^{\lvert \mathrm{Pa}(X_i)\rvert}</d-math> parameters: 18 instead of 255 in the lecture's example.

**Kinds of learning.** *Supervised*: labeled pairs, for regression (<d-math>h:\mathbb{R}^m\to\mathbb{R}</d-math>) or classification (<d-math>h:\mathbb{R}^m\to\{1,\ldots,k\}</d-math>). *Unsupervised*: no labels; find structure (clustering, PCA, autoencoders). *Reinforcement*: learn a policy <d-math>\pi:S\to A</d-math> from rewards that only say how good an action was, often with a delay, so a reward is not a label.

*Semi-supervised*: few labels, many unlabeled examples. *Self-supervised*: labels built from the data itself, such as the next word. A batch of images has shape <d-math>(N,C,H,W)</d-math>, such as <d-math>[128,1,28,28]</d-math> for 128 MNIST digits.

## Lecture 2: A Brief History of Deep Learning

<div class="l-page" markdown="1">

| Year | Milestone | Why it matters |
| ---- | --------- | -------------- |
| 1943 | McCulloch–Pitts neuron | Threshold unit with fixed <d-math>\pm 1</d-math> weights: AND, OR, NOT, but no learning and no XOR |
| 1957 | Rosenblatt's perceptron | First model that learns its weights from data |
| 1969 | Minsky and Papert | One layer cannot represent XOR; the first "winter" follows |
| 1986 | Backpropagation | Rumelhart, Hinton and Williams: efficient gradients for multilayer networks |
| 1998 | LeNet-5 | CNN for digit recognition <d-cite key="lecun1998gradient"></d-cite> |
| 2012 | AlexNet | ImageNet top-5 error 15.4% vs. 26.2% for the runner-up <d-cite key="krizhevsky2012imagenet"></d-cite> |

</div>

**Why one perceptron cannot do XOR.** We need <d-math>b\lt 0</d-math> for (0,0), <d-math>w_1+b\ge 0</d-math> and <d-math>w_2+b\ge 0</d-math> for (1,0) and (0,1), and <d-math>w_1+w_2+b\lt 0</d-math> for (1,1). Adding the middle two gives <d-math>w_1+w_2+b\ge -b\gt 0</d-math>, a contradiction. A monotone activation such as the sigmoid does not help, since the boundary is still a line, but a hidden layer does: <d-math>A\oplus B=\bar A B + A\bar B</d-math> is a two-layer circuit.

Two facts recur all semester. Without a nonlinearity, layers collapse, <d-math>W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2=(W_2W_1)\mathbf{x}+(W_2\mathbf{b}_1+\mathbf{b}_2)</d-math>. And backpropagation only *computes* gradients, while gradient descent *updates* the parameters. Deep learning is representation learning with many levels (pixels, edges, parts, objects); it took off after 2012 thanks to large datasets and GPUs.

## Lecture 3: Linear Algebra, Probability, and Estimation

**Tensors.** The rank of a tensor is its number of indices (`.ndim`); PyTorch defaults to `float32`, NumPy to `float64`. A neuron computes <d-math>z=\mathbf{w}^\top\mathbf{x}+b</d-math> with <d-math>\mathbf{w},\mathbf{x}\in\mathbb{R}^{m}</d-math>, and a batch computes <d-math>\mathbf{z}=\mathbf{X}\mathbf{w}+\mathbf{1}_n b\in\mathbb{R}^{n}</d-math>, written <d-math>\mathbf{X}\mathbf{w}+b</d-math> thanks to broadcasting. Broadcasting fails silently: shape `(5,)` minus shape `(5,1)` gives `(5,5)`. In PyTorch `*` is elementwise and `@` is matrix multiplication.

<div class="l-page" markdown="1">

| Fact | Formula | Remember |
| ---- | ------- | -------- |
| Bernoulli | <d-math>P(x)=\theta^x(1-\theta)^{1-x}</d-math>, mean <d-math>\theta</d-math>, variance <d-math>\theta(1-\theta)</d-math> | <d-math>x\in\{0,1\}</d-math> |
| Gaussian | <d-math>f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)</d-math> | Density can exceed 1; L3 s.28 drops the minus sign |
| Variance | <d-math>\mathrm{Var}(X)=E[X^2]-E[X]^2</d-math>, <d-math>\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)</d-math> | <d-math>E[g(X)]\ne g(E[X])</d-math> in general |
| Linearity | <d-math>E[X_1+X_2]=E[X_1]+E[X_2]</d-math> | Holds without independence; variances add only if the covariance is 0 |
| Covariance | <d-math>\mathrm{Cov}(X,Y)=E[XY]-E[X]E[Y]</d-math> | Independence gives zero covariance, not the reverse: <d-math>X\sim U(-1,1)</d-math>, <d-math>Y=X^2</d-math> |
| Bayes | <d-math>P(A\mid B)=P(B\mid A)P(A)/P(B)</d-math> | Posterior is proportional to likelihood times prior |

</div>

The prior matters: with 1% prevalence, a test that detects 99% of cases with a 5% false positive rate gives <d-math>P(\text{disease}\mid +)=\frac{0.99\cdot 0.01}{0.99\cdot 0.01+0.05\cdot 0.99}\approx 0.17</d-math>.

**Estimation.** The MLE is <d-math>\hat\theta=\mathrm{argmax}_\theta P(D\mid\theta)</d-math>. For <d-math>k</d-math> ones in <d-math>n</d-math> Bernoulli draws, <d-math>\ell(\theta)=k\log\theta+(n-k)\log(1-\theta)</d-math>, and <d-math>\ell'(\theta)=\frac{k}{\theta}-\frac{n-k}{1-\theta}=0</d-math> gives <d-math>\hat\theta=k/n</d-math> (0.6 for the data 1, 0, 1, 1, 0). The log does not move the argmax; it turns products into sums and avoids underflow.

The MAP estimate is <d-math>\mathrm{argmax}_\theta P(D\mid\theta)P(\theta)</d-math>: a uniform prior gives the MLE, and a prior <d-math>P(\theta)\propto e^{-\lambda R(\theta)}</d-math> gives "log-likelihood minus a penalty", so **regularization is MAP**.

**Linear regression.** With <d-math>\mathbf{y}=\mathbf{X}\beta+\epsilon</d-math> and <d-math>\epsilon_i\sim\mathcal{N}(0,\sigma^2)</d-math>, the MLE is least squares, <d-math>\hat{\beta}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}</d-math>. Ridge (L2) is MAP with a Gaussian prior and LASSO (L1) is MAP with a Laplace prior.

## Lecture 4: Single-Layer Networks and the Perceptron

A unit computes the net input <d-math>z=\mathbf{w}^\top\mathbf{x}+b</d-math>, the activation <d-math>a=\sigma(z)</d-math>, and the label <d-math>\hat y=f(a)</d-math>. A threshold <d-math>\theta</d-math> is a bias <d-math>b=-\theta</d-math>, which can be folded into the weights with a constant input <d-math>x_0=1</d-math>. The **perceptron rule**, with <d-math>\hat y=\mathbb{1}[\mathbf{w}^\top\mathbf{x}\gt 0]</d-math> and the bias inside <d-math>\mathbf{w}</d-math>, visits one example at a time:

<d-math block>
\text{err}=y^{[i]}-\hat y^{[i]}\in\{-1,0,1\},\qquad \mathbf{w}:=\mathbf{w}+\text{err}\cdot\mathbf{x}^{[i]}.
</d-math>

A missed positive adds <d-math>\mathbf{x}</d-math> and a false positive subtracts it. Convergence is guaranteed only for linearly separable data; on XOR the rule cycles forever, and on separable data it stops at the first separating line, which may generalize poorly.

**Geometry.** Since <d-math>\mathbf{w}^\top\mathbf{x}=\lVert\mathbf{w}\rVert\lVert\mathbf{x}\rVert\cos\theta</d-math>, the prediction is 1 when the angle is below 90°. So <d-math>\mathbf{w}</d-math> is perpendicular to the decision boundary and points to the positive side, the bias shifts the boundary, and scaling <d-math>\mathbf{w}</d-math> by <d-math>c\gt 0</d-math> changes no prediction.

**The perceptron is not an MLE**: a hard threshold defines no likelihood. With a sigmoid read as <d-math>P(y=1\mid\mathbf{x})</d-math> we get logistic regression, whose log-likelihood gradient <d-math>\sum_i(y_i-\sigma(\mathbf{w}^\top\mathbf{x}_i))\mathbf{x}_i</d-math> has the perceptron's form but no closed-form root.

**A fully-connected layer** with <d-math>h</d-math> outputs is <d-math>\mathbf{a}=\sigma(\mathbf{W}\mathbf{x}+\mathbf{b})\in\mathbb{R}^{h}</d-math>, <d-math>\mathbf{W}\in\mathbb{R}^{h\times m}</d-math>, where <d-math>w_{i,j}</d-math> connects input <d-math>j</d-math> to output <d-math>i</d-math>. With examples in rows, <d-math>\mathbf{A}=\sigma(\mathbf{X}\mathbf{W}^\top+\mathbf{b}^\top)\in\mathbb{R}^{n\times h}</d-math>, using <d-math>(\mathbf{A}\mathbf{B})^\top=\mathbf{B}^\top\mathbf{A}^\top</d-math>. This is `nn.Linear`: `weight` has shape `(out_features, in_features)`, `bias` has shape `(out_features,)`, and it returns `X @ W.T + b`.

## Lecture 5: Parameter Optimization and Gradient Descent

A step function has zero gradient almost everywhere, so it cannot guide learning. The sigmoid <d-math>\sigma(z)=1/(1+e^{-z})</d-math> is smooth, with <d-math>\sigma'(z)=\sigma(z)(1-\sigma(z))\le 0.25</d-math> (maximum at <d-math>z=0</d-math>). Reading <d-math>a=\sigma(z)</d-math> as a Bernoulli probability, the negative log-likelihood of one example is the **binary cross-entropy** <d-math>\mathcal{L}=-[y\log a+(1-y)\log(1-a)]</d-math>, and the chain rule gives

<d-math block>
\frac{\partial\mathcal{L}}{\partial w_j}=\underbrace{\frac{a-y}{a(1-a)}}_{\partial\mathcal{L}/\partial a}\cdot\underbrace{a(1-a)}_{da/dz}\cdot\underbrace{x_j}_{\partial z/\partial w_j}=(a-y)\,x_j.
</d-math>

Likewise <d-math>\partial\mathcal{L}/\partial b=a-y</d-math>: the step is proportional to the error. With squared error the factor <d-math>a(1-a)</d-math> survives, so a saturated unit that is confidently wrong barely learns. Example: <d-math>\mathbf{x}=(1,2)</d-math>, <d-math>\mathbf{w}=\mathbf{0}</d-math>, <d-math>b=0</d-math>, <d-math>y=1</d-math> give <d-math>a=0.5</d-math> and gradient <d-math>(-0.5,-1)</d-math>; one step with <d-math>\eta=0.1</d-math> gives <d-math>\mathbf{w}=(0.05,0.1)</d-math>.

**Gradient descent** moves against the gradient, <d-math>\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}</d-math>. A small <d-math>\eta</d-math> is slow; a large one oscillates or diverges (on <d-math>f(x)=x^2</d-math>, <d-math>x\leftarrow(1-2\eta)x</d-math> diverges for <d-math>\eta\gt 1</d-math>). A minibatch gradient is an **unbiased** but noisy estimate of the full-batch gradient. Critical points are minima, maxima or saddle points; logistic regression is convex, deep networks are not. The softmax <d-math>e^{z_k}/\sum_j e^{z_j}</d-math> is computed on <d-math>\mathbf{z}-\max_j z_j</d-math> to avoid overflow.

**Chain rule on a computation graph.** To get <d-math>\partial l/\partial w</d-math>, multiply local derivatives along each path from <d-math>w</d-math> to the loss, then **add over all paths**; several paths appear when a value fans out (a fully-connected layer) or a weight is shared. For ReLU the local derivative is <d-math>\mathbb{1}[v\gt 0]</d-math>, so no gradient passes a unit with negative input. Backpropagation applies this rule in reverse order, caching intermediate results.

## Lecture 6: Automatic Differentiation with PyTorch

*Numerical* differentiation, <d-math>(f(\mathbf{x}+h\mathbf{e}_i)-f(\mathbf{x}))/h</d-math>, has <d-math>O(h)</d-math> error (<d-math>O(h^2)</d-math> for central differences) and needs <d-math>O(n)</d-math> evaluations, so it is only for gradient checks. *Symbolic* differentiation builds expressions that blow up. *Automatic* differentiation applies exact local rules along the recorded computation <d-cite key="baydin2018automatic"></d-cite>. Forward mode gives a Jacobian-vector product <d-math>J\mathbf{r}</d-math> per pass; reverse mode gives <d-math>J^\top\mathbf{r}</d-math>, so the whole gradient of a scalar loss takes one backward pass. Backpropagation is reverse-mode AD.

PyTorch builds the graph while the forward code runs: tensors with `requires_grad=True` are tracked, each result stores a `grad_fn`, and `loss.backward()` **adds** gradients into `.grad`. Define an `nn.Module` (layers in `__init__`, computation in `forward`), create the model and optimizer, then train and evaluate:

<d-code block language="python">
import torch
import torch.nn.functional as F
from torch import optim
opt = optim.SGD(model.parameters(), lr=0.1)
for x, y in train_loader:
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()   # .grad accumulates
    loss.backward()   # reverse-mode AD
    opt.step()        # w := w - lr * grad
model.eval()          # dropout/BN: eval mode
with torch.no_grad():  # build no graph
    pred = model(x_val).argmax(dim=1)
</d-code>

Losses take **logits**: `F.cross_entropy` applies log-softmax itself, and `BCEWithLogitsLoss` is the stable form of sigmoid plus BCE. The output gradients are <d-math>\hat y-y</d-math> for sigmoid with BCE and <d-math>\mathrm{softmax}(\mathbf{z})_k-\mathbb{1}[k=c]</d-math> for softmax with cross-entropy. `model.eval()` changes dropout and batch norm but still builds graphs; `torch.no_grad()` stops that.

## Lecture 7: Multilayer Perceptrons and Backpropagation

**From two classes to K.** A categorical model uses the softmax <d-math>\hat y_k=e^{z_k}/\sum_j e^{z_j}</d-math>. With a one-hot label, the cross-entropy is <d-math>L=-\sum_k y_k\log\hat y_k=-\log\hat y_c</d-math>, and again <d-math>\partial L/\partial z_k=\hat y_k-y_k</d-math>. The sigmoid is the two-class softmax: <d-math>\mathrm{softmax}(z_1,z_2)_1=\sigma(z_1-z_2)</d-math>. Do not apply a softmax before `nn.CrossEntropyLoss`.

**The MLP.** Layer <d-math>l</d-math> computes <d-math>\mathbf{z}^{(l)}=W^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}</d-math>, <d-math>\mathbf{a}^{(l)}=\sigma(\mathbf{z}^{(l)})</d-math>, with <d-math>W^{(l)}\in\mathbb{R}^{n_l\times n_{l-1}}</d-math> and <d-math>\sum_l(n_{l-1}+1)\,n_l</d-math> parameters in total (784-100-10: <d-math>785\cdot 100+101\cdot 10=79{,}510</d-math>). With nonlinear activations, the hidden layer maps the inputs to a space where the classes are linearly separable, which solves XOR.

<div class="l-page" markdown="1">

| Activation | Range | Derivative | Notes |
| ---------- | ----- | ---------- | ----- |
| Sigmoid | <d-math>(0,1)</d-math> | <d-math>\sigma(1-\sigma)\le 0.25</d-math> | Saturates; not zero-centered |
| Tanh | <d-math>(-1,1)</d-math> | <d-math>1-\tanh^2\le 1</d-math> | Zero-centered; still saturates |
| ReLU | <d-math>[0,\infty)</d-math> | <d-math>0</d-math> or <d-math>1</d-math> | Cheap; units can "die" |
| Leaky ReLU, PReLU | <d-math>\mathbb{R}</d-math> | <d-math>\alpha</d-math> or <d-math>1</d-math> | Leaky ReLU fixes <d-math>\alpha</d-math>; PReLU learns it |

</div>

The MLP loss is not convex, so results depend on the initialization and the seed. All-zero (or all-equal) weights give every unit in a layer the same gradient, so the units stay identical; random initialization breaks the symmetry.

**Backpropagation in matrix form**, for one example, with <d-math>\delta^{(l)}=\partial L/\partial\mathbf{z}^{(l)}\in\mathbb{R}^{n_l}</d-math>:

<d-math block>
\begin{aligned}
\delta^{(L)}&=\hat{\mathbf{y}}-\mathbf{y},\qquad \delta^{(l)}=\big(W^{(l+1)}\big)^\top\delta^{(l+1)}\odot\sigma'\big(\mathbf{z}^{(l)}\big),\\
\frac{\partial L}{\partial W^{(l)}}&=\delta^{(l)}\big(\mathbf{a}^{(l-1)}\big)^\top,\qquad \frac{\partial L}{\partial\mathbf{b}^{(l)}}=\delta^{(l)}.
\end{aligned}
</d-math>

The error flows back through <d-math>W^\top</d-math>, not <d-math>W</d-math> (an <d-math>n_l\times n_{l+1}</d-math> matrix maps an <d-math>n_{l+1}</d-math>-vector to an <d-math>n_l</d-math>-vector), and the weight gradient is an outer product shaped like <d-math>W^{(l)}</d-math>. In the lecture's worked example, a ReLU unit received an upstream gradient of 0.755 but had <d-math>z\le 0</d-math>, so its gradient was 0. Unrolled, <d-math>\delta^{(1)}=D^{(1)}W^{(2)\top}\cdots D^{(L-1)}W^{(L)\top}\delta^{(L)}</d-math> with <d-math>D^{(l)}=\mathrm{diag}(\sigma'(\mathbf{z}^{(l)}))</d-math>; a long product shrinks or grows exponentially (with sigmoids, <d-math>0.25^{10}\approx 10^{-6}</d-math>). This is the vanishing or exploding gradient problem.

**Practice.** The universal approximation theorem only says a wide one-hidden-layer network *exists*; deep networks often need far fewer parameters.

## Lecture 8: Convolutional Neural Networks

A fully-connected neuron on a <d-math>3\times 200\times 200</d-math> image needs 120,000 weights and must relearn each pattern at every position. CNNs use **sparse connectivity** (each output sees a small window, its receptive field, which grows with depth) and **parameter sharing** (one kernel slides over all positions).

<d-math block>
\begin{aligned}
O&=\left\lfloor\frac{W-K+2P}{S}\right\rfloor+1,\\
\#\text{params}&=C_{\text{out}}\,C_{\text{in}}\,K_H\,K_W+C_{\text{out}}.
\end{aligned}
</d-math>

Here <d-math>W</d-math> is the input width, <d-math>K</d-math> the kernel size, <d-math>P</d-math> the padding **per side**, and <d-math>S</d-math> the stride; pooling uses the same formula. For example, <d-math>W=5,K=3,P=0,S=1</d-math> gives <d-math>O=3</d-math>, and <d-math>P=(K-1)/2</d-math> with <d-math>S=1</d-math> keeps the size.

Weights have shape <d-math>(C_{\text{out}},C_{\text{in}},K_H,K_W)</d-math>, so the count does not depend on the image size. Each kernel spans **all input channels** and sums them into one feature map, so there is one output channel per kernel: 64 kernels of <d-math>3\times 3</d-math> on RGB have <d-math>64\cdot 27+64=1{,}792</d-math> parameters. Pooling has no parameters, works per channel, and lowers resolution without changing the channel count.

**Convolution versus cross-correlation.** The name comes from probability: for independent variables, <d-math>P_{X+Y}(z)=\sum_x P_X(x)P_Y(z-x)</d-math>. In 2D, convolution is <d-math>Z[i,j]=\sum_{u,v}K[u,v]A[i-u,j-v]</d-math> (flipped kernel) and cross-correlation is <d-math>Z[i,j]=\sum_{u,v}K[u,v]A[i+u,j+v]</d-math>. `nn.Conv2d` computes cross-correlation; since the kernel is learned, this only matters for hand computations with an asymmetric kernel. Convolution is translation *equivariant* (shift the input and the feature map shifts); pooling adds only approximate *invariance* to small shifts, and neither handles rotation or scaling.

**Backpropagation in CNNs** is the multivariable chain rule: a shared weight's gradient is the **sum** over all positions where it is used. In 1D with <d-math>z_k=w_1a_k+w_2a_{k+1}</d-math>, <d-math>a_2</d-math> feeds <d-math>z_1</d-math> and <d-math>z_2</d-math>, so <d-math>dL/da_2=w_2\delta_1+w_1\delta_2</d-math>. In general the input gradient is the zero-padded <d-math>\delta</d-math> cross-correlated with the flipped kernel (a true convolution), and the kernel gradient is the input cross-correlated with <d-math>\delta</d-math>. LeNet-5 <d-cite key="lecun1998gradient"></d-cite> (convolution and subsampling layers, then fully-connected layers) has about 60,000 parameters.

## Lecture 9: Regularization

**Diagnose first.** For squared loss, expected test error is <d-math>\text{bias}^2+\text{variance}+\sigma^2</d-math>. Low training accuracy means high bias (underfitting): add capacity or weaken regularization. A large train-test gap means high variance (overfitting); every tool below trades a little bias for much less variance. A big model at 99% train and 71% test accuracy beats a small one at 60% and 58%: first overfit, then regularize.

**Data and capacity.** More high-quality data is the best fix; augmentation adds pairs <d-math>(h(\mathbf{x}),y)</d-math> for transformations <d-math>h</d-math> that keep the label (rotating a "6" by 180° does not), and it is applied only to training data. **Early stopping** keeps the weights from the epoch with the best validation score and acts as an implicit regularizer. Hyperparameters, including the stopping point, are chosen on the validation set; the test set is used once.

**Norm penalties are MAP**: a Gaussian prior on the weights gives L2 (ridge), a Laplace prior gives L1 (LASSO):

<d-math block>
\text{Cost}=\frac1n\sum_{i=1}^{n}\mathcal{L}^{[i]}+\frac{\lambda}{n}\lVert\mathbf{w}\rVert_2^2\ \ (\text{L2})\quad\text{or}\quad+\frac{\lambda}{n}\lVert\mathbf{w}\rVert_1\ \ (\text{L1}).
</d-math>

With <d-math>\lambda/n</d-math>, the prior matters less as data grow. Biases are usually not penalized; deep networks sum <d-math>\lVert W^{(l)}\rVert_F^2</d-math> over layers. L2 shrinks weights proportionally and rarely to exactly 0; L1 soft-thresholds and zeroes many, because the corners of the L1 diamond lie on the axes (L1 is rarely used in deep learning). Large <d-math>\lambda</d-math> means high bias. For gradient descent, L2 becomes **weight decay**:

<d-math block>
w:=\Big(1-\frac{2\eta\lambda}{n}\Big)w-\eta\frac{\partial\mathcal{L}}{\partial w}.
</d-math>

PyTorch's `weight_decay=wd` corresponds to adding <d-math>\frac{\mathrm{wd}}{2}\lVert\mathbf{w}\rVert^2</d-math> and also decays biases by default. For Adam, an L2 penalty is not weight decay, hence AdamW.

**Dropout** <d-cite key="srivastava2014dropout"></d-cite> zeroes each hidden unit with probability <d-math>p</d-math> during training: <d-math>m_i\sim\text{Bernoulli}(1-p)</d-math>, <d-math>\tilde a_i=m_ia_i</d-math>, so <d-math>E[\tilde a_i]=(1-p)a_i</d-math>, and the original method multiplies by <d-math>1-p</d-math> at test time. **Inverted dropout** (PyTorch) instead uses <d-math>\tilde a_i=m_ia_i/(1-p)</d-math> in training and does nothing at test time.

PyTorch's <d-math>p</d-math> is the probability of *dropping*; the paper's <d-math>p</d-math> is the probability of *keeping*. Why it works: units cannot co-adapt; it trains an implicit ensemble of <d-math>2^h</d-math> weight-sharing subnetworks; and it suppresses high-order interactions. `model.train()` turns dropout on and `model.eval()` off; a higher, noisier training loss is expected.

## Lecture 10: Normalization and Initialization

**Normalization** controls the scale of signals so that deep networks train at all. Inputs are standardized, <d-math>x'=(x-\mu)/\sigma</d-math>, with training-set statistics, but this fixes only the first layer. **Batch normalization** <d-cite key="ioffe2015batch"></d-cite> does it inside the network, for each unit <d-math>j</d-math> over the minibatch:

<d-math block>
\begin{aligned}
\mu_j&=\frac1n\sum_i z_j^{[i]}, & \sigma_j^2&=\frac1n\sum_i\big(z_j^{[i]}-\mu_j\big)^2,\\
z_j'^{[i]}&=\frac{z_j^{[i]}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}, & a_j'^{[i]}&=\gamma_j z_j'^{[i]}+\beta_j.
\end{aligned}
</d-math>

The learnable <d-math>\gamma,\beta</d-math> (one per unit) restore any mean and scale; <d-math>\gamma=\sigma</d-math>, <d-math>\beta=\mu</d-math> gives the identity. BN goes after the linear layer and before the activation in the original paper, and centering cancels that layer's bias (`bias=False`).

In backprop, <d-math>\partial l/\partial\beta_j=\sum_i\partial l/\partial a_j'^{[i]}</d-math>, <d-math>\partial l/\partial\gamma_j=\sum_i(\partial l/\partial a_j'^{[i]})\,z_j'^{[i]}</d-math>, and the gradient for <d-math>z</d-math> has three paths: direct, through <d-math>\mu</d-math>, and through <d-math>\sigma^2</d-math>. Training uses batch statistics; `model.eval()` switches to running averages, which PyTorch updates as <d-math>\text{running}\leftarrow(1-m)\,\text{running}+m\cdot\text{batch}</d-math> with <d-math>m=0.1</d-math>.<d-footnote>L10 s.24 puts the momentum on the running value instead; PyTorch puts it on the new batch.</d-footnote>

**Why BN helps** is debated: the original paper says it reduces internal covariate shift, but Santurkar et al. (2018) found it still helps when shift is injected on purpose, and argue that it smooths the loss landscape. With tiny batches (about 4 or fewer) batch statistics are too noisy; freeze BN or use a batch-independent norm. **LayerNorm** normalizes each example over its features (Transformers), **InstanceNorm** each (example, channel), and **GroupNorm** each (example, channel group).

**Initialization.** All-zero weights stay symmetric, so weights are small and random. For <d-math>z=\sum_{k=1}^{m}w_ka_k</d-math> with independent zero-mean terms, <d-math>\mathrm{Var}(z)=m\,\mathrm{Var}(W)\,\mathrm{Var}(a)</d-math>, so <d-math>\mathrm{Var}(W)</d-math> must scale like <d-math>1/\text{fan\_in}</d-math> to keep the variance constant across layers:

<div class="l-page" markdown="1">

| Scheme | <d-math>\mathrm{Var}(W)</d-math> | Use with |
| ------ | -------------------------------- | -------- |
| Forward only | <d-math>1/\text{fan\_in}</d-math> | Activations near-linear around 0 |
| Xavier/Glorot <d-cite key="glorot2010understanding"></d-cite> | <d-math>2/(\text{fan\_in}+\text{fan\_out})</d-math> | Tanh; balances forward and backward passes |
| He/Kaiming | <d-math>2/\text{fan\_in}</d-math> | ReLU, which zeroes half its inputs |

</div>

<d-math>U(-r,r)</d-math> has variance <d-math>r^2/3</d-math>, so Xavier-uniform uses <d-math>r=\sqrt{6/(\text{fan\_in}+\text{fan\_out})}</d-math>.<d-footnote>L10 s.46 says `nn.Linear` defaults to He; its default has variance <d-math>1/(3\,\text{fan\_in})</d-math>, one sixth of He.</d-footnote>

## Lecture 11: Optimization and Learning Rates

In <d-math>\mathbf{w}\leftarrow\mathbf{w}-\eta\mathbf{g}</d-math> we can improve the step size <d-math>\eta</d-math> and the direction <d-math>\mathbf{g}</d-math>.

**Step size.** In the lecture, batch sizes 64 and 1024 reached about the same accuracy. Schedules include <d-math>\eta_t=\eta_0e^{-kt}</d-math>, halving every <d-math>T_0</d-math> epochs, <d-math>\eta_t=\eta_0/(1+kt)</d-math>, cyclical rates and cosine annealing, often after a linear **warmup**. Do not decay too early. A larger batch acts like a smaller learning rate (the slides write <d-math>B\propto\epsilon</d-math>, <d-math>\epsilon</d-math> the learning rate). To resume training, save the `state_dict` of the model, the optimizer and the scheduler.

**Momentum** averages past steps, so consistent directions add up and oscillations cancel. The two conventions differ by a factor of <d-math>\eta</d-math>:

<d-math block>
\begin{aligned}
\text{slides:}&\quad \Delta w_t=\alpha\Delta w_{t-1}+\eta\,\frac{\partial\mathcal{L}}{\partial w}, & w&\leftarrow w-\Delta w_t,\\
\text{PyTorch:}&\quad v_t=\beta v_{t-1}+g_t, & w&\leftarrow w-\eta\, v_t.
\end{aligned}
</d-math>

A constant gradient gives <d-math>v_t\to g/(1-\beta)</d-math>, ten times <d-math>g</d-math> at <d-math>\beta=0.9</d-math>; for <d-math>g=2</d-math>, <d-math>v</d-math> goes 2, 3.8, 5.42, toward 20. A gradient that flips sign every step settles near <d-math>\pm g/(1+\beta)\approx\pm 0.53g</d-math>. The averaging form <d-math>v=\beta v+(1-\beta)g</d-math> converges to <d-math>g</d-math> instead.

**Nesterov** evaluates the gradient at the look-ahead point <d-math>\mathbf{w}_t-\alpha\Delta\mathbf{w}_{t-1}</d-math>. **Gradient clipping** sets <d-math>\mathbf{g}\leftarrow\mathbf{g}\,c/\lVert\mathbf{g}\rVert</d-math> when <d-math>\lVert\mathbf{g}\rVert\gt c</d-math>: same direction, capped step, so it fights exploding but not vanishing gradients.

**Per-parameter step sizes.** Newton's method <d-math>\mathbf{w}\leftarrow\mathbf{w}-H^{-1}\nabla\mathcal{L}</d-math> needs <d-math>O(p^2)</d-math> memory and <d-math>O(p^3)</d-math> time for <d-math>p</d-math> parameters, so adaptive methods approximate it elementwise. **RMSProp**: <d-math>r_t=\rho r_{t-1}+(1-\rho)g_t^2</d-math>, <d-math>w\leftarrow w-\eta\,g_t/(\sqrt{r_t}+\epsilon)</d-math>. **Adam** <d-cite key="kingma2014adam"></d-cite> adds momentum and corrects the bias from starting at zero:

<d-math block>
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t, & \hat m_t&=\frac{m_t}{1-\beta_1^t},\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2, & \hat v_t&=\frac{v_t}{1-\beta_2^t},\\
\theta_t&=\theta_{t-1}-\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\end{aligned}
</d-math>

Defaults: <d-math>\alpha=10^{-3}</d-math>, <d-math>\beta_1=0.9</d-math>, <d-math>\beta_2=0.999</d-math>, <d-math>\epsilon=10^{-8}</d-math>. At <d-math>t=1</d-math>, <d-math>m_1=0.1g</d-math> and <d-math>v_1=0.001g^2</d-math>, so without correction the first step would be <d-math>0.1/\sqrt{0.001}\approx 3.2</d-math> times <d-math>\alpha</d-math>; with it, the step is <d-math>\alpha</d-math>. On L11 s.37, <d-math>\alpha,\beta</d-math> name the two decay rates, while in the paper and PyTorch <d-math>\alpha</d-math> (`lr`) is the step size and `betas=(0.9, 0.999)` the decay rates. Adam is a good default, but tuned SGD with momentum can match it.

## Putting It Together

**One recipe for losses.** Pick a distribution for <d-math>y</d-math>, write the likelihood, and minimize the negative log-likelihood. Each standard pairing gives "prediction minus target" at the output:

<div class="l-page" markdown="1">

| Output | Distribution | Loss | <d-math>\partial\mathcal{L}/\partial z</d-math> |
| ------ | ------------ | ---- | ------------------------------------------- |
| Sigmoid | Bernoulli | Binary cross-entropy | <d-math>\hat y-y</d-math> |
| Softmax | Categorical | Cross-entropy | <d-math>\hat{\mathbf{y}}-\mathbf{y}</d-math> |
| Identity | Gaussian | Squared error, times <d-math>\tfrac12</d-math> | <d-math>\hat y-y</d-math> |

</div>

The same chain rule runs through L5 (path sums), L6 (autograd), L7 (the <d-math>\delta</d-math> recursion), L8 (convolutions) and L10 (batch norm), and regularization is MAP in L3, L4 and L9.

**Values for computing by hand** (no calculators at the exam):

<div class="l-page" markdown="1">

| Quantity | Value | Quantity | Value | Quantity | Value |
| -------- | ----- | -------- | ----- | -------- | ----- |
| <d-math>\sigma(0)</d-math> | 0.5 | <d-math>\sigma(\pm 1)</d-math> | 0.731 / 0.269 | <d-math>\sigma(\pm 2)</d-math> | 0.881 / 0.119 |
| <d-math>e^{-1}</d-math> | 0.368 | <d-math>e^{-2}</d-math> | 0.135 | <d-math>\ln 2</d-math> | 0.693 |
| <d-math>\ln 0.1</d-math> | −2.303 | <d-math>\ln 0.9</d-math> | −0.105 | <d-math>0.9^2,\ 0.9^3</d-math> | 0.81, 0.729 |
| <d-math>1/(1-0.9)</d-math> | 10 | <d-math>1/(1+0.9)</d-math> | 0.526 | <d-math>0.25^{10}=2^{-20}</d-math> | <d-math>9.5\times 10^{-7}</d-math> |

</div>
