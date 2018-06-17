---
layout: post
title: 선형대수학의 기초
tags: [선형대수학]
excerpt_separator: <!--more-->

---

계량투자에 필요하다고 생각되는, 선형대수학의 기초적인 부분을 정리합니다. 이 포스트는 수시로 업데이트 될 수 있습됩니다. 
<!--more-->



## 행렬의 크기

행렬 $\mathbf{A} = [a_{ij}] \in \mathbb{R}^{m \times n}$ 에 하나의 실수를 대응시키는 연산을 의미하며, **Norm**, **Trace**, **Determinant** 등이 있다. 
<br/>

### Norm

$p = 1, 2, \infty$ 에 대해서, **$p$-Norm** 은 다음과 같이 정의된다. 

<div class="math">$$
\Vert \mathbf{A} \Vert_p = \left( \sum_{i,~j} |a_{ij} |^p \right)^{1/p} 
$$</div>

이 중 $p = 2$ 인 경우를 프로베니우스 놈 (====TEST==== norm)이라고 하며, 다음과 같이 쓰기도 한다. 

<div class="math">$$
\Vert \mathbf{A} \Vert = \Vert \mathbf{A} \Vert_F = \left( \sum_{i,~j} a_{ij}^2 \right)^{1/2}
$$</div>
<br/>

### Trace

정방행렬 $\mathbf{A} = \mathbf{A}^\mathsf{T} \in \mathbb{R}^{n \times n}$에 대해서, Trace는 다음과 같이 정의된다. 

<div class="math">$$
\text{tr}(\mathbf{A}) = \sum_i {a_{ii}}
$$</div>

Trace는 아래와 같은 성질을 지닌다. 

* $\text{tr}(c\mathbf{A}) = c ~ \text{tr}(\mathbf{A}), ~~c \in \mathbb{R}$
* $\text{tr}(\mathbf{A}^\mathsf{T}) = \text{tr}(\mathbf{A})$
* $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$
* $\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})$
* $\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A}) = \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B})$ : 트레이스 트릭(Trace trick)

이중 트레이스 트릭을 이용하면, 다음과 같은 이차형식(Quadratic form)의 미분을 쉽게 할 수 있다. 벡터 $\mathbf{x} \in \mathbb{R}^n$ 에 대해서, 

<div class="math">$$
\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x} = \text{tr}(\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x}) = \text{tr}(\mathbf{A} \mathbf{x} \mathbf{x}^\mathsf{T}) = \text{tr}(\mathbf{x} \mathbf{x}^\mathsf{T} \mathbf{A})
$$</div>
<br/>

### Determinant

행렬식이라고도 하며, 정방행렬 $\mathbf{A} = \mathbf{A}^\mathsf{T}$에 대해 코팩터 확장(Cofactor expansion)이라고 불리는 재귀적인 방법으로 다음과 같이 정의된다. 

<div class="math">$$
\det{\mathbf{A}} = |\mathbf{A}| = \sum_i \mathbf{C}_{i, ~j_o} a_{i,~j_o} = \sum_j  \mathbf{C}_{i_o, ~j} a_{i_o,~j}
$$</div>

* $\mathbf{C}_{i,j} = (-1)^{i+j} \mathbf{M}_{i,j}$ : 코팩터(Cofactor)
* $\mathbf{M}_{i,j}$ : 마이너(Minor)라고 하며, $\mathbf{A}$에서 $i$-행과 $j$-열을 지워서 얻어진 행렬의 행렬식

Determinant는 다음과 같은 성질을 지닌다. 

* $\det{\mathbf{A}}^\mathsf{T} = \det{\mathbf{A}}$
* $\det{\mathbf{I}} = 1$
* $\det{\mathbf{A}\mathbf{B}} = \det{\mathbf{A}} ~ \det{\mathbf{B}}$
* $\det{\mathbf{A}}^{-1} = \frac{1}{\det{\mathbf{A}}}$



## 역행렬

역행렬 $\mathbf{A}^{-1}$은 정방행렬인 $\mathbf{A}$와 다음의 관계를 만족하는 행렬을 뜻한다.

<div class="math">$$
\mathbf{A}^{-1} \mathbf{A} = \mathbf{A} \mathbf{A}^{-1} = \mathbf{I} 
$$</div>

역행렬과 Determinant는 다음의 관계를 가진다. 


<div class="math">$$
\mathbf{A}^{-1} = \frac{1}{\det{\mathbf{A}}} \mathbf{C}^\mathsf{T} = \frac{1}{\det{\mathbf{A}}} \text{adj}(\mathbf{A})
$$</div>

<div class="math">$$
\mathbf{C} = 
\begin{bmatrix}
\mathbf{C}_{1,~1} & & \mathbf{C}_{1,~n}\\
& \ddots & \\
\mathbf{C}_{n,~1} & & \mathbf{C}_{N,~n}
\end{bmatrix}
$$</div>

여기서 

* $\mathbf{C}$ : 코팩터 행렬(Cofactor matrix, comatrix)
* $\mathbf{C}^\mathsf{T} = \text{adj}(\mathbf{A})$ : Adjugate matrix 또는 Adjoint matrix 라고 불린다. 



## 행렬의 미분


### 스칼라를 벡터로 미분

$y = f(\mathbf{x}) \in \mathbb{R}, ~ \mathbf{x} \in \mathbb{R}^n$ 에 대하여, 

<div class="math">$$
\nabla f = \frac{\partial f}{\partial \mathbf{x}} = 
\begin{bmatrix}
\dfrac{\partial y}{\partial x_1} & \cdots & \dfrac{\partial y}{\partial x_n}
\end{bmatrix}^\mathsf{T} \in \mathbb{R}^n
$$</div>

스칼라를 벡터로 미분한 결과값인 $\nabla f$ 를 그레디언트 벡터(Gradient vector)라고 부른다. 
<br/>

### 벡터를 스칼라로 미분

$\mathbf{y} = [y_1 \cdots y_m]^\mathsf{T} = [f_1(x) \cdots f_m(x)]^\mathsf{T} = \mathbf{f}(x) \in \mathbb{R}^m$,  $x \in \mathbb{R}$ 에 대하여,


<div class="math">$$
\frac{\partial \mathbf{f}}{\partial x} = \left[ \frac{\partial y_1}{\partial x} \cdots \frac{\partial y_m}{\partial x} \right] \in \mathbb{R}^{1 \times m}
$$</div>
<br/>

### 벡터를 벡터로 미분

$\mathbf{y} = \mathbf{f}(\mathbf{x}) \in \mathbb{R}^m, ~ \mathbf{x} \in \mathbb{R}^n$에 대하여, 


<div class="math">$$
\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
\dfrac{\partial y_1}{\partial x_1} & \cdots & \dfrac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y_m}{\partial x_1} & \cdots & \dfrac{\partial y_m}{\partial x_n}
\end{bmatrix}
$$</div>

벡터를 벡터로 미분한 결과값인 $\mathbf{J} \in \mathbb{R}^{m \times n}$를 **자코비안 행렬**(Jacobian matrix)이라고 한다. 특히 $y = f(\mathbf{x}) \in \mathbb{R}, ~ \mathbf{x} \in \mathbb{R}^n$에 대하여, 다변수함수 $f$의 이차도함수 $\mathbf{H} \in \mathbb{R}^{n \times n}$를 **헤시안 행렬**(Hessian matrix)이라고도 한다. 


<div class="math">$$
\begin{aligned}
\mathbf{H} 
&= \frac{\partial^2 y}{\partial \mathbf{x}^2} = 
\frac{\partial}{\partial \mathbf{x}} \nabla f = \mathbf{J}(\nabla f) \\\\
&= \begin{bmatrix}
\dfrac{\partial^2 y}{\partial x_1^2} & \cdots & \dfrac{\partial^2 y}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial^2 y}{\partial x_n \partial x_1} & \cdots & \dfrac{\partial^2 y}{\partial x_n^2} 
\end{bmatrix} 
\end{aligned} 
$$</div>
<br/>

### 스칼라를 행렬로 미분

$y = f(\mathbf{X}) \in \mathbb{R}, ~ \mathbf{X} = [x_{ij}] \in \mathbb{R}^{m \times n}$에 대하여, 

<div class="math">$$
\frac{\partial y}{\partial \mathbf{X}} = 
\begin{bmatrix}
\dfrac{\partial y}{\partial x_{11}} & \cdots & \dfrac{\partial y}{\partial x_{1n}} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y}{\partial x_{m1}} & \cdots & \dfrac{\partial y}{\partial x_{mn}} 
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$</div>
<br/>

### 주요 미분규칙

벡터 $\mathbf{x}, \mathbf{w} \in \mathbb{R}^n$와 행렬 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$에 대하여, 

* $\dfrac{\partial}{\partial \mathbf{x}} \mathbf{w}^\mathsf{T} \mathbf{x}\dfrac{\partial}{\partial \mathbf{x}} \mathbf{x}^\mathsf{T} \mathbf{w} = \mathbf{w}$

* $\dfrac{\partial}{\partial \mathbf{x}} \mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\mathsf{T}) ~\mathbf{x}$

* $\dfrac{\partial}{\partial \mathbf{A}} \text{tr}(\mathbf{B}\mathbf{A}) = \mathbf{B}^\mathsf{T}$

* $\dfrac{\partial}{\partial \mathbf{A}} \log(\det{A}) = (\mathbf{A}^{-1})^\mathsf{T}$









