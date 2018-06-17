---
layout: post
title: 선형대수학의 기초
tags: [선형대수학]
excerpt_separator: <!--more-->

---

계량투자에 필요하다고 생각되는, 선형대수학의 기초적인 부분을 정리합니다. 이 포스트는 수시로 업데이트 될 수 있습됩니다. 
<!--more-->



## 행렬의 크기

행렬 <math>$\mathbf{A} = [a\_{ij}] \in \mathbb{R}^{m \times n}$</math> 에 하나의 실수를 대응시키는 연산을 의미하며, **Norm**, **Trace**, **Determinant** 등이 있다. 
<br/>

### Norm

<math>$p = 1, 2, \infty$</math> 에 대해서, **<math>$p$</math>-Norm** 은 다음과 같이 정의된다. 

<math>$$
\Vert \mathbf{A} \Vert\_p = \left( \sum\_{i,~j} |a\_{ij} |^p \right)^{1/p} 
$$</math>

이 중 <math>$p = 2$</math> 인 경우를 프로베니우스 놈 (Frobenius norm)이라고 하며, 다음과 같이 쓰기도 한다. 

<math>$$
\Vert \mathbf{A} \Vert = \Vert \mathbf{A} \Vert\_F = \left( \sum\_{i,~j} a\_{ij}^2 \right)^{1/2}
$$</math>
<br/>

### Trace

정방행렬 <math>$\mathbf{A} = \mathbf{A}^\mathsf{T} \in \mathbb{R}^{n \times n}$</math>에 대해서, Trace는 다음과 같이 정의된다. 

<math>$$
\text{tr}(\mathbf{A}) = \sum\_i {a\_{ii}}
$$</math>

Trace는 아래와 같은 성질을 지닌다. 

* <math>$\text{tr}(c\mathbf{A}) = c ~ \text{tr}(\mathbf{A}), ~~c \in \mathbb{R}$</math>
* <math>$\text{tr}(\mathbf{A}^\mathsf{T}) = \text{tr}(\mathbf{A})$</math>
* <math>$\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$</math>
* <math>$\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})$</math>
* <math>$\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A}) = \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B})$</math> : 트레이스 트릭(Trace trick)

이중 트레이스 트릭을 이용하면, 다음과 같은 이차형식(Quadratic form)의 미분을 쉽게 할 수 있다. 벡터 <math>$\mathbf{x} \in \mathbb{R}^n$</math> 에 대해서, 

<math>$$
\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x} = \text{tr}(\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x}) = \text{tr}(\mathbf{A} \mathbf{x} \mathbf{x}^\mathsf{T}) = \text{tr}(\mathbf{x} \mathbf{x}^\mathsf{T} \mathbf{A})
$$</math>
<br/>

### Determinant

행렬식이라고도 하며, 정방행렬 <math>$\mathbf{A} = \mathbf{A}^\mathsf{T}$</math>에 대해 코팩터 확장(Cofactor expansion)이라고 불리는 재귀적인 방법으로 다음과 같이 정의된다. 

<math>$$
\det{\mathbf{A}} = |\mathbf{A}| = \sum\_i \mathbf{C}\_{i, ~j\_o} a\_{i,~j\_o} = \sum\_j  \mathbf{C}\_{i\_o, ~j} a\_{i\_o,~j}
$$</math>

* <math>$\mathbf{C}\_{i,j} = (-1)^{i+j} \mathbf{M}\\_{i,j}$</math> : 코팩터(Cofactor)
* <math>$\mathbf{M}\_{i,j}$</math> : 마이너(Minor)라고 하며, <math>$\mathbf{A}$</math>에서 <math>$i$</math>-행과 <math>$j$</math>-열을 지워서 얻어진 행렬의 행렬식

Determinant는 다음과 같은 성질을 지닌다. 

* <math>$\det{\mathbf{A}}^\mathsf{T} = \det{\mathbf{A}}$</math>
* <math>$\det{\mathbf{I}} = 1$</math>
* <math>$\det{\mathbf{A}\mathbf{B}} = \det{\mathbf{A}} ~ \det{\mathbf{B}}$</math>
* <math>$\det{\mathbf{A}}^{-1} = \frac{1}{\det{\mathbf{A}}}$</math>



## 역행렬

역행렬 <math>$\mathbf{A}^{-1}$</math>은 정방행렬인 <math>$\mathbf{A}$</math>와 다음의 관계를 만족하는 행렬을 뜻한다.

<math>$$
\mathbf{A}^{-1} \mathbf{A} = \mathbf{A} \mathbf{A}^{-1} = \mathbf{I} 
$$</math>

역행렬과 Determinant는 다음의 관계를 가진다. 


<math>$$
\mathbf{A}^{-1} = \frac{1}{\det{\mathbf{A}}} \mathbf{C}^\mathsf{T} = \frac{1}{\det{\mathbf{A}}} \text{adj}(\mathbf{A})
$$</math>

<math>$$
\mathbf{C} = 
\begin{bmatrix}
\mathbf{C}\_{1,~1} & & \mathbf{C}\_{1,~n}\\
& \ddots & \\
\mathbf{C}\_{n,~1} & & \mathbf{C}\_{N,~n}
\end{bmatrix}
$$</math>

여기서 

* <math>$\mathbf{C}$</math> : 코팩터 행렬(Cofactor matrix, comatrix)
* <math>$\mathbf{C}^\mathsf{T} = \text{adj}(\mathbf{A})$</math> : Adjugate matrix 또는 Adjoint matrix 라고 불린다. 



## 행렬의 미분


### 스칼라를 벡터로 미분

<math>$y = f(\mathbf{x}) \in \mathbb{R}, ~ \mathbf{x} \in \mathbb{R}^n$</math> 에 대하여, 

<math>$$
\nabla f = \frac{\partial f}{\partial \mathbf{x}} = 
\begin{bmatrix}
\dfrac{\partial y}{\partial x\_1} & \cdots & \dfrac{\partial y}{\partial x\_n}
\end{bmatrix}^\mathsf{T} \in \mathbb{R}^n
$$</math>

스칼라를 벡터로 미분한 결과값인 <math>$\nabla f$</math> 를 그레디언트 벡터(Gradient vector)라고 부른다. 
<br/>

### 벡터를 스칼라로 미분

<math>$\mathbf{y} = [y\_1 \cdots y\_m]^\mathsf{T} = [f\_1(x) \cdots f\_m(x)]^\mathsf{T} = \mathbf{f}(x) \in \mathbb{R}^m$</math>,  <math>$x \in \mathbb{R}$</math> 에 대하여,


<math>$$
\frac{\partial \mathbf{f}}{\partial x} = \left[ \frac{\partial y\_1}{\partial x} \cdots \frac{\partial y\_m}{\partial x} \right] \in \mathbb{R}^{1 \times m}
$$</math>
<br/>

### 벡터를 벡터로 미분

<math>$\mathbf{y} = \mathbf{f}(\mathbf{x}) \in \mathbb{R}^m, ~ \mathbf{x} \in \mathbb{R}^n$</math>에 대하여, 


<math>$$
\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
\dfrac{\partial y\_1}{\partial x\_1} & \cdots & \dfrac{\partial y\_1}{\partial x\_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y\_m}{\partial x\_1} & \cdots & \dfrac{\partial y\_m}{\partial x\_n}
\end{bmatrix}
$$</math>

벡터를 벡터로 미분한 결과값인 <math>$\mathbf{J} \in \mathbb{R}^{m \times n}$</math>를 **자코비안 행렬**(Jacobian matrix)이라고 한다. 특히 <math>$y = f(\mathbf{x}) \in \mathbb{R}, ~ \mathbf{x} \in \mathbb{R}^n$</math>에 대하여, 다변수함수 <math>$f$</math>의 이차도함수 <math>$\mathbf{H} \in \mathbb{R}^{n \times n}$</math>를 **헤시안 행렬**(Hessian matrix)이라고도 한다. 


<math>$$
\begin{aligned}
\mathbf{H} 
&= \frac{\partial^2 y}{\partial \mathbf{x}^2} = 
\frac{\partial}{\partial \mathbf{x}} \nabla f = \mathbf{J}(\nabla f) \\\\
&= \begin{bmatrix}
\dfrac{\partial^2 y}{\partial x\_1^2} & \cdots & \dfrac{\partial^2 y}{\partial x\_1 \partial x\_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial^2 y}{\partial x\_n \partial x\_1} & \cdots & \dfrac{\partial^2 y}{\partial x\_n^2} 
\end{bmatrix} 
\end{aligned} 
$$</math>
<br/>

### 스칼라를 행렬로 미분

<math>$y = f(\mathbf{X}) \in \mathbb{R}, ~ \mathbf{X} = [x\_{ij}] \in \mathbb{R}^{m \times n}$</math>에 대하여, 

<math>$$
\frac{\partial y}{\partial \mathbf{X}} = 
\begin{bmatrix}
\dfrac{\partial y}{\partial x\_{11}} & \cdots & \dfrac{\partial y}{\partial x\_{1n}} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y}{\partial x\_{m1}} & \cdots & \dfrac{\partial y}{\partial x\_{mn}} 
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$</math>
<br/>

### 주요 미분규칙

벡터 <math>$\mathbf{x}, \mathbf{w} \in \mathbb{R}^n$</math>와 행렬 <math>$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$</math>에 대하여, 

* <math>$\dfrac{\partial}{\partial \mathbf{x}} \mathbf{w}^\mathsf{T} \mathbf{x}\dfrac{\partial}{\partial \mathbf{x}} \mathbf{x}^\mathsf{T} \mathbf{w} = \mathbf{w}$</math>

* <math>$\dfrac{\partial}{\partial \mathbf{x}} \mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\mathsf{T}) ~\mathbf{x}$</math>

* <math>$\dfrac{\partial}{\partial \mathbf{A}} \text{tr}(\mathbf{B}\mathbf{A}) = \mathbf{B}^\mathsf{T}$</math>

* <math>$\dfrac{\partial}{\partial \mathbf{A}} \log(\det{A}) = (\mathbf{A}^{-1})^\mathsf{T}$</math>









