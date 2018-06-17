---
layout: post
title: 선형대수학의 기초
tags: [선형대수학]
excerpt_separator: <!--more-->

---

계량투자에 필요하다고 생각되는, 선형대수학의 기초적인 부분을 정리합니다. 이 포스트는 수시로 업데이트 될 수 있습됩니다. 
<!--more-->



## 행렬의 크기

행렬 <span><script type="math/tex">\mathbf{A} = [a_{ij}] \in \mathbb{R}^{m \times n}</script></span> 에 하나의 실수를 대응시키는 연산을 의미하며, **Norm**, **Trace**, **Determinant** 등이 있다. 
<br/>

### Norm

<span><script type="math/tex">p = 1, 2, \infty</script></span> 에 대해서, **<span><script type="math/tex">p</script></span>-Norm** 은 다음과 같이 정의된다. 

<div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{A} \Vert_{p} = \left( \sum_{i,~j} |a_{ij} |^p \right)^{1/p} 
</script></div>

이 중 <span><script type="math/tex">p = 2</script></span> 인 경우를 프로베니우스 놈 (Frobenius norm)이라고 하며, 다음과 같이 쓰기도 한다. 

<div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{A} \Vert = \Vert \mathbf{A} \Vert_F = \left( \sum_{i,~j} a_{ij}^2 \right)^{1/2}
</script></div>
<br/>

### Trace

정방행렬 <span><script type="math/tex">\mathbf{A} = \mathbf{A}^\mathsf{T} \in \mathbb{R}^{n \times n}</script></span>에 대해서, Trace는 다음과 같이 정의된다. 

<div class="math"><script type="math/tex; mode=display">
\text{tr}(\mathbf{A}) = \sum_i {a_{ii}}
</script></div>

Trace는 아래와 같은 성질을 지닌다. 

* <span><script type="math/tex">\text{tr}(c\mathbf{A}) = c ~ \text{tr}(\mathbf{A}), ~~c \in \mathbb{R}</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A}^\mathsf{T}) = \text{tr}(\mathbf{A})</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A}) = \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B})</script></span> : 트레이스 트릭(Trace trick)

이중 트레이스 트릭을 이용하면, 다음과 같은 이차형식(Quadratic form)의 미분을 쉽게 할 수 있다. 벡터 <span><script type="math/tex">\mathbf{x} \in \mathbb{R}^n</script></span> 에 대해서, 

<div class="math"><script type="math/tex; mode=display">
\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x} = \text{tr}(\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x}) = \text{tr}(\mathbf{A} \mathbf{x} \mathbf{x}^\mathsf{T}) = \text{tr}(\mathbf{x} \mathbf{x}^\mathsf{T} \mathbf{A})
</script></div>
<br/>

### Determinant

행렬식이라고도 하며, 정방행렬 <span><script type="math/tex">\mathbf{A} = \mathbf{A}^\mathsf{T}</script></span>에 대해 코팩터 확장(Cofactor expansion)이라고 불리는 재귀적인 방법으로 다음과 같이 정의된다. 

<div class="math"><script type="math/tex; mode=display">
\det{\mathbf{A}} = |\mathbf{A}| = \sum_i \mathbf{C}_{i, ~j_o} a_{i,~j_o} = \sum_j  \mathbf{C}_{i_o, ~j} a_{i_o,~j}
</script></div>

* <span><script type="math/tex">\mathbf{C}_{i,j} = (-1)^{i+j} \mathbf{M}_{i,j}</script></span> : 코팩터(Cofactor)
* <span><script type="math/tex">\mathbf{M}_{i,j}</script></span> : 마이너(Minor)라고 하며, <span><script type="math/tex">\mathbf{A}</script></span>에서 <span><script type="math/tex">i</script></span>-행과 <span><script type="math/tex">j</script></span>-열을 지워서 얻어진 행렬의 행렬식

Determinant는 다음과 같은 성질을 지닌다. 

* <span><script type="math/tex">\det{\mathbf{A}}^\mathsf{T} = \det{\mathbf{A}}</script></span>
* <span><script type="math/tex">\det{\mathbf{I}} = 1</script></span>
* <span><script type="math/tex">\det{\mathbf{A}\mathbf{B}} = \det{\mathbf{A}} ~ \det{\mathbf{B}}</script></span>
* <span><script type="math/tex">\det{\mathbf{A}}^{-1} = \frac{1}{\det{\mathbf{A}}}</script></span>



## 역행렬

역행렬 <span><script type="math/tex">\mathbf{A}^{-1}</script></span>은 정방행렬인 <span><script type="math/tex">\mathbf{A}</script></span>와 다음의 관계를 만족하는 행렬을 뜻한다.

<div class="math"><script type="math/tex; mode=display">
\mathbf{A}^{-1} \mathbf{A} = \mathbf{A} \mathbf{A}^{-1} = \mathbf{I} 
</script></div>

역행렬과 Determinant는 다음의 관계를 가진다. 


<div class="math"><script type="math/tex; mode=display">
\mathbf{A}^{-1} = \frac{1}{\det{\mathbf{A}}} \mathbf{C}^\mathsf{T} = \frac{1}{\det{\mathbf{A}}} \text{adj}(\mathbf{A})
</script></div>

<div class="math"><script type="math/tex; mode=display">
\mathbf{C} = 
\begin{bmatrix}
\mathbf{C}_{1,~1} & & \mathbf{C}_{1,~n}\\
& \ddots & \\
\mathbf{C}_{n,~1} & & \mathbf{C}_{N,~n}
\end{bmatrix}
</script></div>

여기서 

* <span><script type="math/tex">\mathbf{C}</script></span> : 코팩터 행렬(Cofactor matrix, comatrix)
* <span><script type="math/tex">\mathbf{C}^\mathsf{T} = \text{adj}(\mathbf{A})</script></span> : Adjugate matrix 또는 Adjoint matrix 라고 불린다. 



## 행렬의 미분


### 스칼라를 벡터로 미분

<span><script type="math/tex">y = f(\mathbf{x}) \in \mathbb{R}, ~ \mathbf{x} \in \mathbb{R}^n</script></span> 에 대하여, 

<div class="math"><script type="math/tex; mode=display">
\nabla f = \frac{\partial f}{\partial \mathbf{x}} = 
\begin{bmatrix}
\dfrac{\partial y}{\partial x_1} & \cdots & \dfrac{\partial y}{\partial x_n}
\end{bmatrix}^\mathsf{T} \in \mathbb{R}^n
</script></div>

스칼라를 벡터로 미분한 결과값인 <span><script type="math/tex">\nabla f</script></span> 를 그레디언트 벡터(Gradient vector)라고 부른다. 
<br/>

### 벡터를 스칼라로 미분

<span><script type="math/tex">\mathbf{y} = [y_1 \cdots y_m]^\mathsf{T} = [f_1(x) \cdots f_m(x)]^\mathsf{T} = \mathbf{f}(x) \in \mathbb{R}^m</script></span>,  <span><script type="math/tex">x \in \mathbb{R}</script></span> 에 대하여,


<div class="math"><script type="math/tex; mode=display">
\frac{\partial \mathbf{f}}{\partial x} = \left[ \frac{\partial y_1}{\partial x} \cdots \frac{\partial y_m}{\partial x} \right] \in \mathbb{R}^{1 \times m}
</script></div>
<br/>

### 벡터를 벡터로 미분

<span><script type="math/tex">\mathbf{y} = \mathbf{f}(\mathbf{x}) \in \mathbb{R}^m, ~ \mathbf{x} \in \mathbb{R}^n</script></span>에 대하여, 


<div class="math"><script type="math/tex; mode=display">
\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
\dfrac{\partial y_1}{\partial x_1} & \cdots & \dfrac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y_m}{\partial x_1} & \cdots & \dfrac{\partial y_m}{\partial x_n}
\end{bmatrix}
</script></div>

벡터를 벡터로 미분한 결과값인 <span><script type="math/tex">\mathbf{J} \in \mathbb{R}^{m \times n}</script></span>를 **자코비안 행렬**(Jacobian matrix)이라고 한다. 특히 <span><script type="math/tex">y = f(\mathbf{x}) \in \mathbb{R}, ~ \mathbf{x} \in \mathbb{R}^n</script></span>에 대하여, 다변수함수 <span><script type="math/tex">f</script></span>의 이차도함수 <span><script type="math/tex">\mathbf{H} \in \mathbb{R}^{n \times n}</script></span>를 **헤시안 행렬**(Hessian matrix)이라고도 한다. 


<div class="math"><script type="math/tex; mode=display">
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
</script></div>
<br/>

### 스칼라를 행렬로 미분

<span><script type="math/tex">y = f(\mathbf{X}) \in \mathbb{R}, ~ \mathbf{X} = [x_{ij}] \in \mathbb{R}^{m \times n}</script></span>에 대하여, 

<div class="math"><script type="math/tex; mode=display">
\frac{\partial y}{\partial \mathbf{X}} = 
\begin{bmatrix}
\dfrac{\partial y}{\partial x_{11}} & \cdots & \dfrac{\partial y}{\partial x_{1n}} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y}{\partial x_{m1}} & \cdots & \dfrac{\partial y}{\partial x_{mn}} 
\end{bmatrix} \in \mathbb{R}^{m \times n}
</script></div>
<br/>

### 주요 미분규칙

벡터 <span><script type="math/tex">\mathbf{x}, \mathbf{w} \in \mathbb{R}^n</script></span>와 행렬 <span><script type="math/tex">\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}</script></span>에 대하여, 

* <span><script type="math/tex">\dfrac{\partial}{\partial \mathbf{x}} \mathbf{w}^\mathsf{T} \mathbf{x}\dfrac{\partial}{\partial \mathbf{x}} \mathbf{x}^\mathsf{T} \mathbf{w} = \mathbf{w}</script></span>

* <span><script type="math/tex">\dfrac{\partial}{\partial \mathbf{x}} \mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\mathsf{T}) ~\mathbf{x}</script></span>

* <span><script type="math/tex">\dfrac{\partial}{\partial \mathbf{A}} \text{tr}(\mathbf{B}\mathbf{A}) = \mathbf{B}^\mathsf{T}</script></span>

* <span><script type="math/tex">\dfrac{\partial}{\partial \mathbf{A}} \log(\det{A}) = (\mathbf{A}^{-1})^\mathsf{T}</script></span>









