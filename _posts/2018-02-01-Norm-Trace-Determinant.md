---
layout: post
title: Norm, Trace, Determinant
tags: [Linear algebra, Math]
categories: [Linear algebra]
excerpt_separator: <!--more-->

---

**행렬에 하나의 실수값을 대응**시켜야 할 때가 있다. 예를들어 행렬 자체의 크기를 측정하거나, 행렬 간의 크기를 서로 비교해야 하는 경우이다. 하지만 행렬에는 수많은 원소가 포함되어 있기 때문에, 단 하나의 실수값을 부여하는 논리는 상황에 따라 다를 수 있다. 이 중 가장 빈번하게 쓰이는 **Norm**, **Trace**, **Determinant**에 대해 알아본다. 
<!--more-->

* TOC
{:toc}



## Norm
**Norm은 행렬의 크기를 측정하는 가장 일반적인 도구**이다. 엄밀히 말하면 Vector norm과 Matrix norm으로 구분할 수 있는데, 벡터가 행렬의 특수한 형태임을 고려하면, 결국 같은 개념이라고 봐도 무방하다. 각각에 대해서 알아보자. 

### Vector Norm
**Vector norm** 이란, 벡터 <span><script type="math/tex">\mathbf{x}, \mathbf{y} \in \mathbb{R}^n</script></span> 및 <span><script type="math/tex">\alpha \in \mathbb{R}</script></span>에 대하여 다음의 몇 가지 성질을 만족하면서, 각 벡터에 **음이 아닌 실수값을 대응**시키는 연산 <span><script type="math/tex">\Vert \cdot \Vert: \mathbb{R}^n  \mapsto \mathbb{R}_{\ge 0}</script></span>을 의미한다. 즉 벡터의 크기를 표현하는 방식 중 하나이다. 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\small\it{\text{Triangle inequality}} ~~~&\Vert \mathbf{x}+\mathbf{y} \Vert \le \Vert \mathbf{x} \Vert + \Vert \mathbf{y} \Vert \\
\small\it{\text{Absolutely homogeneous}} ~~~&\Vert \alpha \mathbf{x} \Vert = |\alpha| \Vert \mathbf{x} \Vert \\
\small\it{\text{Positive definite}} ~~~&\Vert \mathbf{x} \Vert = 0 ~\Longleftrightarrow~ \mathbf{x}=\mathbf{0}
\end{aligned}
</script></div>


위의 조건을 만족하는 Vector norm은 여러가지가 있을 수 있는데, 현실적으로는 <span><script type="math/tex">p</script></span>-norm (= <span><script type="math/tex">\ell_p</script></span>-norm)을 가장 많이 쓴다. 어떤 벡터 <span><script type="math/tex">\mathbf{x} = [x_i] \in \mathbb{R}^n</script></span> 와 실수 <span><script type="math/tex">p \ge 1</script></span> 에 대하여, 벡터 <span><script type="math/tex">\mathbf{x}</script></span>의 <span><script type="math/tex">p</script></span>-norm 은 다음과 같이 정의된다. 

<div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{x} \Vert_{p} \equiv \left( \sum_{i} |x_{i} |^p \right)^{1/p} 
</script></div>

<span><script type="math/tex">p</script></span> 값에 따라 아래와 같이 여러가지 형태가 있으며, 각자 고유의 명칭이 있다. 특히 <span><script type="math/tex">p=2</script></span> 인 Euclidean norm을 가장 많이 쓰는 편이다. 

* **Taxicab norm** 또는 **Manhattan norm** (<span><script type="math/tex">p=1</script></span>)
<div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{x} \Vert_{1} = \sum_{i} |x_{i} |
</script></div>

* **Euclidean norm** (<span><script type="math/tex">p=2</script></span>)
<div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{x} \Vert_{2} = \left( \sum_{i} x_{i}^2 \right)^{1/2} = \sqrt{\mathbf{x}^\mathsf{T} \mathbf{x}}
</script></div>

* **Maximum norm** 또는 **Infinity norm** (<span><script type="math/tex">p \to \infty</script></span>)
<div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{x} \Vert_{\infty} = \max_i |x_i|
</script></div> 


<br/>

> <big>**Hölder's inequality**</big>
> 
><span><script type="math/tex">\frac{1}{p}+\frac{1}{q}=1</script></span>를 만족하는 실수 <span><script type="math/tex">p, ~q \ge 1</script></span>에 대해서, 
><div class="math"><script type="math/tex; mode=display">
|\mathbf{x}^\mathsf{T} \mathbf{y}| \le \Vert \mathbf{x} \Vert_p  \Vert \mathbf{y} \Vert_q
></script></div> 를 **Hölder's inequality** 라고 한다. 특히 <span><script type="math/tex">p=q=2</script></span> 인 경우, 즉 
><div class="math"><script type="math/tex; mode=display">
|\mathbf{x}^\mathsf{T} \mathbf{y}| \le \Vert \mathbf{x} \Vert_2  \Vert \mathbf{y} \Vert_2
></script></div> 를 **Cauchy–Schwarz inequality** 라고 부른다. 

<br/>


### Matrix Norm
**Matrix norm** 이란, 행렬 <span><script type="math/tex">\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times m}</script></span> 및 <span><script type="math/tex">\alpha \in \mathbb{R}</script></span>에 대하여 다음의 몇 가지 성질을 만족하면서, 각 행렬에 **음이 아닌 실수값을 대응**시키는 연산 <span><script type="math/tex">\Vert \cdot \Vert: \mathbb{R}^{n \times m}  \mapsto \mathbb{R}_{\ge 0}</script></span>을 의미한다. Vector norm과 마찬가지로, 행렬의 크기를 표현하는 방식 중 하나이다. 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\small\it{\text{Triangle inequality}} ~~~&\Vert \mathbf{A}+\mathbf{B} \Vert \le \Vert \mathbf{A} \Vert + \Vert \mathbf{B} \Vert \\
\small\it{\text{Absolutely homogeneous}} ~~~&\Vert \alpha \mathbf{A} \Vert = |\alpha| \Vert \mathbf{A} \Vert \\
\small\it{\text{Positive definite}} ~~~&\Vert \mathbf{A} \Vert = 0 ~\Longleftrightarrow~ \mathbf{A}=\mathbf{O}
\end{aligned}
</script></div>

여기서 <span><script type="math/tex">\mathbf{O} \in \mathbb{R}^{n \times m}</script></span>는 모든 원소가 0인 행렬을 뜻한다. 벡터의 크기와는 달리, 행렬의 크기는 측정하기가 다소 애매한 측면이 있다. 따라서 Vector norm을 이용해서 Matrix norm 을 간접적으로 정의하는데, Vector norm으로부터 induced 되었다고 해서, 이를 **Induced norm** 이라고도 부른다. 어떤 행렬 <span><script type="math/tex">\mathbf{A} = [a_{ij}] \in \mathbb{R}^{n \times m}</script></span> 와 실수 <span><script type="math/tex">p \ge 1</script></span> 에 대하여, 행렬 <span><script type="math/tex">\mathbf{A}</script></span>의 (induced) <span><script type="math/tex">p</script></span>-norm 은 다음과 같이 정의된다. 


<div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{A} \Vert_{p} \equiv \sup_{\mathbf{x} \ne 0} \frac{\Vert \mathbf{A} \mathbf{x} \Vert_p}{\Vert \mathbf{x} \Vert_p}
</script></div>

즉 <span><script type="math/tex">\Vert \mathbf{A} \Vert_p</script></span> 는, 벡터 <span><script type="math/tex">\mathbf{A} \mathbf{x}</script></span>의 크기가 벡터 <span><script type="math/tex">\mathbf{x}</script></span>에 비해 얼마나 큰 지를 나타내는 지표라고 할 수 있다. <span><script type="math/tex">p</script></span> 값에 따라 아래처럼 여러가지 형태로 유도된다. 

* <span><script type="math/tex">\displaystyle \Vert \mathbf{A} \Vert_1 = \max_{1 \le j \le m} \sum_i |a_{ij}|</script></span>
* <span><script type="math/tex">\displaystyle \Vert \mathbf{A} \Vert_\infty = \max_{1 \le i \le n} \sum_j |a_{ij}|</script></span>
* <span><script type="math/tex">\displaystyle \Vert \mathbf{A} \Vert_2 = \sqrt{\lambda_\text{max} (\mathbf{A}^\mathsf{T} \mathbf{A})} = \sigma_{\text{max}}(\mathbf{A})</script></span>

여기서 <span><script type="math/tex">\lambda</script></span>는 [eigen value](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors), <span><script type="math/tex">\sigma</script></span>는 [singular value](https://en.wikipedia.org/wiki/Singular_value)를 의미한다. 

예를 들어보자. 행렬 <span><script type="math/tex">\mathbf{A}</script></span>가 다음과 같이 주어졌을 때, 

<div class="math"><script type="math/tex; mode=display">A = 
\begin{bmatrix}
2 & 1 & -3 \\
-1 & 0 & 1
\end{bmatrix}
</script></div>

* <span><script type="math/tex">\Vert \mathbf{A} \Vert_1 = \max(2+|-1|, 1, |-3|+1)</script></span> <span><script type="math/tex">= \max(3, 1, 4) = 4</script></span>
* <span><script type="math/tex">\Vert \mathbf{A} \Vert_\infty = \max(2+1+|-3|, |-1|+1)</script></span> <span><script type="math/tex">= \max(6, 2) = 6</script></span>

<br/>

> <big>**Frobenius norm**</big>
> 
> 앞서 언급했듯이, 행렬의 Induced norm은 Vector norm을 통해 간접적으로 표현된다. 이와는 별개로, **행렬 자체의 각 원소를 이용해서 행렬의 크기를 정의**하는 방법이 있다. 이를 **Entrywise norm**이라고 한다. 행렬 <span><script type="math/tex">\mathbf{A}</script></span>의 (entrywise) <span><script type="math/tex">p</script></span>-norm 은 다음과 같이 정의된다. 
><div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{A} \Vert_{p} = \left( \sum_{i,~j} |a_{ij} |^p \right)^{1/p} 
></script></div> 여기서 <span><script type="math/tex">p=2</script></span>인 경우를 **Frobenius norm**이라고 부르고, 다음과 같이 쓴다. 
><div class="math"><script type="math/tex; mode=display">
\Vert \mathbf{A} \Vert_{F} = \left( \sum_{i,~j} |a_{ij} |^2 \right)^{1/2} 
></script></div>


<br/>

## Trace
 **정방행렬의 대각성분을 모두 합산**한 값이다. 즉 행렬 <span><script type="math/tex">\mathbf{A} = [a_{ij}] \in \mathbb{R}^{n \times n}</script></span>에 대해서 다음과 같이 정의된다. 

<div class="math"><script type="math/tex; mode=display">
\text{tr}(\mathbf{A}) = \sum_i {a_{ii}}
</script></div> 

Trace는 아래와 같은 성질을 지닌다. 

* <span><script type="math/tex">\text{tr}(c\mathbf{A}) = c ~ \text{tr}(\mathbf{A}), ~~c \in \mathbb{R}</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A}^\mathsf{T}) = \text{tr}(\mathbf{A})</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})</script></span>
* <span><script type="math/tex">\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A})</script></span> <span><script type="math/tex">= \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B})</script></span> 

위의 성질을 이용하면, 벡터 <span><script type="math/tex">\mathbf{x} \in \mathbb{R}^n</script></span> 에 대한 이차형식(Quadratic form) <span><script type="math/tex">\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x}</script></span> 은 다음과 같이 여러가지 방식으로 표현할 수 있다. 

<div class="math"><script type="math/tex; mode=display">
\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x} = \text{tr}(\mathbf{x}^\mathsf{T} \mathbf{A} \mathbf{x}) = \text{tr}(\mathbf{A} \mathbf{x} \mathbf{x}^\mathsf{T}) = \text{tr}(\mathbf{x} \mathbf{x}^\mathsf{T} \mathbf{A})
</script></div>

<br/>

## Determinant

행렬식이라고도 한다. 정방행렬 <span><script type="math/tex">\mathbf{A} = [a_{ij}] \in \mathbb{R}^{n \times n}</script></span>에 대해, 다음과 같이 재귀적(Recurisve)인 방식으로 정의되는데, 이를 **Cofactor expansion**이라고 부른다. 임의의 <span><script type="math/tex">k</script></span>-행 또는 임의의 <span><script type="math/tex">k</script></span>-열에 대해서, 

<div class="math"><script type="math/tex; mode=display">
\det{\mathbf{A}} = |\mathbf{A}| = \sum_i \mathbf{C}_{ik} a_{ik} = \sum_j  \mathbf{C}_{kj} a_{kj}
</script></div>

* <span><script type="math/tex">\mathbf{C}_{ij} = (-1)^{i+j} \mathbf{M}_{ij}</script></span> : Cofactor
* <span><script type="math/tex">\mathbf{M}_{ij}</script></span> : Minor라고 하며, <span><script type="math/tex">\mathbf{A}</script></span>에서 <span><script type="math/tex">i</script></span>-행과 <span><script type="math/tex">j</script></span>-열을 지워서 얻어진 행렬의 Determinant

Determinant가 재귀적인 이유는, Minor인 <span><script type="math/tex">\mathbf{M}_{ij}</script></span> 역시 Determinant이기 때문이다. 즉 Minor가 한 차원씩 작아지며 결국 1차원 실수가 될 때까지 계산이 반복된다. Determinant는 다음과 같은 성질을 지닌다. 

* <span><script type="math/tex">\det{\mathbf{A}}^\mathsf{T} = \det{\mathbf{A}}</script></span>
* <span><script type="math/tex">\det{\mathbf{I}} = 1</script></span>
* <span><script type="math/tex">\det{\mathbf{A}\mathbf{B}} = \det{\mathbf{A}} ~\det{\mathbf{B}}</script></span>

<br/>

> <big>**Cofactor expansion의 일반화**</big>
> 
> 위의 Determinant 정의에서 나오는 Cofactor expansion은 임의의 열(이나 행) **한 개**를 선택하고, 해당 열(이나 행)을 따라서 <span><script type="math/tex">\mathbf{A}</script></span>의 원소와 코팩터의 원소를 서로 곱한다. 이는 좀더 일반화 할 수 있다. <span><script type="math/tex">\mathbf{A}</script></span>의 원소와 코팩터의 원소를 **각기 다른 열(이나 행)에서 선택**하는 것이다. 임의의 (<span><script type="math/tex">h,k</script></span>)-열에 대해서, 
><div class="math"><script type="math/tex; mode=display">
\sum_i \mathbf{C}_{ih} a_{ik} = 
\begin{cases}
\det{\mathbf{A}} & \text{if}~~ h \ne k \\
0 & \text{otherwise}
\end{cases}
></script></div>
>즉 <span><script type="math/tex">\mathbf{A}</script></span>와 코팩터의 각기 다른 열에서 원소를 선택하여 서로 곱하면 그 값은 0이 된다는 사실을 알 수 있다. 임의의 (<span><script type="math/tex">h,k</script></span>)-행에 대해서도 마찬가지이다. 이는 아래에서, <span><script type="math/tex">\mathbf{A}</script></span>의 역행렬을 유도하는 과정에서 사용하게 되는 주요 성질이다. 증명은 [여기](https://proofwiki.org/wiki/Matrix_Product_with_Adjugate_Matrix)를 참고. 



