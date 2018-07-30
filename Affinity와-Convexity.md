---


---

## Affinity
### Affine combination
유한 개의 벡터들로 [선형결합 (Linear combination)](https://en.wikipedia.org/wiki/Linear_combination)을 할 때, 모든 계수(coefficient)들의 합이 1인 경우를 [**Affine combination**](https://en.wikipedia.org/wiki/Affine_combination)이라고 한다. <span><script type="math/tex">n</script></span>개의 벡터 <span><script type="math/tex">\mathbf{x}_1, \cdots, \mathbf{x}_n</script></span> 에 대해서, Affine combination은 다음과 같이 표현된다. 

<div class="math"><script type="math/tex; mode=display">
\gamma_1 \mathbf{x}_1 + \cdots + \gamma_n \mathbf{x}_n
</script></div>

여기서 <span><script type="math/tex">\gamma_i \in \mathbb{R}</script></span> 는 <span><script type="math/tex">\sum_i \gamma_i = 1</script></span> 을 만족한다. 유클리드 공간(Euclidean space)에서 임의의 두 점을 선택했을 때, **두 점을 지나는 직선 상의 모든 점들**은 Affine combination으로 표현할 수 있다. 

<br/>

### Affine set
Affine combination에 대해서 닫혀있는(closed) 집합을 [**Affine set**](https://en.wikipedia.org/wiki/Affine_space) 이라고 한다. 만약 집합 <span><script type="math/tex">\mathbb{A}</script></span>가 Affine set 이라면, 이 집합에서 <span><script type="math/tex">n</script></span>개의 원소인 벡터 <span><script type="math/tex">\mathbf{x}_1, \cdots, \mathbf{x}_n \in \mathbb{A}</script></span> 를 임의로 추출했을 때, 해당 원소들의 Affine combination도 <span><script type="math/tex">\mathbb{A}</script></span>에 속하게 된다. 

<div class="math"><script type="math/tex; mode=display">
\sum_{i=1}^n \gamma_i \mathbf{x}_i \in \mathbb{S}
</script></div>

여기서 <span><script type="math/tex">\sum_i \gamma_i = 1</script></span> 이다. 

<br/>

### Affine map
**Affine map**[^affine_map]이란 **Linear map**[^linear_map]과 **Translation**[^translation]이 결합된 형태의 좌표변환을 의미한다. 

<center><big><b>Affine map = Linear map + Translation</b></big></center>

구체적으로 표현하자면, Affine set <span><script type="math/tex">\mathbb{A} \subset \mathbb{R}^n</script></span> 의 원소 <span><script type="math/tex">\mathbf{x} \in \mathbb{A}</script></span> 와 벡터 <span><script type="math/tex">\mathbf{b} \in \mathbb{R}^m</script></span> 및 행렬 <span><script type="math/tex">\mathbf{A} \in \mathbb{R}^{n \times m}</script></span> 에 대해서 다음과 같은 형태의 함수 <span><script type="math/tex">h(\cdot):\mathbb{A} \mapsto \mathbb{R}^m</script></span> 를 의미한다. 

[^affine_map]: Affine transformation, Affine function 이라고도 한다. [여기](http://mathworld.wolfram.com/AffineFunction.html)를 참고. 

[^linear_map]: 선형변환, Linear transformation, Linear function 이라고도 한다. [여기](https://en.wikipedia.org/wiki/Linear_map)를 참고. 

[^translation]: 평행이동 변환을 의미한다. [여기](https://en.wikipedia.org/wiki/Translation_(geometry))를 참고. 

<div class="math"><script type="math/tex; mode=display">
h(\mathbf{x}) = \mathbf{A}^\mathsf{T} \mathbf{x} + \mathbf{b}
</script></div>



<br/>

<center><b>여러가지 형태의 Affine map</b></center>
<br/>
<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/2D_affine_transformation_matrix.svg/512px-2D_affine_transformation_matrix.svg.png" alt="convex_fn"/></center>
<center><small>(출처: 위키피디아)</small></center>


<br/>


## Convexity
### Convex combination
Affine combination에서 **모든 계수들이 0 이상**이라는 추가적인 제약조건이 있는 경우를 [**Convex combination**](https://en.wikipedia.org/wiki/Convex_combination) 이라고 한다. <span><script type="math/tex">n</script></span>개의 벡터 <span><script type="math/tex">\mathbf{x}_1, \cdots, \mathbf{x}_n</script></span>에 대해서, Convex combination은 다음과 같이 표현된다. 

<div class="math"><script type="math/tex; mode=display">
\lambda_1 \mathbf{x}_1 + \cdots + \lambda_n \mathbf{x}_n
</script></div>

여기서 <span><script type="math/tex">\lambda_i \in \mathbb{R}</script></span> 는 <span><script type="math/tex">\lambda_i \ge 0</script></span> 과 <span><script type="math/tex">\sum_i \lambda_i = 1</script></span> 을 만족한다. 유클리드 공간에서 임의의 두 점을 선택했을 때, **두 점을 잇는 직선 사이의 모든 점들**은 Convex combination으로 표현할 수 있다. 

<br/>

### Convex set
Convex combination에 대해서 닫혀있는 집합을 [**Convex set**](https://en.wikipedia.org/wiki/Convex_set) 이라고 한다. 만약 집합 <span><script type="math/tex">\mathbb{S}</script></span>가 Convex set 이라면, 이 집합에서 <span><script type="math/tex">n</script></span>개의 원소인 벡터 <span><script type="math/tex">\mathbf{x}_1, \cdots, \mathbf{x}_n \in \mathbb{S}</script></span> 를 임의로 추출했을 때, 해당 원소들의 Convex combination도 <span><script type="math/tex">\mathbb{S}</script></span>에 속하게 된다. 

<div class="math"><script type="math/tex; mode=display">
\sum_{i=1}^n \lambda_i \mathbf{x}_i \in \mathbb{S}
</script></div>

여기서 <span><script type="math/tex">\lambda_i \ge 0</script></span> 이고 <span><script type="math/tex">\sum_i \lambda_i = 1</script></span> 이다. 따라서 정의에 의해, **모든 Affine set은 Convex set** 이라고 할 수 있다. 


<br/>


<center><img src="https://gem763.github.io/assets/img/20180729/convex_set.png" alt="convex_set"/></center>
<center><small>(출처: 위키피디아)</small></center>

<br/>




### Convex function
Convex set <span><script type="math/tex">\mathbb{S}</script></span>에서 추출된 원소들 <span><script type="math/tex">\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{S}</script></span> 와 <span><script type="math/tex">\lambda \in [0, 1]</script></span>에 대하여,  함수 <span><script type="math/tex">f(\cdot): \mathbb{S} \mapsto \mathbb{R}</script></span>가 다음의 부등식을 만족할 때, 이 함수를 [**Convex function**](https://en.wikipedia.org/wiki/Convex_function)이라고 부른다. 

<div class="math"><script type="math/tex; mode=display">
f \bigl( \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \bigr) \le  \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2)
</script></div>

쉽게 말해서 Convex function은 **아래로 볼록** 함수를 일컫는다. 다음 차트를 보면, <span><script type="math/tex">\mathbb{S} = [a,b] \subset \mathbb{R}</script></span> 인 경우에 대해 직관적으로 이해할 수 있을 것이다. 

<center><b>Convex function의 형태</b></center>
<center><img src="https://gem763.github.io/assets/img/20180729/convex_fn.PNG" alt="convex_fn"/></center>
<center><small>(출처: https://am207.github.io)</small></center>

<br/>

만약 <span><script type="math/tex">-f</script></span>가 Convex function이라면, 이 경우의 함수 <span><script type="math/tex">f</script></span>를 **Concave function** (위로 볼록 함수)이라고 부른다. 

### Strictly convex function

모든 <span><script type="math/tex">\mathbf{x}_1 \ne \mathbf{x}_2 \in \mathbb{S}</script></span> 와 <span><script type="math/tex">\lambda \in (0, 1)</script></span>에 대하여 다음의 부등식이 성립하는 경우, 이 함수 <span><script type="math/tex">f(\cdot): \mathbb{S} \mapsto \mathbb{R}</script></span> 를 **Strictly convex function** 이라고 한다. 

<div class="math"><script type="math/tex; mode=display">
f \bigl( \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \bigr) \lt  \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2)
</script></div>





위의 식에서 **등호가 성립하는 경우가** <span><script type="math/tex">\mathbf{x}_1 = \mathbf{x}_2</script></span> 밖에 없는 경우, 이 함수를 **Strictly convex function**이라고 한다. 

Convex function의 정의에서와는 달리, <span><script type="math/tex">\lambda</script></span>의 범위에서 0과 1이 빠졌다는 점에 주의하기 바란다.  <span><script type="math/tex">\lambda</script></span>가 0 또는 1이라면, <span><script type="math/tex">\mathbf{x}_1 \ne \mathbf{x}_2</script></span> 에 대해서도 위의 식에서 등호가 성립하게 되고, 이는 정의에 어긋나게 된다. 마찬가지 논리로, **affine function이 Strictly convex function에 포함되지 않는다**는 점도 알 수 있다. 만약 <span><script type="math/tex">f</script></span>가 affine function 이라면, <span><script type="math/tex">\mathbf{x}_1 \ne \mathbf{x}_2</script></span> 에 대해서 윗식의 등호가 성립하게 되기 때문이다. 따라서 Strictly convex function은, Convex function 중 선형(Linear)인 구간이 전혀 없는 함수로 이해할 수 있다. 


<br/>

> <big><b>비교: Combinations and Sets</b></big>
> 
> 유한 개의 벡터 <span><script type="math/tex">\mathbf{x}_1, \cdots, \mathbf{x}_n</script></span>와 <span><script type="math/tex">\lambda_i \in \mathbb{R}</script></span> 에 대하여 Affine combination과 Convex combination는 다음과 같이 선형결합(Linear combination)의 형태가 된다. 
> <div class="math"><script type="math/tex; mode=display">\lambda_1 \mathbf{x}_1 + \cdots + \lambda_n \mathbf{x}_n</script></div>
> 
> 단 제약조건이 다르다. 
> 
> 
>| <center>Affine combination</center> | <center>Convex combination</center> |
>|--|--|
>| <script type="math/tex; mode=display">\sum_i \lambda_i = 1</script> | <script type="math/tex; mode=display">\sum_i \lambda_i = 1, ~\lambda_i \in [0,1]</script> |
>
> Affine set과 Convex set은 각각 Affine combination과 Convex combination에 닫혀있는 공간을 의미한다. 임의의 원소 <span><script type="math/tex">\mathbf{x}_1, \mathbf{x}_2</script></span>가 포함되어 있는 **최소한의 Affine set과 Convex set**을 구성해보면 다음 그림과 같다.  [^hull]
>
><br/>
><center><img src="https://gem763.github.io/assets/img/20180729/combinations.PNG" alt="convex_fn"/></center>
>
> 참고로 [Linear subspace](https://en.wikipedia.org/wiki/Linear_subspace)란, 모든 선형결합에 대해서 닫혀있는 공간을 뜻한다. 

[^hull]: 이처럼 어떤 집합(그림에서는 <span><script type="math/tex">\{ \mathbf{x}_1, \mathbf{x}_2 \}</script></span>)을 포함하고 있는 최소한의 Affine set과 Convex set을 각각 [Affine hull](https://en.wikipedia.org/wiki/Affine_hull), [Convex hull](https://en.wikipedia.org/wiki/Convex_hull) 이라고 부른다. 

<br/>


## Convex + Concave = Affine function
어떤 함수가 아래로 볼록(convex)이면서 동시에 위로 볼록(concave)이라면, 직관적으로 이 함수는 선형함수일 것으로 예상된다. 보다 명확하게 서술하면 다음과 같다. 

Affine set <span><script type="math/tex">\mathbb{A} \subset \mathbb{R}^n</script></span>에서 정의된 함수 <span><script type="math/tex">f(\cdot): \mathbb{A} \mapsto \mathbb{R}</script></span> 에 대하여, 

<div class="math"><script type="math/tex; mode=display">
(f= \text{Affine}) ~\Longleftrightarrow~ (f= \text{Convex \& Concave})
</script></div>



**Proof.**

<span><script type="math/tex">(1)\Rightarrow</script></span>

함수 <span><script type="math/tex">f</script></span>가 Affine map 이라고 하자. 정의에 의해, 벡터 <span><script type="math/tex">\mathbf{a} \in \mathbb{R}^n</script></span>과 상수 <span><script type="math/tex">b \in \mathbb{R}</script></span> 에 대하여 <span><script type="math/tex">f(\mathbf{x}) = \mathbf{a}^\mathsf{T} \mathbf{x} + b</script></span> 로 나타낼 수 있다. Affine set <span><script type="math/tex">\mathbb{A}</script></span>는 Convex set이므로, <span><script type="math/tex">\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{A}</script></span> 및 <span><script type="math/tex">\lambda \in [0,1]</script></span> 에 대하여, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
f \left( \mathbf{\lambda \mathbf{x}_1} + (1-\lambda) \mathbf{x}_2 \right)
&= \mathbf{a}^\mathsf{T} \left( \mathbf{\lambda \mathbf{x}_1} + (1-\lambda) \mathbf{x}_2 \right) + b \\
&= \lambda (\mathbf{a}^\mathsf{T} \mathbf{x}_1 + b) + (1-\lambda) (\mathbf{a}^\mathsf{T} \mathbf{x}_2 + b) \\
&= \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2)
\end{aligned}
</script></div>

즉 다음 두 개의 부등식, 
<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
f \bigl( \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \bigr) &\le  \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2) \\
f \bigl( \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \bigr) &\ge  \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2)
\end{aligned}
</script></div>

을 동시에 만족하게 되고, 따라서 함수 <span><script type="math/tex">f</script></span>는 <span><script type="math/tex">\mathbb{A}</script></span>에서 Convex 및 Concave 하게 된다. 

<br/>

<span><script type="math/tex">(2)\Leftarrow</script></span>

함수 <span><script type="math/tex">f</script></span>가 Convex & Concave 하다고 가정하자. <span><script type="math/tex">f</script></span>를 Affine function 의 형태로 유도하면 증명이 완성된다. 임의의 <span><script type="math/tex">\alpha \in \mathbb{A}</script></span>에 대하여 <span><script type="math/tex">f(\alpha) \overset{\text{let}}{=} b</script></span> 로 놓는다. Affine set <span><script type="math/tex">\mathbb{A}</script></span> 전체를 <span><script type="math/tex">-\alpha</script></span> 만큼 좌표변환한 집합 <span><script type="math/tex">\mathbb{A}'</script></span>을 다음과 같이 정의하면, 

<div class="math"><script type="math/tex; mode=display">
\mathbb{A}' = \{ \mathbf{x} - \alpha \mid \mathbf{x} \in \mathbb{A} \} \ni 0
</script></div>

즉 원점이 <span><script type="math/tex">\mathbb{A}'</script></span>에 포함되게 된다. 게다가 임의의 <span><script type="math/tex">\mathbf{z}_i \in \mathbb{A}'</script></span> 및 <span><script type="math/tex">\sum_i \theta_i = 1</script></span> 에 대하여, 

<div class="math"><script type="math/tex; mode=display">
\mathbf{x}_i \equiv \mathbf{z}_i + \alpha \in \mathbb{A}
</script></div>

<div class="math"><script type="math/tex; mode=display">
\sum_i \theta_i \mathbf{x}_i \in \mathbb{A}
</script></div>

이므로, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\sum_i\theta_i \mathbf{z}_i 
&= \sum_i \theta_i (\mathbf{x}_i - \alpha) \\
&= \sum_i \theta_i \mathbf{x}_1  - \alpha \\
&\in \mathbb{A}'
\end{aligned}
</script></div>


따라서 <span><script type="math/tex">\mathbb{A}'</script></span>는 원점을 지나는 Affine set 이라고 할 수 있다. 이제 <span><script type="math/tex">\mathbb{A}'</script></span> 에서 정의된 함수 <span><script type="math/tex">g(\cdot): \mathbb{A}' \mapsto \mathbb{R}</script></span> 를 다음과 같이 새로 정의한다. 

<div class="math"><script type="math/tex; mode=display">
g(\mathbf{z}) \equiv f(\mathbf{z+\alpha}) - b 
</script></div>

여기서 <span><script type="math/tex">g(0)</script></span> <span><script type="math/tex">= f(\alpha) - b = 0</script></span> 이다. 함수 <span><script type="math/tex">g</script></span> 은 다음의 세 가지 특성을 가지고 있다. 이들을 추가적으로 증명해보자. 
* **[Convex & Concave]** <span><script type="math/tex">g</script></span>는 <span><script type="math/tex">f</script></span>와 마찬가지로 Convex & Concave 하다
* **[Multiplication]** 모든 <span><script type="math/tex">\mathbf{z} \in \mathbb{A}'</script></span> 와 <span><script type="math/tex">\theta \ge 0</script></span>에 대하여, <span><script type="math/tex">g(\theta \mathbf{z}) = \theta g(\mathbf{z})</script></span>
* **[Additivity]** <span><script type="math/tex">\mathbf{z}_1, \mathbf{z}_2 \in \mathbb{A}'</script></span>에 대하여, <span><script type="math/tex">g(\mathbf{z}_1 + \mathbf{z}_2) = g(\mathbf{z}_1)  + g(\mathbf{z}_2)</script></span>

**Convex & Concave**
함수 <span><script type="math/tex">f</script></span>가 Convex & Concave 하므로, <span><script type="math/tex">\mathbf{z}_1, \mathbf{z}_2 \in \mathbb{A}'</script></span> 및 <span><script type="math/tex">\theta \in \mathbb{R}</script></span> 에 대하여, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
g \left( \theta \mathbf{z}_1 + (1-\theta) \mathbf{z}_2 \right) 
&= f \left( \theta \mathbf{z}_1 + (1-\theta) \mathbf{z}_2 + \alpha \right) - b \\
&= f \left( \theta (\mathbf{z}_1 + \alpha) + (1-\theta) (\mathbf{z}_2 + \alpha) \right) - b \\
&= \theta f (\mathbf{z}_1 + \alpha) + (1-\theta) f(\mathbf{z}_2 + \alpha) - b \\
&= \theta \left( g(\mathbf{z}_1) + b \right) + (1-\theta) \left( g(\mathbf{z}_2) + b \right) - b \\
&= \theta g(\mathbf{z}_1) + (1-\theta) g(\mathbf{z}_2)
\end{aligned}
</script></div>


**Multiplication** 
* <span><script type="math/tex">\theta \in [0, 1]</script></span>: Affine set <span><script type="math/tex">\mathbb{A}'</script></span>는 원점을 지나는 Convex set 이므로, 
<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
g(\theta \mathbf{z}) 
&= g(\theta \mathbf{z} + (1-\theta) 0) \\
&= \theta g(\mathbf{z}) + (1-\theta) g(0) \\
&= \theta g(\mathbf{z})
\end{aligned}
</script></div> 

* <span><script type="math/tex">\theta \gt 1</script></span>: 이 경우 <span><script type="math/tex">1/\theta\in [0,1]</script></span> 이므로, 
<div class="math"><script type="math/tex; mode=display">
g \left(\tfrac{1}{\theta} \theta \mathbf{z} + (1- \tfrac{1}{\theta}) 0 \right) = \tfrac{1}{\theta} g(\theta \mathbf{z}) + (1-\tfrac{1}{\theta}) g(0) = \tfrac{1}{\theta} g(\theta \mathbf{z})
</script></div>

<div class="math"><script type="math/tex; mode=display">
\therefore g(\theta \mathbf{z}) = \theta g(\mathbf{z})
</script></div>

따라서 모든 <span><script type="math/tex">\theta \ge 0</script></span> 에 대하여 <span><script type="math/tex">g(\theta \mathbf{z}) = \theta g(\mathbf{z})</script></span> 임을 알 수 있다. 

**Additivity**
<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
g(\mathbf{z}_1 + \mathbf{z}_2) 
&= g(\tfrac{1}{2} 2\mathbf{z}_1 + \tfrac{1}{2} 2 \mathbf{z}_2) \\
&= \tfrac{1}{2} g(2 \mathbf{z}_1) + \tfrac{1}{2} g(2 \mathbf{z}_2) \\
&=  g(\mathbf{z}_1) + g(\mathbf{z}_2) \\
\end{aligned}
</script></div>


이제 벡터 <span><script type="math/tex">\mathbf{a} = [a_i] \in \mathbb{R}^n</script></span> 을 다음과 같이 정의하자. 

<div class="math"><script type="math/tex; mode=display">
a_i = g(e_i)
</script></div>

여기서 <span><script type="math/tex">e_i \in \mathbb{R}^n</script></span> 는 <span><script type="math/tex">i</script></span> 번째 항목만 1 이고 나머지는 0인 벡터를 말한다. <span><script type="math/tex">g(\mathbf{z})</script></span> 를 전개해보면, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
g(\mathbf{z}) &= g(z_1 e_1 + \cdots + z_n e_n) \\
&=z_1 g(e_1) + \cdots + z_n g(e_n) \\
&= z_1 a_1 + \cdots + z_n a_n \\
&= \mathbf{a}^\mathsf{T} \mathbf{z}
\end{aligned}
</script></div>

따라서, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
f(\mathbf{x}) &= g(\mathbf{x}-\alpha) + b \\
&= \mathbf{a}^\mathsf{T} (\mathbf{x} - \alpha) + b \\
&= \mathbf{a}^\mathsf{T} \mathbf{x} + (b - \mathbf{a}^\mathsf{T} \alpha)
\end{aligned}
</script></div>

<br/>


## Jesen 부등식

Convex set <span><script type="math/tex">\mathbb{S}</script></span> (<span><script type="math/tex">\subset \mathbb{R}</script></span>)에서 정의된 함수 <span><script type="math/tex">f(\cdot): \mathbb{S} \mapsto \mathbb{R}</script></span> 가 있다.  유한 개의 <span><script type="math/tex">x_i \in \mathbb{S}</script></span> 및 <span><script type="math/tex">\lambda_i \ge 0</script></span> 에 대하여, <span><script type="math/tex">\sum_i \lambda_i = 1</script></span> 라고 할 때, 다음의 부등식이 성립한다. 이를 **Jensen 부등식**이라고 한다. 

* <span><script type="math/tex">f</script></span>가 **Convex function**: [^jensen_concave]
<div class="math"><script type="math/tex; mode=display">
f \left(\sum_{i=1}^n \lambda_i x_i\right) \le \sum_{i=1}^n \lambda_i f(x_i)
</script></div>

[^jensen_concave]: 참고로 <span><script type="math/tex">f</script></span>가 Concave function 인 경우에는 부등호의 방향이 반대가 된다. <script type="math/tex; mode=display">f \left(\sum_{i=1}^n \lambda_i x_i\right) \ge \sum_{i=1}^n \lambda_i f(x_i)</script> Strictly concave function 인 경우에도 마찬가지이다. 

<br/>

* <span><script type="math/tex">f</script></span>가 **Strictly convex function**: 
<div class="math"><script type="math/tex; mode=display">
f \left(\sum_{i=1}^n \lambda_i x_i\right) \le \sum_{i=1}^n \lambda_i f(x_i)
</script></div>

<div class="math"><script type="math/tex; mode=display">
 f \left(\sum_{i=1}^n \lambda_i x_i\right) = \sum_{i=1}^n \lambda_i f(x_i)  \Longleftrightarrow x_1 = \cdots = x_n
</script></div>


<br/>

**Convex function 인 경우의 증명:**
Jensen 부등식은 [귀납법 (Induction)](https://en.wikipedia.org/wiki/Mathematical_induction)으로 쉽게 증명할 수 있다. <span><script type="math/tex">f</script></span>가 Convex function인 경우에 대해서 우선 증명해보자. <span><script type="math/tex">n=2</script></span> 인 경우의 Jensen 부등식은 Convex function의 정의에 의해 자명하게 성립함을 알 수 있다. <span><script type="math/tex">n=k</script></span> 에 대해서도 다음의 Jensen 부등식이 성립한다고 가정하자.  

<div class="math"><script type="math/tex; mode=display">
f \left(\sum_{i=1}^k \lambda_i x_i\right) \le \sum_{i=1}^k \lambda_i f(x_i)
</script></div>

이제 <span><script type="math/tex">n=k+1</script></span> 에 대해서 식을 전개하면, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
f \left( \sum_{i=1}^{k+1} \lambda_i x_i \right) 
&= f \left( \sum_{i=1}^{k} \lambda_i x_i + \lambda_{k+1} x_{k+1} \right)\\
&= f \left( (1-\lambda_{k+1}) \sum_{i=1}^{k} \frac{\lambda_i}{1-\lambda_{k+1}} x_i + \lambda_{k+1} x_{k+1} \right)\\
&\le (1-\lambda_{k+1}) f \left( \sum_{i=1}^{k} \frac{\lambda_i}{1-\lambda_{k+1}} x_i \right) + \lambda_{k+1} f(x_{k+1})
\end{aligned}
</script></div>

여기서 <span><script type="math/tex">\frac{\lambda_i}{1-\lambda_{k+1}} \overset{\text{let}}{=} \eta_i</script></span> 로 치환하면, 

<div class="math"><script type="math/tex; mode=display">
\sum_{i=1}^{k} \eta_i = \frac{\lambda_1 + \cdots + \lambda_k}{1-\lambda_{k+1}} = \frac{1-\lambda_{k+1}}{1-\lambda_{k+1}} = 1
</script></div>

이고, <span><script type="math/tex">n=k</script></span> 에 대해서 Jensen 부등식이 성립한다고 했으므로, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
(1-\lambda_{k+1}) f \left( \sum_{i=1}^{k} \eta_i x_i \right) 
&\le (1-\lambda_{k+1}) \sum_{i=1}^k \eta_i f(x_i) \\
&= (1-\lambda_{k+1}) \sum_{i=1}^k \frac{\lambda_i}{1-\lambda_{k+1}} f(x_i) \\
&= \sum_{i=1}^k \lambda_i f(x_i)
\end{aligned}
</script></div>

따라서 다음과 같이 증명이 완성된다. 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
f \left( \sum_{i=1}^{k+1} \lambda_i x_i \right) 
&\le \sum_{i=1}^k \lambda_i f(x_i) + \lambda_{k+1} f(x_{k+1}) \\
&= \sum_{i=1}^{k+1} \lambda_i f(x_i)
\end{aligned}
</script></div>

<br/>

**Strictly convex function 인 경우의 증명:**
부등호의 증명방식은 Convex function인 경우와 동일하다. 등호부분만 추가로 증명하면 된다. 마찬가지로, <span><script type="math/tex">n=2</script></span> 인 경우는 자명하다. <span><script type="math/tex">n=k</script></span> 에 대해서 다음의 Jensen 부등식과 등식이 성립한다고 가정한다. 

<div class="math"><script type="math/tex; mode=display">
f \left(\sum_{i=1}^k \lambda_i x_i\right) \le \sum_{i=1}^k \lambda_i f(x_i)
</script></div>

<div class="math"><script type="math/tex; mode=display">
\underbrace{f \left(\sum_{i=1}^k \lambda_i x_i\right) = \sum_{i=1}^k \lambda_i f(x_i) \Longleftrightarrow x_1 = \cdots = x_k}_{\text{(*)}}
</script></div>

이제 <span><script type="math/tex">n=k+1</script></span> 에 대해서 식을 전개하면, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\underbrace{f \left( \sum_{i=1}^{k+1} \lambda_i x_i \right)}_{\text{(1)}}
&\le \underbrace{(1-\lambda_{k+1}) f \left( \sum_{i=1}^{k} \frac{\lambda_i}{1-\lambda_{k+1}} x_i \right) + \lambda_{k+1} f(x_{k+1})}_{\text{(2)}} \\
&\le \underbrace{\frac{1-\lambda_{k+1}}{1-\lambda_{k+1}} \sum_{i=1}^{k} \lambda_{i} f(x_i)  + \lambda_{k+1} f(x_{k+1})}_{\text{(3)}} \\
&= \sum_{i=1}^{k+1} \lambda_i f(x_i)
\end{aligned}
</script></div>

따라서 부등호 부분은 쉽게 증명된다. 이제 다음의 등호부분를 증명해보자. 

<span><script type="math/tex">\boxed{f \left(\sum_{i=1}^{k+1} \lambda_i x_i\right) = \sum_{i=1}^{k+1} \lambda_i f(x_i) \Longleftarrow x_1 = \cdots = x_{k+1}}</script></span>

<span><script type="math/tex">x_1 = \cdots = x_{k+1} \overset{\text{let}}{=} x</script></span> 라고 하면, 식(*)에 의해 (2)=(3) 임을 알 수 있다. 게다가 
<div class="math"><script type="math/tex; mode=display">
\sum_{i=1}^k \frac{\lambda_i}{1-\lambda_{k+1}} x_i = \frac{x}{1-\lambda_{k+1}} \sum_{i=1}^k \lambda_i = x = x_{k+1}
</script></div>

이므로, (1)=(2)도 성립한다.  따라서 <span><script type="math/tex">f \left(\sum_{i=1}^{k+1} \lambda_i x_i\right) = \sum_{i=1}^{k+1} \lambda_i f(x_i)</script></span> 를 얻게 된다. 

<br/>

<span><script type="math/tex">\boxed{f \left(\sum_{i=1}^{k+1} \lambda_i x_i\right) = \sum_{i=1}^{k+1} \lambda_i f(x_i) \Longrightarrow x_1 = \cdots = x_{k+1}}</script></span>

<span><script type="math/tex">f \left(\sum_{i=1}^{k+1} \lambda_i x_i\right) = \sum_{i=1}^{k+1} \lambda_i f(x_i)</script></span> 라고 하면, (1)=(2)=(3) 이 된다. 식(*)에 의해, (2)=(3)은 곧 <span><script type="math/tex">x_1 = \cdots = x_k</script></span> 를 의미한다. 한편 (1)=(2)는 <span><script type="math/tex">\sum_{i=1}^k \frac{\lambda_i}{1-\lambda_{k+1}} x_i = x_{k+1}</script></span> 를 의미하므로, 이 식에 <span><script type="math/tex">x_1 = \cdots = x_k \overset{\text{let}}{=} x</script></span> 를 대입하면, 


<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
x_{k+1} 
&= \sum_{i=1}^k \frac{\lambda_i}{1-\lambda_{k+1}} x_i \\
&= \frac{x}{1-\lambda_{k+1}} \sum_{i=1}^k \lambda_i \\
&= x
\end{aligned}
</script></div>

따라서 <span><script type="math/tex">x_1 = \cdots = x_k = x_{k+1}</script></span> 를 얻게 된다. 증명끝.



<br/>

> <big><b>확률변수의 Jensen 부등식</b></big>
> 

