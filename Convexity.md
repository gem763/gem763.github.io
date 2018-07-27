---


---

## Convex combination
유한 개의 벡터들로 [선형결합 (Linear combination)](https://en.wikipedia.org/wiki/Linear_combination)을 할 때, 모든 계수(coefficient)들이 0 이상이고 합이 1인 경우, 이를 [**Convex combination**](https://en.wikipedia.org/wiki/Convex_combination)이라고 한다. <span><script type="math/tex">n</script></span>개의 벡터 <span><script type="math/tex">\mathbf{x}_1, \cdots, \mathbf{x}_n</script></span> 에 대해서, Convex combination은 다음과 같이 표현된다. 

<div class="math"><script type="math/tex; mode=display">
\sum_{i=1}^n \lambda_i \mathbf{x}_i = \lambda_1 \mathbf{x}_1 + \cdots + \lambda_n \mathbf{x}_n
</script></div>

여기서 <span><script type="math/tex">\lambda_i \in \mathbb{R}</script></span> 는 <span><script type="math/tex">\lambda_i \ge 0</script></span> 과 <span><script type="math/tex">\sum_i \lambda_i = 1</script></span> 을 만족한다. 유클리드 공간(Euclidean space)에서 임의의 두 점을 선택했을 때, 두 점 사이를 잇는 직선 상의 모든 점들은 Convex combination으로 표현할 수 있다. 

<br/>

## Convex set
Convex combination에 대해서 닫혀있는(closed) 집합을 [**Convex set**](https://en.wikipedia.org/wiki/Convex_set) 이라고 한다. 즉 집합 <span><script type="math/tex">\mathbb{S}</script></span>가 Convex set 이라면, 이 집합에서 <span><script type="math/tex">n</script></span>개의 원소(벡터) <span><script type="math/tex">\mathbf{x}_1, \cdots, \mathbf{x}_n \in \mathbb{S}</script></span> 를 임의로 추출했을 때, 해당 원소들의 Convex combination도 <span><script type="math/tex">\mathbb{S}</script></span>에 속하게 된다. 

<div class="math"><script type="math/tex; mode=display">
\sum_{i=1}^n \lambda_i \mathbf{x}_i \in \mathbb{S}
</script></div>

여기서 <span><script type="math/tex">\lambda_i \ge 0</script></span> 이고 <span><script type="math/tex">\sum_i \lambda_i = 1</script></span> 이다. 

<br/>


<center><img src="https://gem763.github.io/assets/img/20180729/convex_set.png" alt="convex_set"/></center>
<center><small>(출처: 위키피디아)</small></center>


## Convex function
### Convex function
Convex set <span><script type="math/tex">\mathbb{S}</script></span> 에서 정의된 함수 <span><script type="math/tex">f(\cdot): \mathbb{S} \mapsto \mathbb{R}</script></span>가 있다. <span><script type="math/tex">\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{S}</script></span> 와 <span><script type="math/tex">\lambda \in [0, 1]</script></span>에 대하여 다음을 만족하는 경우, 이 함수를 [**Convex function**](https://en.wikipedia.org/wiki/Convex_function)이라고 부른다. 

<div class="math"><script type="math/tex; mode=display">
f \bigl( \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \bigr) \le  \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2)
</script></div>

쉽게 말해서 Convex function은 **아래로 볼록** 함수를 일컫는다. 다음 차트를 통해 직관적으로 이해할 수 있을 것이다. <span><script type="math/tex">\mathbb{S} = [a,b] \subset \mathbb{R}</script></span> 인 경우에 해당한다. 

<center><b>Convex function의 형태</b></center>
<center><img src="https://gem763.github.io/assets/img/20180729/convex_fn.PNG" alt="convex_fn"/></center>
<center><small>(출처: https://am207.github.io)</small></center>

<br/>

정의에 의해, Convex function은 다음의 성질을 가지고 있다. 

<div class="math"><script type="math/tex; mode=display">
\begin{matrix}
(\mathbf{x}_1=\mathbf{x}_2) ~\text{or}~ (\lambda = 0) ~\text{or}~ (\lambda = 1) ~\text{or}~ (f = {\small\text{affine function}}) \\[5pt]
\Large\Downarrow \\[5pt]
f \bigl( \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \bigr) =  \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2)
\end{matrix}
</script></div>

참고로 [affine function](http://mathworld.wolfram.com/AffineFunction.html) 이란, 벡터 <span><script type="math/tex">\mathbf{x}, \mathbf{a}, \mathbf{b} \in \mathbb{R}^n</script></span> 에 대하여 <span><script type="math/tex">\mathbf{a}^\mathsf{T} \mathbf{x} + \mathbf{b}</script></span> 의 형태를 지닌 함수[^affine]를 뜻한다. 따라서 **선형함수도 Convex function에 포함**된다는 사실을 알 수 있다. 증명은 생략.

[^affine]: affine function은, 상수항이 있다는 점을 제외하고는 선형함수(Linear function)와 동일하다. 선형함수의 보다 일반적인 형태라고 볼 수 있다. 

<br/>

### Strictly convex function

만약 모든 <span><script type="math/tex">\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{S}</script></span> 와 <span><script type="math/tex">\lambda \in (0, 1)</script></span>에 대하여, Convex function <span><script type="math/tex">f</script></span>가 다음의 부등식을 만족하되

<div class="math"><script type="math/tex; mode=display">
f \bigl( \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \bigr) \le  \lambda f(\mathbf{x}_1) + (1-\lambda) f(\mathbf{x}_2)
</script></div>

위의 식에서 **등호가 성립하는 경우가** <span><script type="math/tex">\mathbf{x}_1 = \mathbf{x}_2</script></span> 밖에 없는 경우, 이 함수를 **Strictly convex function**이라고 한다. 

Convex function의 정의에서와는 달리, <span><script type="math/tex">\lambda</script></span>의 범위에서 0과 1이 빠졌다는 점에 주의하기 바란다.  <span><script type="math/tex">\lambda</script></span>가 0 또는 1이라면, <span><script type="math/tex">\mathbf{x}_1 \ne \mathbf{x}_2</script></span> 에 대해서도 위의 식에서 등호가 성립하게 되고, 이는 정의에 어긋나게 된다. 마찬가지 논리로, **affine function이 Strictly convex function에 포함되지 않는다**는 점도 알 수 있다. 만약 <span><script type="math/tex">f</script></span>가 affine function 이라면, <span><script type="math/tex">\mathbf{x}_1 \ne \mathbf{x}_2</script></span> 에 대해서 윗식의 등호가 성립하게 되기 때문이다. 따라서 Strictly convex function은, Convex function 중 선형(Linear)인 구간이 전혀 없는 함수로 이해할 수 있다. 



<br/>

### Concave function

만약 <span><script type="math/tex">-f</script></span>가 Convex function 이라면, <span><script type="math/tex">f</script></span>를 **Concave function** (위로 볼록 함수)이라고 부른다. 마찬가지로 <span><script type="math/tex">-f</script></span>가 Strictly convex function 이라면, <span><script type="math/tex">f</script></span>는 Strictly concave function 이 된다. 

<br/>



## Jesen 부등식

Convex set <span><script type="math/tex">\mathbb{S}</script></span> (<span><script type="math/tex">\subset \mathbb{R}</script></span>)에서 정의된 함수 <span><script type="math/tex">f(\cdot): \mathbb{S} \mapsto \mathbb{R}</script></span> 가 있다.  유한 개의 <span><script type="math/tex">x_i \in \mathbb{S}</script></span> 및 <span><script type="math/tex">\lambda_i \ge 0</script></span> 에 대하여, <span><script type="math/tex">\sum_i \lambda_i = 1</script></span> 라고 할 때, 다음의 부등식이 성립한다. 이를 **Jensen 부등식**이라고 한다. 

* <span><script type="math/tex">f</script></span>가 **Convex function**: [^jensen_concave]
<div class="math"><script type="math/tex; mode=display">
f \left(\sum_{i=1}^n \lambda_i x_i\right) \le \sum_{i=1}^n \lambda_i f(x_i)
</script></div>

[^jensen_concave]: 참고로 <span><script type="math/tex">f</script></span>가 Concave function 인 경우에는 부등호의 방향이 반대가 된다. 
<div class="math"><script type="math/tex; mode=display">f \left(\sum_{i=1}^n \lambda_i x_i\right) \ge \sum_{i=1}^n \lambda_i f(x_i)</script></div>
Strictly concave function 인 경우에도 마찬가지이다. 

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

