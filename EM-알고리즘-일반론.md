---


---

## Motivation

아래의 왼쪽 차트와 같은 데이터 집합이 주어졌다고 해보자. 데이터들은 3개의 그룹으로 구분되어 있고, 편의를 위해 색깔로 해당 그룹을 표시하였다. 

<center><img src="https://gem763.github.io/assets/img/20180719/cluster_data.png" alt="cluster_data"/></center>
<center><small>(출처: 위키피디아)</small></center>

이제 각 데이터가 속한 그룹을 모르고 있다고 가정해보자. 우리는 클러스터링 기법을 통해 위의 색깔로 구분된 것과 유사하게 그룹을 나누고 싶은 것이다. 

이전 포스트의 [K-means clustering](https://gem763.github.io/machine%20learning/K-means-clustering.html)이 이 문제를 해결해줄 수 있을까? 문제의 성격에 따라 해결할 수도 있고, 못할 수도 있다. 그리고 결정적으로, 위의 문제는 해결해주지 못한다. K-means clustering은 유클리드 거리를 사용하여, 모든 클러스터의 총 왜곡도를 최소화하는 방향으로 클러스터링을 수행한다. 따라서 이 문제와 같이 클러스터의 크기가 상이한 경우에는 다소 납득할 수 없는 결과가 도출될 수도 있다. 위의 오른쪽 차트처럼 말이다. [K-means clustering의 한계](https://gem763.github.io/machine%20learning/K-means-clustering.html#%ED%95%9C%EA%B3%84)를 참조하기 바란다. 

데이터가 클러스터에 속할 지의 여부를 (K-means clustering 처럼 칼같이 결정[^hard_cl]하지 말고) **확률적**으로 묘사하면 좀 낫지 않을까? 이를 위해 다음의 몇 가지를 가정하면, 

* 데이터 <span><script type="math/tex">\mathbf{x} \in \mathbb{R}^d</script></span> 는 어떤 확률변수 <span><script type="math/tex">X</script></span> 로부터 독립적으로 도출
* 각 데이터는 클러스터 <span><script type="math/tex">\mathbf{S} = \{ S_1, S_2, S_3 \}</script></span> 중 어느 하나에 포함
* 각 클러스터는 다변수 [가우시안 정규분포](https://en.wikipedia.org/wiki/Normal_distribution) <span><script type="math/tex">\mathcal{N}_d</script></span>으로 묘사


각 가우시안 정규분포의 모수 <span><script type="math/tex">\boldsymbol{\mu}_j \in \mathbb{R}^d</script></span>, <span><script type="math/tex">\mathbf{\Sigma}_j \in \mathbb{R}^{d \times d}</script></span> <span><script type="math/tex">(j=1,2,3)</script></span>에 대하여, 확률변수 <span><script type="math/tex">X</script></span>는 다음과 같이 표현할 수 있다. 

[^hard_cl]: Hard clustering 이라고 한다. 반대로, 클러스터를 확률적으로 결정하는 방식을 Soft clustering 이라고 한다. 

<div class="math"><script type="math/tex; mode=display">
X \sim 
\begin{cases}
\mathcal{N}_d (\boldsymbol{\mu}_1, \mathbf{\Sigma}_1) & \text{if}~~ X \in S_1 \\
\mathcal{N}_d (\boldsymbol{\mu}_2, \mathbf{\Sigma}_2) & \text{if}~~ X \in S_2 \\
\mathcal{N}_d (\boldsymbol{\mu}_3, \mathbf{\Sigma}_3) & \text{if}~~ X \in S_3
\end{cases}
</script></div>

어느 클러스터에 속하는지의 여부도 확률적으로 결정되어야 할 것이므로, 새로운 확률변수 <span><script type="math/tex">Z</script></span> (잠재변수라고 한다)를 도입하면 다음과 같이 조건부 확률분포로 나타날 수 있게 된다. 

<div class="math"><script type="math/tex; mode=display">
X \mid Z \sim 
\begin{cases}
\mathcal{N}_d (\boldsymbol{\mu}_1, \mathbf{\Sigma}_1) & \text{if}~~ Z=1 \\
\mathcal{N}_d (\boldsymbol{\mu}_2, \mathbf{\Sigma}_2) & \text{if}~~ Z=2 \\
\mathcal{N}_d (\boldsymbol{\mu}_3, \mathbf{\Sigma}_3) & \text{if}~~ Z=3
\end{cases}
</script></div>

여기서 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\Pr[Z=1] ~\overset{\text{let}}{=}~ \tau_1 \\
\Pr[Z=2] ~\overset{\text{let}}{=}~ \tau_2 \\
\Pr[Z=3] ~\overset{\text{let}}{=}~ \tau_3
\end{aligned}
</script></div>

이고, <span><script type="math/tex">\tau_1 + \tau_2 + \tau_3 = 1</script></span> 라고 하면, <span><script type="math/tex">X</script></span>의 최종적인 확률분포 <span><script type="math/tex">p</script></span> 는 다음의 형태가 될 것이다. 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
p(\mathbf{x}) 
&= \sum_{j=1}^n \Pr[\mathbf{X}=\mathbf{x} \mid Z=j] \Pr[Z=j] \\
&= \tau_1\mathcal{N}_d (\boldsymbol{\mu}_1, \mathbf{\Sigma}_1) + \tau_2\mathcal{N}_d (\boldsymbol{\mu}_2, \mathbf{\Sigma}_2) + \tau_3\mathcal{N}_d (\boldsymbol{\mu}_3, \mathbf{\Sigma}_3) \\
&= p(\mathbf{x}; ~\underbrace{\tau_1, \tau_2, \boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \mathbf{\Sigma}_1, \mathbf{\Sigma}_2}_{\boldsymbol{\theta}}) \\
&= p(\mathbf{x}; \boldsymbol{\theta})
\end{aligned}
</script></div>

이처럼 여러 개의 확률분포가 섞여서 만들어진 분포를 **혼합분포**(Mixture distribution)라고 하고, 특히 위의 전개에서와 같이 가우시안 정규분포로 이루어진 혼합분포를 **가우시한 혼합분포**(Gaussian Mixture distribution)이라고 부른다. 데이터가 어느 클러스터에 속해 있는지를 확인하기 위해서는 우선 해당 혼합분포를 명확하게 이해하는 것이 중요하다. 따라서 결국 우리가 해야 할 일은 혼합분포의 모수 <span><script type="math/tex">\boldsymbol{\theta}</script></span>를 추정하는 것이 된다. 

<div class="math"><script type="math/tex; mode=display">
\boldsymbol{\theta} = (\tau_1, \tau_2, \boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \mathbf{\Sigma}_1, \mathbf{\Sigma}_2)
</script></div>

그리고 이 모수를 추정하는 효과적인 방법 중 하나가 바로 [**EM 알고리즘**](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) (Expectation–maximization algorithm, 기대값 최대화 알고리즘)이다. 

다시 처음으로 돌아가서, 맨 위의 좌측 데이터 집합(original data)을 살펴보자. 이 데이터 집합이 어떤 혼합분포에서 추출되었다고 가정하고, EM 알고리즘으로 해당 모수를 추정한다. 이제 각 샘플 데이터 별로 

<center><img src="https://gem763.github.io/assets/img/20180719/cluster_by_em.png" alt="cluster_em"/></center>

<center><small>(출처: 위키피디아)</small></center>


## 모델링

어떤 확률분포로부터 데이터 집합 <span><script type="math/tex">\mathbf{X}</script></span>이 관측되었다. 이 관측데이터들은, 관측되지 않는 어떤 이산(discrete) 데이터의 집합 <span><script type="math/tex">\mathbf{Z}</script></span>에 의존한다고 가정해보자. 즉 각각의 관측 데이터 <span><script type="math/tex">\mathbf{x} \in \mathbf{X}</script></span>는 그에 대응하는 하나의 비관측 데이터 <span><script type="math/tex">\mathbf{z} \in \mathbf{Z}</script></span>을 가진다. 이런 비관측 데이터 <span><script type="math/tex">\mathbf{Z}</script></span>를 **잠재변수**(Latent variables)라고 한다. 이런 상황에서, 각 관측 데이터가 따르는 확률분포의 모수 <span><script type="math/tex">\boldsymbol{\theta}</script></span>를 추정할 수 있을까?




* **관측이 가능**한 데이터 집합 <span><script type="math/tex">\mathbf{X} \rightarrow</script></span> 각 원소는 모수가 <span><script type="math/tex">\boldsymbol{\theta}</script></span>인 어떤 확률분포로부터 출력
* **관측이 불가능**한 데이터 집합 <span><script type="math/tex">\mathbf{Z} \rightarrow</script></span> 잠재변수


우선 <span><script type="math/tex">(\mathbf{X}, \mathbf{Z})</script></span>가 모두 관측가능(Complete data) 하다는 비정상적인 가정을 하고, 모수 <span><script type="math/tex">\boldsymbol{\theta}</script></span>를 추정해보자. 이 경우 로그 우도함수는 다음과 같이 표현된다. 

<div class="math"><script type="math/tex; mode=display">
\ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z}) \equiv \ln p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})
</script></div>

가정에 의해 <span><script type="math/tex">\mathbf{X}</script></span>와 <span><script type="math/tex">\mathbf{Z}</script></span>가 모두 주어졌기 때문에, 일반적인 LME 방법에 의해 <span><script type="math/tex">\boldsymbol{\theta}</script></span>는 다음과 같이 추정된다. 

<div class="math"><script type="math/tex; mode=display">
\hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta}}{\arg \max} \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z})
</script></div>

<span><script type="math/tex">\hat{\boldsymbol{\theta}}</script></span>는 <span><script type="math/tex">\boldsymbol{\theta}</script></span>의 추정값을 의미한다. 하지만 여기서 문제가 풀린 것은 아니다. 앞서 Complete data 가정을 했으나, 실제로 <span><script type="math/tex">\mathbf{Z}</script></span>는 관측이 불가능한 잠재변수이며, 우리에게 주어진 데이터는 <span><script type="math/tex">\mathbf{X}</script></span> 뿐(Incomplete data)이기 때문이다. 따라서 현실적인 로그 우도함수는 잠재변수 <span><script type="math/tex">\mathbf{Z}</script></span>의 모든 경우의 수에 대한 주변확률밀도함수로 표현된다.

<div class="math"><script type="math/tex; mode=display">
\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X}) \equiv \ln p(\mathbf{X} \mid \boldsymbol{\theta}) = \ln \left[ \sum_\mathbf{Z} p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) \right]
</script></div>


이제 다시 <span><script type="math/tex">\boldsymbol{\theta}</script></span>를 추정해보자. 

<div class="math"><script type="math/tex; mode=display">
\hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta}}{\arg \max} \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X})
</script></div>

<div class="math"><script type="math/tex; mode=display">
\frac{\partial}{\partial \boldsymbol{\theta}} \ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X}) = \frac{\partial}{\partial \boldsymbol{\theta}} \ln \left[ \sum_\mathbf{Z} p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) \right] = 0
</script></div>

문제는 바로 이 지점에 있다. 미분이 잘 되는가? 잠재변수 <span><script type="math/tex">\mathbf{Z}</script></span>로 인해 <span><script type="math/tex">\sum_\mathbf{Z}</script></span>이 <span><script type="math/tex">\log</script></span> 속으로 들어가면서, 대수적인 미분전개가 굉장히 어려워졌다. 정확한 해를 계산하는 것이 쉽지않은 것이다. 




-----


EM 알고리즘은 이를 다음의 방법으로 해결하고자 한다. 

1. 잠재변수 <span><script type="math/tex">\mathbf{Z}</script></span>의 확률분포를 추정하여, <span><script type="math/tex">\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z})</script></span>의 <span><script type="math/tex">\mathbf{Z}</script></span>에 대한 기대값을 구한 후, 
3. 이 기대값을 최대화하는 <span><script type="math/tex">\boldsymbol{\theta}</script></span>를 추정한다.   

즉 Incomplete 로그 우도함수 <span><script type="math/tex">\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X})</script></span>를 직접적으로 최대화하기가 어려운 상황에서, **대신 Complete 로그 우도함수 <span><script type="math/tex">\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z})</script></span>의 <span><script type="math/tex">\mathbf{Z}</script></span>에 대한 기대값을 최대화하는 간접적인 방식**을 쓰자는 것이다. 

물론 이 아이디어에도 문제점은 있다. 애초에 최대화하는 대상이 다르기 때문에, 여기서 추정한 <span><script type="math/tex">\boldsymbol{\theta}</script></span>가 진정한 답이라고 볼 수 없다. EM 알고리즘은 반복적인(iterative) 방법으로 이를 극복한다. 즉 모수 <span><script type="math/tex">\boldsymbol{\theta}</script></span>와 잠재변수 <span><script type="math/tex">\mathbf{Z}</script></span>의 확률분포를 번갈아가며 반복적으로 추정함으로써 추정의 정확도를 높이는 것이다. 



<br/>

## 알고리즘
모수 <span><script type="math/tex">\boldsymbol{\theta}</script></span>의 추정값을 임의의 값 <span><script type="math/tex">\boldsymbol{\theta}^{(0)}</script></span>으로 초기화한 후, 다음의 두 단계를 반복한다. 모수의 추정값이 특정값으로 수렴하면 알고리즘을 종료한다.   
<br/>

* **E-Step**: <span><script type="math/tex">\boldsymbol{\theta}^{(t)}</script></span>가 주어진 상태에서, 로그 우도함수의 <span><script type="math/tex">\mathbf{Z}</script></span>에 대한 조건부 기대값 <span><script type="math/tex">Q</script></span>를 구하는 단계

<div class="math"><script type="math/tex; mode=display">
Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) = \sum_{\mathbf{Z}} p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)}) \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z}) 
</script></div>
<br/>


* **M-Step**: <span><script type="math/tex">\boldsymbol{\theta}^{(t)}</script></span>를 <span><script type="math/tex">\boldsymbol{\theta}^{(t+1)}</script></span>로 업데이트하는 단계

<div class="math"><script type="math/tex; mode=display"> 
\boldsymbol{\theta}^{(t+1)} = \underset{\boldsymbol{\theta}}{\arg \max} ~Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) 
</script></div>
<br/>

<div class="math"><script type="math/tex; mode=display">
\begin{matrix}
\begin{matrix}
{\scriptstyle\text{Initialize}} \\
\boldsymbol{\theta}^{(0)} \\ 
\bigg\downarrow
\end{matrix} &  &  \\
\boldsymbol{\theta}^{(t)} & \xrightarrow{\text{approach}} & \hat{\boldsymbol{\theta}} \\
{\scriptstyle\text{E-Step} \Bigg\downarrow} {\Bigg\uparrow\scriptstyle\text{M-Step}} &  & \Bigg\updownarrow \\
Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) & \xrightarrow{\phantom{\text{long}}} \ln \mathcal{L} (\boldsymbol{\theta}^{(t)}; \mathbf{X}) \xrightarrow{\phantom{long}} & \ln \mathcal{L} (\hat{\boldsymbol{\theta}}; \mathbf{X}) 
\end{matrix}
</script></div> 

<br/>


## 세부내용

### Underbound
우선, 위에서 제시한 (현실적인, 즉 incomplete data의) 로그 우도함수의 형태를 조금 변경해보자. <span><script type="math/tex">\mathbf{Z}</script></span>의 임의의 확률밀도함수 <span><script type="math/tex">q(\mathbf{Z})</script></span> [^pdf_of_Z]를 이용하면, 로그 우도함수 <span><script type="math/tex">\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X})</script></span>은 어떤 **Underbound 함수** <span><script type="math/tex">U(\boldsymbol{\theta}, q)</script></span>에 대하여 다음 식이 성립한다. (증명은 아랫쪽에)

[^pdf_of_Z]: 개별 잠재 데이터 <span><script type="math/tex">\mathbf{z} \in \mathbf{Z}</script></span>의 확률밀도함수가 아닌, 잠재변수 <span><script type="math/tex">\mathbf{Z}</script></span> 전체의 확률밀도함수임을 주의한다. 


<div class="math"><script type="math/tex; mode=display">
\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X}) \ge U(\boldsymbol{\theta}, q)
</script></div>  
   
<span><script type="math/tex">U</script></span>는 다음의 두 가지 방식으로 다시 분해될 수 있다. 향후의 전개를 위해 필요하기 때문에 미리 준비해 두는 거라고 보면 된다. 
  
<div class="math"><script type="math/tex; mode=display">
U(\boldsymbol{\theta}, q) = 
\begin{cases}
(1) ~\ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}) - \mathcal{D}_{KL} \left( q(\mathbf{Z}) \parallel p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}) \right) \\
(2) ~\operatorname{E}_{q(\mathbf{Z})} \left[ \ln \mathcal{L} (\boldsymbol{\theta} ; \mathbf{X}, \mathbf{Z}) \right] + \mathcal{H}_q(\mathbf{Z})
\end{cases}
</script></div>


이 때, 

* <span><script type="math/tex">\displaystyle \mathcal{D}_{KL} (q \parallel p) \equiv \sum_i q(i) \ln \frac{q(i)}{p(i)} \rightarrow</script></span> KL 다이버전스 (Kullback-Leiber divergence)

* <span><script type="math/tex">\mathbf{E}_{q(\mathbf{Z})}[\cdot] \rightarrow</script></span> 확률밀도함수 <span><script type="math/tex">q(\mathbf{Z})</script></span>에 대한 기대값 연산자

* <span><script type="math/tex">\displaystyle \mathcal{H}_q(\mathbf{Z}) \equiv -\sum_{\mathbf{Z}} q(\mathbf{Z}) \ln q(\mathbf{Z}) \ge 0 \rightarrow</script></span> <span><script type="math/tex">q</script></span>에 대한 <span><script type="math/tex">\mathbf{Z}</script></span>의 엔트로피


  
이는 [Jensen 부등식](https://wikidocs.net/10809)과 다음의 전개를 통해 간단히 증명된다. 


<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X})
&= \ln \left[ \sum_\mathbf{Z} p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) \right] \\
&= \ln \left[ \sum_\mathbf{Z} q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \right] \\
&= \ln \mathbf{E}_{q(\mathbf{Z})} \left[ \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \right] \\
&\ge \mathbf{E}_{q(\mathbf{Z})} \left[ \ln \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \right] \overset{\text{let}}{=} U(\boldsymbol{\theta}, q) \\\\
U(\boldsymbol{\theta}, q)
&= \mathbf{E}_{q(\mathbf{Z})} \left[ \ln \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \right] \\
&= \mathbf{E}_{q(\mathbf{Z})} \left[ \ln \frac{p(\mathbf{X} \mid \boldsymbol{\theta}) ~p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})}{q(\mathbf{Z})} \right] \\
&= \mathbf{E}_{q(\mathbf{Z})} \left[ \ln p(\mathbf{X} \mid \boldsymbol{\theta}) \right] - \mathbf{E}_{q(\mathbf{Z})} \left[ \ln \frac{q(\mathbf{Z})}{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})} \right] \\
&= \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}) - \mathcal{D}_{KL} \left( q(\mathbf{Z}) \parallel p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}) \right) \\\\
U(\boldsymbol{\theta}, q)
&= \mathbf{E}_{q(\mathbf{Z})} \left[ \ln \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} \right] \\
&= \mathbf{E}_{q(\mathbf{Z})} \left[ \ln p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) \right] - \mathbf{E}_{q(\mathbf{Z})} \left[ \ln q(\mathbf{Z}) \right] \\
&= \mathbf{E}_{q(\mathbf{Z})} \left[ \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z}) \right] + \mathcal{H}_q(\mathbf{Z})
\end{aligned}
</script></div>  
    
<br/> 

이처럼 로그 우도함수를 부등호의 형태로 바꾼 이유는 무엇일까? 등호(=)를 포기한 대신 <span><script type="math/tex">\log</script></span> **안에 summation이 있는 구조를 제거**한 것이 그 본질이다. 아래 식을 보자. 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\arg \max_{\boldsymbol{\theta}} U(\boldsymbol{\theta}, q)
&= \underset{\boldsymbol{\theta}}{\arg \max} \Bigl( \mathbf{E}_{q(\mathbf{Z})} \left[ \ln \mathcal{L} (\boldsymbol{\theta} ; \mathbf{X}, \mathbf{Z}) \right] + \mathcal{H}_q(\mathbf{Z}) \Bigr) \\
&= \underset{\boldsymbol{\theta}}{\arg \max} ~\mathbf{E}_{q(\mathbf{Z})} \left[ \ln \mathcal{L} (\boldsymbol{\theta} ; \mathbf{X}, \mathbf{Z}) \right] \\
&= \underset{\boldsymbol{\theta}}{\arg \max} \sum_\mathbf{Z} q(\mathbf{Z}) \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z})
\end{aligned}
</script></div>

<span><script type="math/tex">U</script></span>에 MLE를 적용하는 것은 <span><script type="math/tex">\mathbf{E}_{q(\mathbf{Z})} \left[ \ln \mathcal{L} (\boldsymbol{\theta} ; \mathbf{X}, \mathbf{Z}) \right]</script></span>에 MLE를 적용하는 것과 동일한데, 이 경우 summation이 <span><script type="math/tex">\log</script></span> 밖으로 빠져나오게 된다. 즉 <span><script type="math/tex">U</script></span>는 <span><script type="math/tex">\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X})</script></span>보다 훨씬 더 쉽게 <span><script type="math/tex">\boldsymbol{\theta}</script></span>에 대해 미분할 수 있는 것이다.  물론 <span><script type="math/tex">U</script></span>에 LME를 적용한다고 해서 정확한 해(즉 모수 <span><script type="math/tex">\boldsymbol{\theta}</script></span>의 추정값)를 구할 수는 없으며, 이는 반복적(iterative)인 방법의 추정이 필요한 이유가 된다. 

<br/>



### E-Step

기대값 단계(Expectation step) 즉, 로그 우도함수의 기대값을 도출하는 단계이다. <span><script type="math/tex">\boldsymbol{\theta}^{(t)}</script></span>이 주어진 상태에서, 우선 우리는 <span><script type="math/tex">\mathbf{Z}</script></span>의 확률밀도함수 <span><script type="math/tex">q</script></span>를 다음과 같이 추정할 것이다. 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
q^{(t)} 
&= \underset{q}{\arg \max} ~U(\boldsymbol{\theta}^{(t)}, q) \\
&= \underset{q}{\arg \min} ~\mathcal{D}_{KL} \left( q(\mathbf{Z}) \parallel p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)}) \right) \\
&= p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})
\end{aligned}
</script></div>


즉 <span><script type="math/tex">q</script></span>는 로그 우도함수의 Underbound <span><script type="math/tex">U</script></span>를 최대화하는 값으로 추정되며, 결국 **<span><script type="math/tex">\mathbf{Z}</script></span>의 사후확률분포**로 계산된다. 이렇게 하는 이유는, <span><script type="math/tex">U(\boldsymbol{\theta}^{(t)}, q^{(t)})</script></span>를 점진적으로 크게 만들기 위해서이다(아래 [알고리즘 작동원리](https://wikidocs.net/10699#_7)에서 설명). 

이제 <span><script type="math/tex">\ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z})</script></span>의 <span><script type="math/tex">q^{(t)}</script></span>에 대한 기대값을 구할 수 있다. 이 값을 <span><script type="math/tex">Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})</script></span>라고 두자. 이는 <span><script type="math/tex">\boldsymbol{\theta}</script></span>에 대한 함수가 된다. 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\mathbf{E}_{q^{(t)}} \left[ \ln \mathcal{L}(\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z}) \right] 
&= \sum_\mathbf{Z} q^{(t)} (\mathbf{Z}) \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z}) \\
&= \sum_\mathbf{Z} p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)}) \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X}, \mathbf{Z}) \\
&\overset{\text{let}}{=} Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})
\end{aligned}
</script></div>


### M-Step

최대화 단계(Maximization step) 즉 로그 우도함수의 기대값을 최대화하는 모수값을 찾는 단계이다. <span><script type="math/tex">Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})</script></span>가 주어진 상태에서, MLE를 통해 <span><script type="math/tex">\boldsymbol{\theta}^{(t)}</script></span>을 <span><script type="math/tex">\boldsymbol{\theta}^{(t+1)}</script></span>로 업데이트 한다. 

<div class="math"><script type="math/tex; mode=display">
\boldsymbol{\theta}^{(t+1)} 
= \underset{\boldsymbol{\theta}}{\arg \max} ~U(\boldsymbol{\theta}, q^{(t)})
= \underset{\boldsymbol{\theta}}{\arg \max} ~Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) 
</script></div>


<br/>

## 알고리즘의 작동원리


모수 <span><script type="math/tex">\boldsymbol{\theta}</script></span>의 최종 추정값 <span><script type="math/tex">\hat{\boldsymbol{\theta}}</script></span>은 정의에 의해 다음과 같이 표현된다. 

<div class="math"><script type="math/tex; mode=display">
\hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta}}{\arg \max} ~\ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X})
</script></div>

<div class="math"><script type="math/tex; mode=display">
\ln \mathcal{L}(\hat{\boldsymbol{\theta}} ; \mathbf{X}) = \max_\boldsymbol{\theta} \ln \mathcal{L} (\boldsymbol{\theta; \mathbf{X}})
</script></div>


[개요](https://wikidocs.net/10699#_1)에서 언급했듯이, 우리는 이 <span><script type="math/tex">\hat{\boldsymbol{\theta}}</script></span> 값을 정확히 계산할 수 없다. 따라서, 다음과 같이 정의된 <span><script type="math/tex">t</script></span> 시점에서의 Underbound <span><script type="math/tex">U_t</script></span>에 대해서, 

<div class="math"><script type="math/tex; mode=display">
U_{t} \equiv U(\boldsymbol{\theta}^{(t)}, q^{(t)})
</script></div>

수열 <span><script type="math/tex">\lbrace U_0, U_1, U_2, \cdots ~\rbrace</script></span>이 <span><script type="math/tex">\ln \mathcal{L} (\hat{\boldsymbol{\theta}}, \mathbf{X})</script></span>에 점점 가까워짐을 우선 증명하고, 결과적으로 **<span><script type="math/tex">\lbrace \boldsymbol{\theta}^{(0)}, \boldsymbol{\theta}^{(1)}, \boldsymbol{\theta}^{(2)}, \cdots ~\rbrace</script></span> 역시 <span><script type="math/tex">\hat{\boldsymbol{\theta}}</script></span>에 가까워진다고 주장**할 것이다. (하지만 <span><script type="math/tex">\hat{\boldsymbol{\theta}}</script></span>에 가까워진다는 것이, 수렴한다는 의미는 아니다.  EM 알고리즘은 전역해를 보장하지 않는다. [한계](https://wikidocs.net/11009#_8) 참조)


<span><script type="math/tex">\boldsymbol{\theta}^{(t)}</script></span>가 주어졌다고 가정하면, E-Step에서 <span><script type="math/tex">q^{(t)} = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})</script></span>이므로

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
U_t 
&= U(\boldsymbol{\theta}^{(t)}, q^{(t)}) \\
&= \ln \mathcal{L} (\boldsymbol{\theta}^{(t)}; \mathbf{X}) - \mathcal{D}\_{KL} \left( q^{(t)} (\mathbf{Z}) \parallel p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})) \right) \\
&= \ln \mathcal{L} (\boldsymbol{\theta}^{(t)}; \mathbf{X})
\end{aligned}
</script></div>

이고, 따라서 현재까지의 상황은 아래와 같다. 


<div class="math"><script type="math/tex; mode=display">
\ln \mathcal{L}(\hat{\boldsymbol{\theta}} ; \mathbf{X}) \ge \ln \mathcal{L}(\boldsymbol{\theta}^{(t)} ; \mathbf{X}) = U_{t}
</script></div>


이제 <span><script type="math/tex">U_{t+1}</script></span>을 구해보자. M-Step과 E-Step을 차례로 적용하면, 

* **M-Step** : <span><script type="math/tex">\displaystyle U(\boldsymbol{\theta}^{(t+1)}, q^{(t)}) = \max_\boldsymbol{\theta} U(\boldsymbol{\theta}, q^{(t)}) \ge U(\boldsymbol{\theta}^{(t)}, q^{(t)})</script></span>

* **E-Step** : <span><script type="math/tex">\displaystyle U_{t+1} = U(\boldsymbol{\theta}^{(t+1)}, q^{(t+1)}) = \max_q U(\boldsymbol{\theta}^{(t+1)}, q) \ge U(\boldsymbol{\theta}^{(t+1)}, q^{(t)})</script></span>

이므로, 

<div class="math"><script type="math/tex; mode=display">
\ln \mathcal{L}(\boldsymbol{\theta}^{(t+1)}; \mathbf{X}) = U_{t+1} 
\ge U(\boldsymbol{\theta}^{(t+1)}, q^{(t)})
\ge U(\boldsymbol{\theta}^{(t)}, q^{(t)})
= U_t 
</script></div>

따라서 우리는 다음을 알 수 있다. 

<div class="math"><script type="math/tex; mode=display">
\begin{matrix}
\ln \mathcal{L}(\hat{\boldsymbol{\theta}} ; \mathbf{X}) & 
\begin{matrix}
\ge & \cdots & \ge
\end{matrix} & \ln \mathcal{L}(\boldsymbol{\theta}^{(t+1)}; \mathbf{X}) & \ge & \ln \mathcal{L}(\boldsymbol{\theta}^{(t)}; \mathbf{X}) & \begin{matrix}
\ge & \cdots & \ge
\end{matrix} & \ln \mathcal{L}(\boldsymbol{\theta}^{(0)}; \mathbf{X}) \\
 &  & 
 \begin{matrix}
 \big\| \\ 
 U_{t+1} \\ 
 \bigg\downarrow
\end{matrix} &  & 
\begin{matrix}
\big\| \\ 
U_t \\ 
\bigg\downarrow
\end{matrix} &  & 
\begin{matrix}
\big\| \\ 
U_0 \\ 
\bigg\downarrow
\end{matrix} \\
\hat{\boldsymbol{\theta}} & \xleftarrow{\text{approach}} & \boldsymbol{\theta}^{(t+1)} & \xleftarrow{\phantom{\text{approach}}} & \boldsymbol{\theta}^{(t)} & \xleftarrow{\phantom{\text{approach}}} & \boldsymbol{\theta}^{(0)} \\
\end{matrix}
</script></div>

<br/>

## 모형의 한계

* EM 알고리즘은 모수의 초기값 <span><script type="math/tex">\boldsymbol{\theta}^{(0)}</script></span> 근처에서 지역해(Local solution)을 찾는 알고리즘이다. 즉, 전역해(Global solution)을 보장하지는 않고, <span><script type="math/tex">\boldsymbol{\theta}^{(t)}</script></span>가 반드시 <span><script type="math/tex">\underset{\boldsymbol{\theta}}{\arg \max} \ln \mathcal{L} (\boldsymbol{\theta}; \mathbf{X})</script></span>에 수렴한다고 볼 수 없다. 따라서 모수의 초기값을 터무니없이 잡으면 안된다. 
* 어떤 문서에서는, EM 알고리즘을 수행하기 전에 K-means를 우선 돌려서 적절한 초기값을 설정한다고 되어있다. (이 부분은 나중에 다시 정리하자)

