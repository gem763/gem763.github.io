---
layout: post
title: Elastic Asset Allocation
tags: [Asset allocation]
categories: [Asset allocation]
excerpt_separator: <!--more-->

---

Keller 교수의 Elastic Asset Allocation 전략논문을 리뷰하고, 나름의 실험결과를 기록한다. 

<!--more-->

* TOC
{:toc}


## 개요
이 포스트는 투자논문 [A Century of Generalized Momentum: From Flexible Asset Allocation (FAA) to Elastic Asset Allocation (EAA)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2543979) 에 대한 리뷰와 추가적인 실험결과를 기록하였다. 이 논문은 Wouter J. Keller (네덜란드 암스테르담 대학교수)와 Adam Butler ([ReSolve 자산운용](https://investresolve.com) 대표)가 공동 집필하였으며 2014년에 발표하였다. 

논문의 내용은 전형적인 **모멘텀 전략**의 연장선에 있다. 즉 (사전에 정의된) 모멘텀 지표가 큰 자산에 투자하자는 것이 결론이다. 유용한 모멘텀 지표의 개발은 오랫동안 수많은 논문들의 연구주제가 되어왔다. 예를들어, [Antonacci의 듀얼모멘텀 전략](https://books.google.co.kr/books/about/Dual_Momentum_Investing_An_Innovative_St.html?id=PVGoBAAAQBAJ&source=kp_cover&redir_esc=y)은 가격 모멘텀을 Time-series와 Cross-sectional로 구분하여 투자의 수익성과 안정성을 동시에 강화했다. 이 논문에서는 **독자적인 모멘텀 지표의 소개**와 더불어 **포트폴리오 비중결정의 논리**를 동시에 제공하였다. 저자는 이를 Generalized momentum 이라고 이름을 붙였는데, 기존의 다른 모멘텀 지표들 보다 유연한 Framework을 제공한다는 측면에서 제법 적절한 명칭이라고 생각된다. 



<br/>

## 포트폴리오 결정 프로세스
논문에서는 다음의 프로세스를 통해 포트폴리오 비중을 결정한다. 
1. **Generalized momentum**: 좀더 일반화된 모멘텀 지표를 정의하고, 자산별로 스코어링
2. **Asset selection**: 모멘텀 지표가 큰 자산들을 선택
3. **Crash protection**: 현금관리(Cash management) 전략

<br/>

### Generalized momentum
우선 각 자산 <span><script type="math/tex">i</script></span>에 대해서 다음의 세 가지를 정의한다. 

* 자산 <span><script type="math/tex">i</script></span>의 **가격모멘텀** <span><script type="math/tex">\mathbf{R}_i</script></span>: 가격 모멘텀을 정의하는 방식은 여러가지가 있다. 가장 흔한 방식으로는, 해당 자산가격의 12개월[^less12m] 수익률을 쓴다. 논문에서는 **1, 3, 6, 12개월**에 대하여, **T-bill 대비 초과수익률의 평균값**을 썼다[^multi_mix]. 가격 모멘텀이 큰 자산을 선호한다. 

[^less12m]: 다른 논문들에서의 실험에 의하면, 12개월 이내의 수익률과 미래의 수익률은 (+)의 상관관계가 있다(즉 예측력이 있다)고 알려져있다. 반면 12개월 이상의 긴 수익률은 Mean-reverting 가능성이 높다고 한다. [Asness의 논문(2012)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2174501) 참고. 

[^multi_mix]: 여러 구간의 수익률을 조합하여 가격모멘텀 지표로 활용하는 것은 꾸준히 시도되어 왔다. [Faber의 논문(2010)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1585517)과 [Hurst의 논문(2012)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026)을 참고. 

* 자산 <span><script type="math/tex">i</script></span>와 **동일가중 유니버스 지수와의 상관계수** <span><script type="math/tex">\mathbf{C}_i</script></span>: 투자 유니버스 구성종목들을 동일가중하여 생성된 포트폴리오와의 12개월 상관계수를 측정하였다. 상관계수가 낮은 자산을 선호한다. 


* 자산 <span><script type="math/tex">i</script></span>의 **[변동성](https://gem763.github.io/investment%20base/%ED%88%AC%EC%9E%90%EC%84%B1%EA%B3%BC%EC%9D%98-%EC%B8%A1%EC%A0%95.html#%EB%B3%80%EB%8F%99%EC%84%B1)** <span><script type="math/tex">\mathbf{V}_i</script></span>: 12개월 수익률의 표준편차로 측정한다. 변동성이 낮은 자산을 선호한다. 

<br/>

이제 자산 <span><script type="math/tex">i</script></span>의 모멘텀 스코어 <span><script type="math/tex">z_i</script></span>를 다음과 같이 정의하자. 논문에서는 이를 **Generalized momentum score** 라고 부른다. 

<div class="math"><script type="math/tex; mode=display">
z_i \equiv 
\begin{cases}
\mathbf{R}_i^\alpha (1-\mathbf{C}_i)^\beta  \mathbf{V}_i^{-\gamma} & \text{if~~} \mathbf{R}_i \gt 0 \\[8pt]
0 & \text{otherwise}
\end{cases}
</script></div>

여기서 <span><script type="math/tex">\alpha, \beta, \gamma \in \mathbb{R}_{\ge 0}</script></span> 는 각각 <span><script type="math/tex">\mathbf{R}_i</script></span>, <span><script type="math/tex">\mathbf{C}_i</script></span>, <span><script type="math/tex">\mathbf{V}_i</script></span> 에 대한 **Elasticity**[^el] (투자전략의 이름인 Elastic Asset Allocation 은 여기서 따온 것으로 추측) 라고 하며, 나중에 최적화의 대상이 된다. <span><script type="math/tex">\mathbf{R}_i</script></span>, <span><script type="math/tex">1-\mathbf{C}_i</script></span>,  <span><script type="math/tex">\mathbf{V}_i</script></span> 모두 양의 실수이므로, 

[^el]: Elasticity라고 불리는 이유는, <span><script type="math/tex">\mathbf{R}_i</script></span>, <span><script type="math/tex">\mathbf{C}_i</script></span>, <span><script type="math/tex">\mathbf{V}_i</script></span> 값의 변화에 대해 <span><script type="math/tex">z_i</script></span>가 얼마나 탄력적으로 변동하는 지를 결정하는 상수이기 때문인 것으로 보인다. 한편, 논문의 Notation과 다르므로 유의하기 바란다. 논문에서는 <span><script type="math/tex">\alpha, \beta, \gamma</script></span> 가 아니라 <span><script type="math/tex">wR, wC, wV</script></span> 라고 표기하였다. 취향의 차이임을 밝힌다. 

* 가격모멘텀 <span><script type="math/tex">\mathbf{R}_i</script></span>가 클 수록
* 상관계수 <span><script type="math/tex">\mathbf{C}_i</script></span>가 작을 수록
* 변동성 <span><script type="math/tex">\mathbf{V}_i</script></span>가 작을 수록

모멘텀 스코어 <span><script type="math/tex">z_i</script></span>가 커진다는 사실을 알 수 있다.  


<br/>


> <big>**상관계수 <span><script type="math/tex">\mathbf{C}_i</script></span>와 변동성 <span><script type="math/tex">\mathbf{V}_i</script></span>를 왜 반영했을까**</big>
> 
> 저자는 **포트폴리오의 변동성을 낮추고자** 하였다. <span><script type="math/tex">n</script></span>개 종목으로 구성되어 있는 포트폴리오의 변동성 <span><script type="math/tex">\mathbf{V}_p</script></span>는, 비중벡터 <span><script type="math/tex">\mathbf{w}=[w_i] \in \mathbb{R}^n</script></span> 및 공분산행렬 <span><script type="math/tex">\mathbf{\Sigma} = [\sigma_{ij}] \in \mathbf{R}^{n \times n}</script></span> 에 대하여 다음과 같이 나타낼 수 있다. 
> <div class="math"><script type="math/tex; mode=display">
>\begin{aligned}
>\mathbf{V}_p^2 
>&= \mathbf{w}^\mathsf{T} \mathbf{\Sigma} \mathbf{w} \\
>&= \sum_i w_i^2 \sigma_{ii} + 2 \sum_{j \ne k} w_j w_k \sigma_{jk} \\
>&= \sum_i w_i^2 \sigma_{i}^2 + 2 \sum_{j \ne k} w_j w_k \sigma_{j} \sigma_{k} \rho_{jk}
>\end{aligned} 
></script></div>
> 
> 따라서 다른 조건이 모두 동일하다고 할 때, 변동성 <span><script type="math/tex">\mathbf{V}_i</script></span> (= <span><script type="math/tex">\sigma_i</script></span>) 가 높은 자산 <span><script type="math/tex">i</script></span>의 비중 <span><script type="math/tex">w_i</script></span>를 줄이면 포트폴리오 변동성 <span><script type="math/tex">\mathbf{V}_p</script></span>이 작아질 것은 자명하다. 상관계수의 경우에는 약간 애매하다. 상관계수는 자산과 자산간의 일대일 관계이므로, 특정자산 <span><script type="math/tex">i</script></span>에 대해서 <span><script type="math/tex">n-1</script></span>개의 상관계수가 생성되고, 이들을 모두 고려하려면 모멘텀 스코어의 차원이 매우 커지게(즉 독립변수가 많아지게) 된다. 이 논문에서는 문제를 단순화하기 위해, 동일가중(즉 <span><script type="math/tex">w_i=1/n</script></span>) 유니버스 지수와의 상관계수 <span><script type="math/tex">\mathbf{C}_i</script></span> 하나로 통일한 것으로 보인다. 


<br/>

### Asset selection
<span><script type="math/tex">n</script></span>개의 자산 중, 위에서 정의한 모멘텀 스코어를 기준으로 상위 <span><script type="math/tex">m</script></span>개의 자산을 선택한다. 논문에서는 
<div class="math"><script type="math/tex; mode=display">
m = \min \left( 1+\sqrt{n}, ~n/2 \right)
</script></div>

라는 공식을 쓰는데, 논문에 그 이유는 나와있지 않았다. 아무래도 백테스트 과정에서 적절한 논리를 찾은 듯하다. 하지만 꼭 이렇게 할 필요는 없어 보이며, 전체 유니버스 종목 갯수 <span><script type="math/tex">n</script></span>의 30% 수준에서 선택하면 무리가 없을 것으로 생각된다. 

<br/>

### Crash protection
Crash protection은 현금관리 전략이다. 즉 전체 예산 중 얼마만큼을 현금에 할당할 지를 결정한다. Crash protection 비율 <span><script type="math/tex">w_{cp}</script></span>은 전체 가격모멘텀 집합 <span><script type="math/tex">\{ \mathbf{R}_i \}</script></span> 의 원소 중, **(+)가 아닌 값의 비율**을 의미한다.  

<div class="math"><script type="math/tex; mode=display">
w_{cp} \equiv \frac{\mathbf{n}(\{ i \mid \mathbf{R}_i \le 0 \})}{n}
</script></div>

여기서 <span><script type="math/tex">\mathbf{n}(\cdot)</script></span>는 집합의 원소 갯수를 세어주는 연산자이다. 참고로 논문에서는, 현금의 proxy로 **미국채 10년물**(UST 10Y)을 쓰고 있다. 


<br/>

### Final portfolio
모멘텀 스코어 <span><script type="math/tex">z_i</script></span>가 내림차순으로 정열되어 있다고 가정하면, 1 부터 <span><script type="math/tex">m</script></span>까지 선택되어 있을 것이다. 따라서 최종 포트폴리오는 다음과 같이 정해진다. 

<div class="math"><script type="math/tex; mode=display">
w_i = \frac{z_i}{\sum_{j=1}^m z_j} (1 - w_{cp})
</script></div>


<br/>

> <big>**Special cases**</big>
> 
> 저자는 논문에 나오는 모멘텀 스코어를 Generalized momentum 이라고 불렀는데, 이유가 있다. 기존의 다른 모멘텀 스코어를 포함하는 개념으로 이해할 수 있기 때문이다. 예를 들면, (여기에서는 <span><script type="math/tex">w_{cp}=0</script></span> 라고 하자)
> 
> * **Dual momentum** (<span><script type="math/tex">\alpha=\beta=\gamma=0</script></span>)
> <div class="math"><script type="math/tex; mode=display">w_i = \frac{1}{m}</script></div>
> 
> * **동일가중** (<span><script type="math/tex">\mathbf{R}_i \gt 0</script></span> 조건 제거, <span><script type="math/tex">\alpha=\beta=\gamma=0</script></span>)
> <div class="math"><script type="math/tex; mode=display">w_i = \frac{1}{n}</script></div>
> 
> * **Naive Risk-parity** (<span><script type="math/tex">\gamma=1</script></span>, <span><script type="math/tex">\alpha=\beta=0</script></span>)
> <div class="math"><script type="math/tex; mode=display">w_i = \frac{1}{\mathbf{V}_i} \frac{1}{\sum_{j=1}^m (1/\mathbf{V}_j)}</script></div>
>  

<br/>



## 논문의 백테스트

논문의 백테스트는 다음의 순서로 진행되었다. 
1. **IS 테스트** (In-sample test): Elasticity 변수를 최적화하여 두 개의 주요 모델 도출
2. **OS 테스트** (Out-of-sample test): 도출된 모델의 검증

한 가지 특이할 만한 점은, 논문이 제시하는 전략이 다양한 자산군들에서도 유효한지를 검증하기 위해, 총 세가지의 유니버스에 대해 백테스트를 진행하였다는 점이다. 

* **Global multi-asset small univ** (<span><script type="math/tex">n</script></span>=7): S&P500, EAFE, EEM, US Tech, Japan Topix, UST 10Y, US HY
* **US-sector univ** (<span><script type="math/tex">n</script></span>=15): 10개의 US Equity sector + 5개의 US Bond sector (UST 10Y, UST 30Y, Muni, IG, HY)
* **Global multi-asset large univ** (<span><script type="math/tex">n</script></span>=38): Global multi-asset small univ + US-sector univ + 기타 (논문참조)


한편 IS 테스트를 통해, 각 자산별 모멘텀 스코어 <span><script type="math/tex">z_i</script></span>에서 변동성 <span><script type="math/tex">\mathbf{V}_i</script></span>의 영향이 생각보다 크지 않다는 사실을 발견했다는 내용이 논문에 나온다. 따라서 저자는, IS를 포함한 모든 백테스트에서 모멘텀 스코어를 다음의 형태로 변경하여 사용한다. 최적화의 대상은 이제 <span><script type="math/tex">\alpha, \beta</script></span>가 되었다. 

<div class="math"><script type="math/tex; mode=display">
z_i = 
\begin{cases}
\mathbf{R}_i^\alpha (1-\mathbf{C}_i)^\beta & \text{if~~} \mathbf{R}_i \gt 0 \\[8pt]
0 & \text{otherwise}
\end{cases}
</script></div>


### Calmar ratio
IS 테스트 및 OS 테스트에 들어가기 앞서, 우선 이 논문에서 주요 성과지표로 활용하고 있는 [**Calmar ratio**](https://en.wikipedia.org/wiki/Calmar_ratio)에 대해 소개할 필요가 있다. Calmar ratio <span><script type="math/tex">\mathbf{CR}_t</script></span> 는 위험조정수익률의 한 종류로서, **초과수익률과 MDD간의 비율**을 의미한다.  

<div class="math"><script type="math/tex; mode=display">
\mathbf{CR}_t \equiv \frac{\mathbf{CAGR}-t}{\mathbf{MDD}}
</script></div>

여기서 <span><script type="math/tex">t</script></span>는 목표수익률을 뜻한다. 이를테면 <span><script type="math/tex">\mathbf{CR}_5</script></span>는 목표수익률 5%를 초과하는 위험조정수익률이다. Calmar ratio는 Sharpe의 MDD 버전이라고 이해하면 쉽다. 논문의 저자는, **장기투자**하는 경우에는 **Tail-risk**가 매우 중요해지기 때문에, Sharpe보다는 Calmar ratio가 더 의미있는 지표라고 주장하였다. 

<br/>

### IS 테스트



Elasticity 변수인 <span><script type="math/tex">\alpha</script></span>, <span><script type="math/tex">\beta \in \mathbb{R}_{\ge 0}</script></span>를 최적화하는 과정이다. <span><script type="math/tex">\alpha</script></span>와 <span><script type="math/tex">\beta</script></span>를 일정간격(Grid)으로 나누어서 여러 (<span><script type="math/tex">\alpha, \beta</script></span>) 조합을 만든다. 각 (<span><script type="math/tex">\alpha, \beta</script></span>) 조합에 대하여 다음의 백테스트를 수행하였다. 

* 기간: 1914년 4월 - 1964년 3월 (50년간)
* 해당 (<span><script type="math/tex">\alpha, \beta</script></span>) 조합으로 매월말 포트폴리오 구성 (월간 리밸런싱)
* 50년간의 [CAGR](https://gem763.github.io/investment%20base/%ED%88%AC%EC%9E%90%EC%84%B1%EA%B3%BC%EC%9D%98-%EC%B8%A1%EC%A0%95.html#cagr)과 [MDD](https://gem763.github.io/investment%20base/%ED%88%AC%EC%9E%90%EC%84%B1%EA%B3%BC%EC%9D%98-%EC%B8%A1%EC%A0%95.html#mdd)를 Calmar Scatter[^calmar_scatter]에 표시


[^calmar_scatter]: MDD를 <span><script type="math/tex">x</script></span> 축에, CAGR을 <span><script type="math/tex">y</script></span> 축에 그려넣는다. Risk-Return profile과 같은 개념이라고 생각하면 된다. 

모든 (<span><script type="math/tex">\alpha, \beta</script></span>) 조합에 대하여, 다음의 Calmar scatter를 얻게 된다. 

<center><img src="https://gem763.github.io/assets/img/20180708/calmar_scatter.PNG" alt="cum_dm"/></center>

**저자는 방어적인 투자의 목표수익률을 5%, 공격적인 투자자의 목표수익률을 10%로 보았다**. 따라서 (Risk-Return profile에서 Efficient frontier를 그리는 것과 마찬가지로) Calmar ratio <span><script type="math/tex">\mathbf{CR}_5</script></span>, <span><script type="math/tex">\mathbf{CR}_{10}</script></span>를 극대화하는 접선(Calmar frontier)를 그렸다. 그리고 해당 접점의 (<span><script type="math/tex">\alpha, \beta</script></span>) 조합을 각각 Golden defensive model, Golden offensive model이라고 정의하였다. 


<center><img src="https://gem763.github.io/assets/img/20180708/calmar_frontier.PNG" alt="cum_dm" width=500/></center>

결과적으로 다음의 두 가지 모델을 도출하게 된다. 

* **Golden Defensive EAA model** (<span><script type="math/tex">\alpha = \beta = 0.5</script></span>)
<div class="math"><script type="math/tex; mode=display">
z_i^{d} = 
\sqrt{\mathbf{R}_i (1-\mathbf{C}_i)} ~~~(\text{if~~} \mathbf{R}_i \gt 0)
</script></div>

<br/>

* **Golden Offensive EAA model** (<span><script type="math/tex">\alpha=2</script></span>, <span><script type="math/tex">\beta=1</script></span>)
<div class="math"><script type="math/tex; mode=display">
z_i^{o} = 
\mathbf{R}_i^2 (1-\mathbf{C}_i) ~~~(\text{if~~} \mathbf{R}_i \gt 0)
</script></div>

<br/>

### OS 테스트

IS 테스트에서 도출한 두 개의 모델 각각의 Out-of-sample 성과가 어땠는지를 확인해본다. 테스트 구간은 1964년 4월부터 2014년 8월까지 총 50년간(IS 구간과 동일)이다. 

<center><b>Cumulative return</b></center>
<center><img src="https://gem763.github.io/assets/img/20180708/cum_golden.PNG" alt="cum_dm" width=500/></center>

<br/>
<center><b>Statistics</b></center>
<center><img src="https://gem763.github.io/assets/img/20180708/golden_table.PNG" alt="cum_dm" width=400/></center>

* **연수익률 13-15%** 수준으로 동일가중 및 미국주식시장 대비 양호한 성과를 기록하였다. 
* 변동성이 동일가중 포트폴리오에 비해 크게 개선되었다고 보기는 힘들었으나, 
* **MDD는 동일가중, 미국주식시장의 50% 이하로 축소**되었다. 
* 결과적으로, **위험조정수익률(Calmar 및 Sharpe) 측면에서 양호**한 성과를 보였다. 

<br/>

## 실전 백테스트
논문에서 제시한 두 개의 Golden EAA 전략, 즉 Defensive model과 Offensive model을 실제로 백테스트해보자. 비교용으로 활용하기 위해, Dual momentum 과 동일가중(EW: Equal-weighted) 전략도 포함하였다. 

* 유니버스(<span><script type="math/tex">n</script></span>=7): **SPY**(미국주식), **EFA**(선진국주식), **EEM**(신흥국주식), **DBC**(원자재), **VNQ**(미국 부동산), **HYG**(미국 High-yield), **IEF**(미국 중기채)
* 백테스트 기간: 2002.12.31 ~ 2018.03.31 (약 15년)
* Monthly rebalancing
* 매매비용 10bp
* Gross exposure 99% (매매비용 인출을 감안)
* Cash asset:  **IEF**(미국 중기채)

<br/>

### 매매규칙

1.  투자의사결정: 매월 마지막 영업일
    - **EAA**: 모멘텀 스코어 <span><script type="math/tex">z_i</script></span>에 따라 자산을 선택하고 비중결정한다. 
    - **Dual momentum**: 가격모멘텀 <span><script type="math/tex">\mathbf{R}_i</script></span>에 따라 자산을 선택하고, 동일가중한다. 
    - **EW**: 유니버스 내 전종목을 동일가중한다. 
    
2.  매매: 매월 첫번째 영업일
    -   전일 투자의사결정된 포트폴리오가 전월의 포트폴리오와 다른 경우에 한해, 당일 종가(Adjusted)로 매매한다.
    -   만약 어떤 종목이 시장에서 아직 거래되지 않는다면, 해당 종목의 기초지수를 이용하여 그 종목의 시장가격을 역으로 추정한다.

<br/>

### Dual momentum
<center><img src="https://gem763.github.io/assets/img/20180708/cum_dm.PNG" alt="cum_dm"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/cum_def_off.PNG" alt="cum_def_off"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/stats_def_off.PNG" alt="stats_def_off"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/max_weight_def_off.PNG" alt="max_weight_def_off"/></center>

<br/>


<center><img src="https://gem763.github.io/assets/img/20180708/heat_0_2.PNG" alt="heat_0_2"/></center>

<br/>

## 전략 비틀기

Elasticity가 음수여도 괜찮지 않을까? 
<span><script type="math/tex">\beta=1</script></span>, <span><script type="math/tex">\gamma=0</script></span>

<div class="math"><script type="math/tex; mode=display">
z_i \equiv 
\begin{cases}
\mathbf{R}_i^\alpha (1-\mathbf{C}_i) & \text{if~~} \mathbf{R}_i \gt 0 \\[8pt]
0 & \text{otherwise}
\end{cases}
</script></div>


각 자산별 모멘텀 스코어 <span><script type="math/tex">z_i</script></span>를 해당 자산의 기대수익률 <span><script type="math/tex">\mathbf{E}[R_i]</script></span> 로 가중평균한 값을 총 모멘텀 <span><script type="math/tex">\mathbf{M}</script></span> 이라고 정의하자. 이는 <span><script type="math/tex">\alpha</script></span>의 함수가 된다. 

<div class="math"><script type="math/tex; mode=display">
\mathbf{M} \equiv \mathbf{E}[R_1] z_1  + \cdots + \mathbf{E}[R_n] z_n = \mathbf{E}[R]^\mathsf{T} \mathbf{z} = \mathbf{M}(\alpha)
</script></div>

이 총 모멘텀 <span><script type="math/tex">\mathbf{M}</script></span>을 최대로 만드는 <span><script type="math/tex">\alpha</script></span> 값을 찾는다. 단 Elasticity <span><script type="math/tex">\alpha</script></span>가 너무 크거나 작으면 특정 자산의 비중이 과도하게 쏠리는 현상이 생길 수 있다. 따라서 어떤 실수 <span><script type="math/tex">\theta \gt 0</script></span> 에 대해, <span><script type="math/tex">\alpha</script></span>를 <span><script type="math/tex">\pm \theta</script></span> 내의 값으로 제한하였다. 

<div class="math"><script type="math/tex; mode=display">
\alpha^* = \underset{|\alpha| \le \theta}{\arg \max} \mathbf{E}[R]^\mathsf{T} \mathbf{z}
</script></div>

<center><img src="https://gem763.github.io/assets/img/20180708/elas.PNG" alt="elas"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/cum_optima_3.PNG" alt="cum_optima_3"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/max_weight_3.PNG" alt="max_weight_3"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/grid_1_6.PNG" alt="grid_1_6"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/compare_stats_1_6.PNG" alt="compare_stats_1_6"/></center>

<br/>

### Crash protection

<center><img src="https://gem763.github.io/assets/img/20180708/cum_optima_all.PNG" alt="cum_optima_all"/></center>

<center><img src="https://gem763.github.io/assets/img/20180708/stats_optima_all.PNG" alt="stats_optima_all"/></center>

