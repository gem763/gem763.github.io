---


---

## 개요

<span><script type="math/tex">n</script></span>개의 입력데이터 <span><script type="math/tex">\mathbf{X} = (\mathbf{x}_1 \cdots \mathbf{x}_n)</script></span>를 <span><script type="math/tex">k</script></span> (<span><script type="math/tex">< n</script></span>)개의 클러스터 <span><script type="math/tex">\mathbf{S} = \{ S_1, \cdots, S_k \}</script></span>에 배정하려고 한다. 이때 클러스터의 갯수 <span><script type="math/tex">k</script></span>는 미리 정해져 있다고 가정한다. 

이 문제를 모델링 해보자. 우선 임의의 클러스터 <span><script type="math/tex">\mathbf{S}</script></span>에 대해서, 클러스터 <span><script type="math/tex">j</script></span>의 왜곡도(Distortion) <span><script type="math/tex">\mathcal{D}_j</script></span>를 다음과 같이 정의하고, 

<div class="math"><script type="math/tex; mode=display">
\mathcal{D}_j \equiv \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \mathbf{\mu}_j \Vert^2
</script></div>

목적함수(Objective function) <span><script type="math/tex">\mathcal{J}_\mathbf{S}</script></span>를 이 왜곡도들의 합으로 설정하자. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{J}_\mathbf{S} = \sum_{j=1}^k \mathcal{D}_j = \sum^k_{j=1} \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \mathbf{\mu}_j \Vert^2
</script></div>


여기서 <span><script type="math/tex">\mathbf{\mu} = \{ \mathbf{\mu}_1, \cdots, \mathbf{\mu}_k \}</script></span>는 각 클러스터의 **Centroid**(중심점)을 의미한다. 즉 왜곡도는 각 클러스터에 속한 데이터들이 해당 Centroid로부터 얼마나 떨어져 있는지를 나타내는 지표이다. 


**K-means clustering**은 총 왜곡도를 최소화하는 클러스터 <span><script type="math/tex">\mathbf{S}</script></span>를 찾는 최적화 알고리즘이다. 즉, 
<div class="math"><script type="math/tex; mode=display">
\hat{\mathbf{S}} = \underset{\mathbf{S}}{\arg \min} ~\mathcal{J}_\mathbf{S} = \underset{\mathbf{S}}{\arg \min} \sum^k_{j=1} \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \mathbf{\mu}_j \Vert^2
</script></div>

이 문제는 대수적으로 명쾌하게 풀리지 않는다. 위의 목적함수를 최소화하는 클러스터 <span><script type="math/tex">\mathbf{S}</script></span>를 추정하기 위해서는 우선 Centroid <span><script type="math/tex">\mathbf{\mu}</script></span>에 대한 정보를 알고 있어야 하는데, 이 값들은 클러스터 <span><script type="math/tex">\mathbf{S}</script></span>가 정해져 있어야 알 수 있는 값들이기 때문이다. 따라서 K-means clustering은 수치적인 접근방법을 쓴다. 반복적인(iterative) 절차를 통해 <span><script type="math/tex">\mathbf{S}</script></span>와 <span><script type="math/tex">\mathbf{\mu}</script></span>를 번갈아가며 추정하는 과정에서 최적 클러스터 <span><script type="math/tex">\hat{\mathbf{S}}</script></span>로 수렴해 나간다. 


<br/>

## 프로세스

K-means clustering 문제의 수치적인 접근을 위해, 위에서 정의된 왜곡도 <span><script type="math/tex">\mathcal{D}_j</script></span>를 다음과 같이 풀어쓴다. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{D}_j = \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \boldsymbol{\mu}_j \Vert^2 = \sum^n_{i=1} r_{ij} \Vert \mathbf{x}_i - \mathbf{\mu}_j \Vert^2
</script></div>


여기서 <span><script type="math/tex">\mathbf{r} = \{ r_{ij} \}</script></span>는 각 데이터가 어떤 클러스터에 속해있는 지를 나타내주는 이산변수(Discrete variable) <span><script type="math/tex">r_{ij}</script></span>들의 집합으로서, 다음과 같이 정의된다. 

<div class="math"><script type="math/tex; mode=display">
r_{ij} \equiv 
\begin{cases}
1 & \text{if} ~~\mathbf{x}_i \in S_j \\
0 & \text{otherwise}
\end{cases}
</script></div>

앞으로 <span><script type="math/tex">\mathbf{r}</script></span>을 클러스터라고 부를 것이다. 이제 알고리즘의 목적함수는 다음과 같이 <span><script type="math/tex">\mathcal{J}_\mathbf{r}</script></span>로 변경되며, 원래의 클러스터링 문제는 결국 최적의 클러스터 <span><script type="math/tex">\mathbf{r}</script></span>을 찾는 문제로 귀결된다. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{J}_\mathbf{r} = \sum^n_{i=1} \sum^k_{j=1} r_{ij} \Vert \mathbf{x}_i - \mathbf{\mu}_j \Vert^2
</script></div>

<div class="math"><script type="math/tex; mode=display">
\min_{\mathbf{S}} \mathcal{J}_\mathbf{S} = \min_{\mathbf{r}} \mathcal{J}_\mathbf{r}
</script></div>

K-means 클러스터링은 Centroid <span><script type="math/tex">\mathbf{\mu}</script></span>의 추정값이 수렴할 때까지 Assignment 와 Update 를 반복하여 수행한다. 

* **Centroid 초기화**
    * <span><script type="math/tex">\hat{\mathbf{\mu}}</script></span>를 임의의 값 <span><script type="math/tex">\hat{\mathbf{\mu}}(0) = \{ \hat{\mathbf{\mu}}_1(0), \cdots, \hat{\mathbf{\mu}}_k(0) \}</script></span> 으로 초기화
    * 보통 <span><script type="math/tex">n</script></span>개의 샘플데이터 중 <span><script type="math/tex">k</script></span>개를 임의로 선택
    
* **Assignment**
    * 이전 단계에서 추정한 Centroid <span><script type="math/tex">\hat{\mathbf{\mu}}</script></span>을 이용하여 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}</script></span>을 추정하는 단계
    * **각각의 샘플데이터에 가장 가까운 클러스터를 할당**
    
* **Update**
    * 이전 단계에서 추정한 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}</script></span>을 이용하여 Centroid <span><script type="math/tex">\hat{\mathbf{\mu}}</script></span>을 업데이트하는 단계
    * **<span><script type="math/tex">\hat{\mathbf{\mu}} =</script></span> 각각의 클러스터에 속한 데이터들의 샘플 평균값**
    
* **종료**
    * <span><script type="math/tex">\hat{\mathbf{\mu}}</script></span>가 수렴하면 알고리즘을 종료



<br/>

## 전개

### Assignment

직전 단계에서 Centroid 추정값 <span><script type="math/tex">\hat{\mathbf{\mu}}(t) = \{ \hat{\mathbf{\mu}}_1 (t), \cdots, \hat{\mathbf{\mu}}_k (t) \}</script></span>가 주어졌다면, 목적함수는 다음과 같이 분해할 수 있다. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{J}_\mathbf{r} = \underbrace{\sum^k_{j=1} r_{1j} \Vert \mathbf{x}_1 -\hat{\mathbf{\mu}}_j(t) \Vert^2}_{\text{minimize}} + \cdots + \underbrace{\sum^k_{j=1} r_{nj} \Vert \mathbf{x}_n -\hat{\mathbf{\mu}}_j(t) \Vert^2}_{\text{minimize}}
</script></div>

이 때 <span><script type="math/tex">r_{ij} \ge 0</script></span> 이고 <span><script type="math/tex">\Vert \cdot \Vert \ge 0</script></span> 이므로,  <span><script type="math/tex">\mathcal{J}_\mathbf{r}</script></span>을 <span><script type="math/tex">\mathbf{r}</script></span>에 대해서 최소화 한다는 것은, 각각의 데이터가 속한 클러스터와의 거리를 최소화 한다는 말과 동일하다고 할 수 있다. 따라서 **주어진 Centroid <span><script type="math/tex">\hat{\mathbf{\mu}}(t)</script></span>와의 거리가 가장 가까운 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}(t) = \{ \hat{r}_{ij}(t) \}</script></span>를 선택**하면 된다. 

<div class="math"><script type="math/tex; mode=display">
\hat{r}_{ij} (t) =
\begin{cases}
1 & \text{if} ~~j = \underset{\ell}{\arg \min} \Vert \mathbf{x}_i - \hat{\mathbf{\mu}}_\ell (t) \Vert \\
0 & \text{otherwise}
\end{cases}
</script></div>



### Update

직전 단계에서 추정된 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}(t) = \{ \hat{r}_{ij}(t) \}</script></span>가 주어진 상태에서, 목적함수 <span><script type="math/tex">\mathcal{J}_\mathbf{r}</script></span>에 LME를 적용해보자. 

<span><script type="math/tex">\displaystyle \frac{\partial \mathcal{J}_\mathbf{r}}{\partial \mathbf{\mu}} = \left[ \frac{\partial \mathcal{J}_\mathbf{r}}{\partial \mathbf{\mu}_1} \cdots \frac{\partial \mathcal{J}_\mathbf{r}}{\partial \mathbf{\mu}_k} \right]^T = 0</script></span> 에서, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
0 
&= \frac{\partial}{\partial \mathbf{\mu}_j} \left[
\sum^n_{i=1} \sum^k_{j=1} \hat{r}_{ij}(t) \Vert \mathbf{x}_i - \mathbf{\mu}_j \Vert^2 \right] \\
&= \frac{\partial}{\partial \mathbf{\mu}_j} \left[
\sum^n_{i=1} \hat{r}_{ij}(t) ( \mathbf{x}_i - \mathbf{\mu}_j )^T ( \mathbf{x}_i - \mathbf{\mu}_j ) \right] \\
&= -2 \sum^n_{i=1} \hat{r}_{ij}(t) ( \mathbf{x}_i - \mathbf{\mu}_j ) \\
&= -2 \left[ \sum^n_{i=1} \hat{r}_{ij}(t) ~\mathbf{x}_i - \left( \sum^n_{i=1} \hat{r}_{ij}(t) \right) \mathbf{\mu}_j \right]
\end{aligned}
</script></div>

<div class="math"><script type="math/tex; mode=display">
\therefore \hat{\mathbf{\mu}}_j (t+1) = \frac{\displaystyle \sum^n_{i=1} \hat{r}_{ij}(t) ~\mathbf{x}_i}{\displaystyle \sum^n_{i=1} \hat{r}_{ij}(t)}
</script></div>

따라서 **Centroid <span><script type="math/tex">\hat{\mathbf{\mu}}(t+1) = \{ \hat{\mathbf{\mu}}_1(t+1), \cdots, \hat{\mathbf{\mu}}_k(t+1)  \}</script></span>는 각각의 클러스터에 속한 데이터들의 샘플 평균값**으로 계산된다. 이 알고리즘을 K-means 라고 부르는 이유이기도 하다. 


<br/>

## 한계
1. k의 갯수
2. centroid의 초기값
3. 유클리드 거리의 한계
4. Hard cluster

GMM은 3,4를 해결


<br/>

## KNN과의 비교
KNN(K-Nearest Neighbors, K-최근접 알고리즘)은 우리가 판단하고자 하는 점 주위의 K개 점이 각각 어떤 클래스에 속하는지를 보고, 해당 점의 클래스를 판단하는 알고리즘이며, Supervised learning 에 속한다. 

