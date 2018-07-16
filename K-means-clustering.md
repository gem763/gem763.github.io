---


---

* TOC
{:toc}

## 개요

<span><script type="math/tex">n</script></span>개의 입력데이터 <span><script type="math/tex">\mathbf{X} = (\mathbf{x}_1 \cdots \mathbf{x}_n)</script></span>, <span><script type="math/tex">\mathbf{x}_i \in \mathbb{R}^d</script></span> 를 <span><script type="math/tex">k</script></span>개의 클러스터 <span><script type="math/tex">\mathbf{S} = \{ S_1, \cdots, S_k \}</script></span>에 배정하려고 한다. 이때 클러스터의 갯수 <span><script type="math/tex">k</script></span> (<span><script type="math/tex">\le n</script></span>)는 미리 정해져 있다고 가정한다. 

이 문제를 모델링 해보자. 우선 임의의 클러스터 <span><script type="math/tex">\mathbf{S}</script></span>에 대해서, 클러스터 <span><script type="math/tex">j</script></span>의 왜곡도 (Distortion)[^distortion] <span><script type="math/tex">\mathcal{D}_j</script></span>를 정의하고, 

[^distortion]: 비용함수(Cost function) 이라고도 한다. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{D}_j \equiv \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \boldsymbol{\mu}_j \Vert^2
</script></div>

총 왜곡도 (Total distortion) [^total_distortion] <span><script type="math/tex">\mathcal{J}_\mathbf{S}</script></span>를 다음과 같이 설정하자. 

[^total_distortion]: WCSS(Within-Cluster Sum-of-Squares) 또는 Inertia 라고 부르기도 한다. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{J}_\mathbf{S} = \sum_{j=1}^k \mathcal{D}_j = \sum^k_{j=1} \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \boldsymbol{\mu}_j \Vert^2
</script></div>


여기서 <span><script type="math/tex">\boldsymbol{\mu} = \{ \boldsymbol{\mu}_1, \cdots, \boldsymbol{\mu}_k \}</script></span>는 각 클러스터의 **Centroid**(중심점)을 의미한다. 즉 총 왜곡도는 각 클러스터에 속한 데이터들이 해당 Centroid로부터 얼마나 떨어져 있는지를 나타내는 지표이다. 


[**K-means clustering**](https://en.wikipedia.org/wiki/K-means_clustering)은 총 왜곡도를 최소화하는 클러스터 <span><script type="math/tex">\hat{\mathbf{S}}</script></span>를 찾는 최적화 알고리즘이다. 즉, 

<div class="math"><script type="math/tex; mode=display">
\hat{\mathbf{S}} = \underset{\mathbf{S}}{\arg \min} ~\mathcal{J}_\mathbf{S} = \underset{\mathbf{S}}{\arg \min} \sum^k_{j=1} \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \boldsymbol{\mu}_j \Vert^2
</script></div>

이 문제는 대수적으로 명쾌하게 풀리지 않는다. 위의 총 왜곡도를 최소화하기 위해서는 우선 Centroid <span><script type="math/tex">\boldsymbol{\mu}</script></span>에 대한 정보를 알고 있어야 하는데, 이 값들은 클러스터 <span><script type="math/tex">\mathbf{S}</script></span>가 정해져 있어야 알 수 있는 값들이기 때문이다. 따라서 K-means clustering은 수치적인 접근방법을 쓴다. **반복적인(iterative)** 절차를 통해 <span><script type="math/tex">\mathbf{S}</script></span>와 <span><script type="math/tex">\boldsymbol{\mu}</script></span>를 번갈아가며 추정하는 과정에서 최적 클러스터 <span><script type="math/tex">\hat{\mathbf{S}}</script></span>로 수렴해 나간다. K-means clustering은 데이터에 대한 사전정보(즉 레이블)가 없는 상태로 데이터 간의 특정 패턴을 추정한다는 측면에서, 머신러닝의 [**비지도 학습** (Unsupervised learning)](https://en.wikipedia.org/wiki/Unsupervised_learning)에 해당한다. 


<br/>

## 알고리즘

K-means clustering 문제의 수치적인 해결을 위해, 위에서 정의된 왜곡도 <span><script type="math/tex">\mathcal{D}_j</script></span>를 다음과 같이 풀어쓰자. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{D}_j = \sum_{\mathbf{x} \in S_j} \Vert \mathbf{x} - \boldsymbol{\mu}_j \Vert^2 = \sum^n_{i=1} r_{ij} \Vert \mathbf{x}_i - \boldsymbol{\mu}_j \Vert^2
</script></div>


여기서 <span><script type="math/tex">\mathbf{r} = \{ r_{ij} \}</script></span>는 각 데이터가 어떤 클러스터에 속해있는 지를 나타내주는 이산변수(Discrete variable) <span><script type="math/tex">r_{ij}</script></span>들의 집합이며, 다음과 같이 정의된다. 

<div class="math"><script type="math/tex; mode=display">
r_{ij} \equiv 
\begin{cases}
1 & \text{if} ~~\mathbf{x}_i \in S_j \\
0 & \text{otherwise}
\end{cases}
</script></div>

예를들어 <span><script type="math/tex">\mathbf{x}_1 \in S_2</script></span>, <span><script type="math/tex">\mathbf{x}_2 \in S_2</script></span>, <span><script type="math/tex">\mathbf{x}_3 \in S_1</script></span> 이라면, 

<div class="math"><script type="math/tex; mode=display">
\begin{pmatrix}
r_{11}=0 \\
r_{12}=1
\end{pmatrix}, ~
\begin{pmatrix}
r_{21}=0 \\
r_{22}=1
\end{pmatrix}, ~
\begin{pmatrix}
r_{31}=1 \\
r_{32}=0
\end{pmatrix}
</script></div>

이 된다. 앞으로 <span><script type="math/tex">\mathbf{r}</script></span>을 클러스터라고 부를 것이다. 이제 총 왜곡도는 다음과 같이 <span><script type="math/tex">\mathcal{J}_\mathbf{r}</script></span>로 변경되며, 원래의 클러스터링 문제는 결국 최적의 클러스터 <span><script type="math/tex">\mathbf{r}</script></span>을 찾는 문제로 귀결된다. 

<div class="math"><script type="math/tex; mode=display">
\min_{\mathbf{S}} \mathcal{J}_\mathbf{S} = \min_{\mathbf{r}} \mathcal{J}_\mathbf{r}
</script></div>

<div class="math"><script type="math/tex; mode=display">
\mathcal{J}_\mathbf{r} = \sum^n_{i=1} \sum^k_{j=1} r_{ij} \Vert \mathbf{x}_i - \boldsymbol{\mu}_j \Vert^2
</script></div>

K-means clustering은 Centroid <span><script type="math/tex">\boldsymbol{\mu}</script></span>의 추정값이 수렴할 때까지 다음의 Assignment 와 Update 를 반복하여 수행한다. 

* **Centroid 초기화**
    * Centroid의 추정값 <span><script type="math/tex">\hat{\boldsymbol{\mu}}</script></span>를 임의의 값 <span><script type="math/tex">\hat{\boldsymbol{\mu}}(0) = \{ \hat{\boldsymbol{\mu}}_1(0), \cdots, \hat{\boldsymbol{\mu}}_k(0) \}</script></span> 으로 초기화

    
* **Assignment**
    * 이전 단계에서 추정한 Centroid <span><script type="math/tex">\hat{\boldsymbol{\mu}}</script></span>을 이용하여 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}</script></span>을 추정하는 단계
    * **각각의 데이터에 가장 가까운 클러스터를 할당**
    
* **Update**
    * 이전 단계에서 추정한 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}</script></span>을 이용하여 Centroid <span><script type="math/tex">\hat{\boldsymbol{\mu}}</script></span>을 업데이트하는 단계
    * **<span><script type="math/tex">\hat{\boldsymbol{\mu}} =</script></span> 각각의 클러스터에 속한 데이터들의 샘플 평균값**
    
* **종료**
    * <span><script type="math/tex">\hat{\boldsymbol{\mu}}</script></span>가 수렴하면(즉 <span><script type="math/tex">\hat{\mathbf{r}}</script></span>의 변동이 없으면) 알고리즘을 종료

<br/>

다음은 iteration이 진행됨에 따라 클러스터가 어떻게 추정되는 지를 도식화한 그림이다. <span><script type="math/tex">d=2</script></span> (즉 2차원 데이터), <span><script type="math/tex">k=3</script></span> (3개의 클러스터로 구분) 에 대하여, 3개의 Centroid (<span><script type="math/tex">\color{red}{+}</script></span>, <span><script type="math/tex">\color{gold}{\times}</script></span>, <span><script type="math/tex">\color{blue}{\circ}</script></span>)가 어떻게 update 되는 지를 확인할 수 있다. 

<span style="color:red"><big><big>**＋**</big></big></span>, <span style="color:gold"><big><big>**×**</big></big></span>, <span style="color:blue"> <big><big>**○**</big></big></span>

<center><b>Convergence of K-means</b></center>
<center><img src="https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif" alt="K-means convergence.gif" width="400"></center>
<center><small>(출처: 위키피디아)</small></center>

<br/>

## 알고리즘의 유도

### Assignment

직전 단계에서 Centroid 추정값 <span><script type="math/tex">\hat{\boldsymbol{\mu}}(t) = \{ \hat{\boldsymbol{\mu}}_1 (t), \cdots, \hat{\boldsymbol{\mu}}_k (t) \}</script></span>가 주어졌다면, 총 왜곡도는 다음과 같이 분해된다. 

<div class="math"><script type="math/tex; mode=display">
\mathcal{J}_\mathbf{r} = \sum^k_{j=1} r_{1j} {\underbrace{\Vert \mathbf{x}_1 -\hat{\boldsymbol{\mu}}_j(t) \Vert}_{\text{minimize}}}^2 + \cdots + \sum^k_{j=1} r_{nj} {\underbrace{\Vert \mathbf{x}_n -\hat{\boldsymbol{\mu}}_j(t) \Vert}_{\text{minimize}}}^2
</script></div>

이 때 <span><script type="math/tex">r_{ij} \ge 0</script></span> 이고 <span><script type="math/tex">\Vert \cdot \Vert \ge 0</script></span> 이다. 그러므로 <span><script type="math/tex">\mathcal{J}_\mathbf{r}</script></span>을 <span><script type="math/tex">\mathbf{r}</script></span>에 대해서 최소화 한다는 얘기는, 각각의 데이터 <span><script type="math/tex">\mathbf{x}_i</script></span> 에 대해 <span><script type="math/tex">\Vert \mathbf{x}_i -\hat{\boldsymbol{\mu}}_j(t) \Vert</script></span> 를 최소화 한다는 말과 동일하고, 이는 **각 데이터를 가장 가까운 클러스터에 단순히 할당**한다는 의미가 된다. 따라서 주어진 Centroid <span><script type="math/tex">\hat{\boldsymbol{\mu}}(t)</script></span>에 대해, 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}(t) = \{ \hat{r}_{ij}(t) \}</script></span>는 다음과 같이 정해진다.  

<div class="math"><script type="math/tex; mode=display">
\hat{r}_{ij} (t) =
\begin{cases}
1 & \text{if} ~~j = \underset{\ell}{\arg \min} \Vert \mathbf{x}_i - \hat{\boldsymbol{\mu}}_\ell (t) \Vert \\
0 & \text{otherwise}
\end{cases}
</script></div>

<br/>

### Update

이전 단계에서 추정된 클러스터 <span><script type="math/tex">\hat{\mathbf{r}}(t) = \{ \hat{r}_{ij}(t) \}</script></span>가 주어진 상태에서, 총 왜곡도 <span><script type="math/tex">\mathcal{J}_\mathbf{r}</script></span>를 최소화 해보자. 

<span><script type="math/tex">\displaystyle \frac{\partial \mathcal{J}_\mathbf{r}}{\partial \boldsymbol{\mu}} = \left[ \frac{\partial \mathcal{J}_\mathbf{r}}{\partial \boldsymbol{\mu}_1} \cdots \frac{\partial \mathcal{J}_\mathbf{r}}{\partial \boldsymbol{\mu}_k} \right]^\mathsf{T} = 0</script></span> 에서, 

<div class="math"><script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial \mathcal{J}_\mathbf{r}}{\partial \boldsymbol{\mu}_j} \bigg|_{\hat{\boldsymbol{\mu}}_j}
&= \frac{\partial}{\partial \boldsymbol{\mu}_j} \left[
\sum^n_{i=1} \sum^k_{j=1} \hat{r}_{ij}(t) \Vert \mathbf{x}_i - \boldsymbol{\mu}_j \Vert^2 \right]_{\hat{\boldsymbol{\mu}}_j} \\
&= \frac{\partial}{\partial \boldsymbol{\mu}_j} \left[
\sum^n_{i=1} \hat{r}_{ij}(t) ( \mathbf{x}_i - \boldsymbol{\mu}_j )^T ( \mathbf{x}_i - \boldsymbol{\mu}_j ) \right] _{\hat{\boldsymbol{\mu}}_j}\\
&= -2 \sum^n_{i=1} \hat{r}_{ij}(t) ( \mathbf{x}_i - \hat{\boldsymbol{\mu}}_j ) \\
&= -2 \left[ \sum^n_{i=1} \hat{r}_{ij}(t) ~\mathbf{x}_i - \left( \sum^n_{i=1} \hat{r}_{ij}(t) \right) \hat{\boldsymbol{\mu}}_j \right] \\
&= 0
\end{aligned}
</script></div>

<div class="math"><script type="math/tex; mode=display">
\therefore \hat{\boldsymbol{\mu}}_j (t+1) = \frac{\sum^n_{i=1} \hat{r}_{ij}(t) ~\mathbf{x}_i}{\sum^n_{i=1} \hat{r}_{ij}(t)}
</script></div>

따라서 **Centroid <span><script type="math/tex">\hat{\boldsymbol{\mu}}(t+1) = \{ \hat{\boldsymbol{\mu}}_1(t+1), \cdots, \hat{\boldsymbol{\mu}}_k(t+1)  \}</script></span>는 각각의 클러스터에 속한 데이터들의 샘플 평균값**으로 계산된다. 이 알고리즘을 K-means 라고 부르는 이유이기도 하다. 


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

