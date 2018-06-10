# Formulation of Tracking-error Control

### Key idea
* 포트폴리오와 BM간의 비중을 조절하여 Ex-ante Tracking error를 통제할 수 있다. 
* 과거 11개월간의 Historical Tracking error와 향후 1개월에 대한 Ex-ante Tracking error를 섞어서 생성한 최종 Tracking error가, 일정수준 이하가 되도록 관리한다. 



### Ex-ante Tracking error

$n$개의 종목에 대하여, $\omega_p$와 $\phi$를 포트폴리오와 BM의 각 종목별 비중이라고 하자. 

$$\omega_p = 
\begin{bmatrix}
\omega_{1} \\
\vdots \\
\omega_{n}
\end{bmatrix} \in \mathbb{R}^n,  ~~\phi = \begin{bmatrix}
\phi_{1} \\
\vdots \\
\phi_{n}
\end{bmatrix} \in \mathbb{R}^n$$

예를들어 포트폴리오가 주식:채권:대체투자=40:30:30 이고, BM이 주식:채권=50:50 인 경우라면, $\omega_p$와 $\phi$는 다음과 같이 설정된다. 

$$\omega_p = 
\begin{bmatrix}
0.4 \\
0.3 \\
0.3
\end{bmatrix},  ~~\phi = \begin{bmatrix}
0.5 \\
0.5 \\
0.0
\end{bmatrix}$$


포트폴리오에 $\eta \in \mathbb{R}^+$만큼 투자하고, 나머지 비중 $1-\eta$를 BM에 투자한다고 하면, **최종 포트폴리오** $\omega$는 다음과 같이 결정된다. 

$$
\omega = \eta \omega_p + (1-\eta) \phi
$$

이를 이용하면, 최종 포트폴리오와 BM과의 차이에 대한 Variance는, 

$$
\begin{aligned}
\mathbf{Var} [(\omega-\phi)^T \mathbf{R}] 
&= \mathbf{Var}[\eta (\omega_p - \phi)^T \mathbf{R}] \\
&= \eta^2 (\omega_p - \phi)^T \mathbf{\Sigma} (\omega_p - \phi)
\end{aligned}
$$

여기서 $\mathbf{R} \in \mathbb{R}^n$은 각 종목별 수익률을 나타내는 확률변수이고, $\mathbf{\Sigma}$는 Covariance matrix를 의미한다. 
$$
\mathbf{\Sigma} = \mathbf{Var}(\mathbf{R}) = \mathbf{Cov}(\mathbf{R},\mathbf{R}) = 
\begin{bmatrix}
\sigma_{11} & \cdots & \sigma_{1n} \\
\vdots & \ddots & \vdots \\
\sigma_{n1} & \cdots & \sigma_{nn}
\end{bmatrix} \in \mathbb{R}^{n \times n}
$$

따라서 Ex-ante Tracking error $\mathbf{TE}_f$는 $\eta$의 함수가 되며, 다음과 같이 계산된다. 

$$
\begin{aligned}
\mathbf{TE}_f = \mathbf{TE}_f (\eta)
&= \sqrt{\mathbf{Var} [(\omega-\phi)^T \mathbf{R}]}  \\
&= \eta \sqrt{(\omega_p - \phi)^T \mathbf{\Sigma} (\omega_p - \phi)} \\
&= \eta~ \mathbf{TE}_f (1)
\end{aligned}
$$

이때 $\mathbf{TE}_f(1)$은, 포트폴리오에 100% 투자했을 때(BM에 투자하지 않는 경우, 즉 $\eta=1$)의 Ex-ante Tracking error를 뜻한다. 

### Algebraic solution
편의를 위해 몇 가지 변수들을 미리 정리해 놓자. 
* 연간 총 영업일수 $d$ ($\approx$ 250)
* Ex-ante로 예측하는 일수 $d_f$ ($\approx$ 20)
* **Total** Tracking error $\mathbf{TE}$
* **Historical** Tracking error $\mathbf{TE}_h$
* **Ex-ante** Tracking error $\mathbf{TE}_f$
* **Target** Tracking error $\mathbf{T}$
* Safety buffer $\theta$ ($0 \le \theta \le 1$)

이제 Ex-ante Tracking error가 $d_f$일 만큼 반영된 Total Tracking error $\mathbf{TE}$는 다음과 같이 표현된다.  
$$
\begin{aligned}
\mathbf{TE}^2 
&= \frac{d-d_f}{d} \mathbf{TE}_h^2 + \frac{d_f}{d} \mathbf{TE}_f^2 \\
&= \frac{d-d_f}{d} \mathbf{TE}_h^2 + \eta^2 \frac{d_f}{d} \mathbf{TE}_f^2(1)
\end{aligned}
$$

따라서 우리의 목적은, 다음의 두 가지 제약조건(Constraints)을 만족하는 **포트폴리오 비중 $\eta$의 최대값을 찾는 것**이 된다. 
$$
\mathbf{TE} \le \theta \mathbf{T}
$$

$$
0 \le \eta \le 1
$$

이 문제는 본질적으로 Optimization problem에 속하지만, 제약조건이 실수 $\eta$에 대한 단순 2차함수로 되어 있어서, 다음과 같이 Algebraic solution $\eta^*$를 쉽게 구할 수 있다. 

$$
\mathbf{TE}^2
= \frac{d-d_f}{d} \mathbf{TE}_h^2 + \eta^2 \frac{d_f}{d} \mathbf{TE}_f^2(1) 
\le (\theta \mathbf{T})^2
$$

$$
\begin{aligned}
\eta^2 d_f \mathbf{TE}^2_f(1) 
&\le d (\theta \mathbf{T})^2 - (d - d_f) \mathbf{TE}_h^2 \\
\eta^2 &\le \frac{d (\theta \mathbf{T})^2 - (d - d_f) \mathbf{TE}_h^2}{d_f \mathbf{TE}^2_f(1)}
\end{aligned}
$$

$$
\therefore \eta^* = \min \left(1, ~~~ \sqrt{\frac{d (\theta \mathbf{T})^2 - (d - d_f) \mathbf{TE}_h^2}{d_f \mathbf{TE}^2_f(1)}} \right)
$$
