---
layout: post
title: Global Fixed-income Allocation Model
tags: [Asset allocation]
categories: [Asset allocation]
excerpt_separator: <!--more-->

---

미국에 상장되어 있는 글로벌 채권 ETF를 활용하여 **Global Fixed-income Allocation Model**을 설계하였다. 개리 안토나치의 **Dual momentum** 전략을 기본 골격으로 하였으나, **Reinforcement**라는 또다른 전략을 추가하여, 투자자 입장에서 좀더 실용적인 Model을 만들고자 하였다. 유니버스 내에서 5개 이상의 종목을 동적으로 선택하는 것이 Sharpe 측면에서 가장 적절하였다. 

<center><b>Cumulative return</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/cum_base.png" alt="cum_base"/></center>

<!--more-->

기대효과: 
* **중장기: CAGR 7-9%, 연변동성 7-8%, Sharpe 1.1-1.2**
* **단기: CAGR 7-8%, 연변동성 7%, Sharpe 1.1-1.2**
* **MDD 7-15%**
* **1년 손실가능성 1-3%**

<br/>

* TOC 
{:toc}

<br/>

## Background: Dual momentum strategy

Dual momentum은 게리 안토나치(Gary Antonacci)가 2012년 고안한 투자전략의 이름이다. 안토나치는 모멘텀 전략을 다음과 같이 크게 두 가지로 구분하였다. 

* **Absolute momentum**: 
상승장에 투자하고 하락장에 현금(또는 안전자산)을 보유하는 전략이다. 그는 해당 자산의 과거 12개월 수익률이 예금금리를 상회하는 경우를 상승장, 그 반대의 경우를 하락장으로 정의(물론 정의하기 나름이며, 이 포스트에서는 약간 다른 방법을 썼다)했다. 추세추종 전략 또는 시계열(Time-series) 모멘텀 전략이라고도 한다. 

* **Relative momentum**: 
여러 자산들간의 모멘텀 지표를 서로 비교하여, 상대적으로 모멘텀이 큰 자산에 집중투자하는 방법이다. 안토나치는 Absolute momentum과 마찬가지로, 자산들간의 12개월 과거수익률을 서로 비교했다. 횡단면(Cross-sectional) 모멘텀 전략이라고도 부른다. 

안토나치의 실험에 따르면, Absolute momentum 전략은 Market shock 등의 불확실성에 대해 탁월한 방어력이 있다. 반면 Relative momentum 전략은 변동성에 매우 취약하긴 하지만, 기대수익률은 더 높다. 

<center><big><b>Absolute + Relative = Dual momentum</b></big></center>

안토나치의 아이디어는 **위의 두 가지 모멘텀을 동시에 적용**하는 것이다. 즉 여러 자산들 중 상대적으로 모멤텀이 큰 자산에 투자하되, 해당 자산의 상승 모멘텀이 약해지면 바로 현금(또는 안전자산)으로 갈아탐으로써, 공격력과 방어력을 동시에 취한다. 

Dual momentum 전략에 대한 자세한 설명은 다음을 참고. 
* Paper: [Risk Premia Havesting Through Dual Momentum](http://www.optimalmomentum.com/RiskPremiaHarvesting.pdf) (Portfolio Management Consultants, 2012)
* Book: [Dual Momentum](https://books.google.co.kr/books/about/Dual_Momentum_Investing_An_Innovative_St.html?id=PVGoBAAAQBAJ&source=kp_cover&redir_esc=y) (McGraw-Hill, 2014) 
* Website: https://www.optimalmomentum.com

<br/>

> <big>**Herding effect**</big>
> 
>모멘텀 전략이 작동하는 원인에는 논의가 분분한데, 가장 잘 알려진 것이 Behavioral finance의 대표적인 Anomaly인 **Herding effect**이다. 
><center><i>Herding effect</i></center>
><center><img src="https://gem763.github.io/assets/img/20180505/herding.jpg" alt="herding"/></center>
>
>이에 따르면, 과거 일정기간 동안의 Winner가 앞으로도 한동안 Winner가 될 가능성이 높다(Relative momentum). 비슷한 논리로, 잘 나가던 금융자산의 가격이 갑자기 고꾸라지는 경우는 드물다. 상승하던 금융자산의 현재가치에 급속한 변화가 생겼다고 하더라도, Herding effect는 해당 자산의 가치하락에 마찰적 요인(friction)으로 작용하기 때문이다. 이는 하락장 초기에 현금성 자산으로 갈아탈 수 있는 기회(Absolute momentum)를 제공한다. 참고로 Herding effect는, 개별종목보다 **자산군(Asset class) 레벨에서 훨씬 효과적으로 작동**한다고 알려져 있다. 

<br/>

## Objectives
안토나치의 논문에는 Dual momentum 전략을 Fixed-income universe에 적용한 사례가 기술되어 있으며, 성과도 상당히 양호하다. 다음은 그의 논문에 실려있는 결과 중 일부이다. 

<center><img src="https://gem763.github.io/assets/img/20180505/stats_antonacci.PNG" alt="stats_antonacci"/></center>

<center><img src="https://gem763.github.io/assets/img/20180505/cum_antonacci.PNG" alt="cum_antonacci"/></center>


결과를 간단히 요약하자면, 
* **US High yield**(Bank of America Merrill Lynch U.S. Cash Pay High Yield Index)와 **US Credit bond**(Barclays Capital Aggregate Bond Index), 이 두 개의 인덱스에 Dual momentum 적용
* **US T-bill**(Bank of America Merrill Lynch 3-Month Treasury bill Index)을 안전자산으로 사용
* **CAGR 10.5%** (US High yield와 유사)
* **연변동성 4.7%** (US High yield의 약 50%)
* **Sharpe 1.0** (US High yield의 약 2배)
* **MDD 8.2%** (US High yield의 약 25%)


하지만 이를 실전에 바로 적용하기에는 몇 가지 문제점이 있어 보인다. 그 이유는, 

* **투자 유니버스가 매우 제한적이다**: 
US High yield와 US Credit bond로만 한정하여 실험하였기 때문에, 투자자가 이를 이용하여 Asset allocation을 수행하기에는 투자종목의 수가 너무 적은 측면이 있다. 한편 안토나치는, 투자 유니버스를 확대하고 자산군의 Segmentation을 좀더 세분화하는 것이 반드시 좋은 것만은 아니며, 분산효과를 저해하는 부정적인 효과가 오히려 더 크다고 주장하였다. 

* **Backtest가 다소 비현실적이다**: 
실제 매매가능한 ETF나 주식 또는 채권이 아닌, 인덱스를 사용하여 Backtest 하였다. 매매가능한 자산군들의 Time horizon이 지극히 짧기 때문에, 어쩔수 없는 선택이긴 하다. 게다가 **월말 종가**를 기준으로, 투자의사결정(모멘텀 계산 등)과 리밸런싱을 동시에 진행하였다. 곰곰히 생각해보면 알겠지만, 이런 식의 매매는 현실적으로 불가능하다. 

<br/>
따라서 위의 문제점들을 감안하여, 모형설계의 지향점을 다음과 같이 설정하였다. 

* **투자 유니버스를 충분히 확장하여, 실질적인 Global Fixed-income Allocation 전략을 구현**
	* 미국에 상장되어 있는 채권 ETF (미국 이외의 거래소는 아직 고려하지 않음)
	* 미국 채권시장 뿐만 아니라 미국 이외의 채권시장까지 커버
	* Sector, Duration, 표시통화, USD hedge 여부 등으로 세분화
* **보다 현실적인 Backtest를 통해, 실현가능한 전략 빌드**
	* 매매가능 ETF 자산들의 Daily market price를 이용하여, Rigorous한 시뮬레이션 (투자의사결정 익영업일에 매매수행, 비용반영 등)
	* Cash management

<br/>

## Methodology

### 유니버스
미국에 상장되어 있는 다음의 24개 ETF를 투자 유니버스로 선정한다. 대부분의 시가총액이 1조원 이상이다. 참고로, 최초설정일이 다소 최근이더라도, Underlying index가 충분히 긴 시간동안 존재하는 종목으로 선택하였다. 

*As of 2018.04.30*
<div class="table-wrapper" markdown="block">

| Ticker |               Description               | Duration (Year) | MarketCap (B,USD) | Expense (%) |  Inception date | Underlying start |
|:------:|:---------------------------------------:|:--------:|:-----------------:|:-------:|:-------:|:----------------:|
| AGG    | US Aggregate                            | 6.1      | 55.2              | 0.05    | 2003-09 | 1976-01          |
| BIL    | US T-bill                               | 0.2      | 3.4               | 0.14    | 2007-05 | 1991-12          |
| SHY    | US Treasury Short                       | 1.9      | 12.6              | 0.15    | 2002-07 | 2004-12          |
| IEF    | US Treasury Intermediate                | 7.5      | 8.7               | 0.15    | 2002-07 | 2004-12          |
| TLT    | US Treasury Long                        | 17.5     | 7.7               | 0.15    | 2002-07 | 2004-12          |
| TIP    | US Tips                                 | 7.8      | 24.3              | 0.20    | 2003-12 | 1997-03          |
| LQD    | US Investment grade                     | 8.6      | 32.1              | 0.15    | 2002-07 | 1998-12          |
| HYG    | US High yield                           | 3.9      | 15.1              | 0.49    | 2007-04 | 1998-12          |
| MBB    | US MBS                                  | 5.5      | 12.1              | 0.09    | 2007-03 | 1976-01          |
| MUB    | US Muni                                 | 4.9      | 9.0               | 0.25    | 2007-09 | 2007-08          |
| BKLN   | US Bankloan                             | 4.0      | 8.3               | 0.65    | 2011-03 | 2001-12          |
| CWB    | US Convertible                          |       | 4.3               | 0.40    | 2009-04 | 2003-01          |
| HYD    | US High yield muni                      | 6.5      | 2.4               | 0.35    | 2009-02 | 1995-12          |
| PFF    | US Preferred stock                      | 6.1      | 16.1              | 0.47    | 2007-03 | 2003-09          |
| BWX    | Ex-US Treasury local (USD unhedged)     | 8.0      | 2.1               | 0.50    | 2007-10 | 2007-09          |
| WIP    | Ex-US Tips local (USD unhedged)         | 7.2      | 0.7               | 0.50    | 2008-03 | 2011-01          |
| BNDX   | Ex-US Treasury local (USD hedged)       | 7.8      | 11.5              | 0.11    | 2013-06 | 2013-01          |
| IGOV   | Developed Treasury local (USD unhedged) | 8.4      | 1.1               | 0.35    | 2009-01 | 2001-04          |
| FLOT   | Developed Float-rate USD                | 1.9      | 9.1               | 0.20    | 2011-06 | 2003-10          |
| PICB   | Developed IG local (USD unhedged)       | 7.5      | 0.1               | 0.50    | 2010-06 | 2010-04          |
| HYXU   | Developed HY local (USD unhedged)       | 3.9      | 0.1               | 0.40    | 2012-04 | 2009-06          |
| EMB    | EM Treasury USD                         | 7.3      | 11.4              | 0.39    | 2007-12 | 1997-12          |
| EMLC   | EM Treasury local (USD unhedged)        | 5.2      | 5.6               | 0.42    | 2010-07 | 2007-12          |
| EMHY   | EM HY USD                               | 5.8      | 0.6               | 0.50    | 2012-04 | 2001-10          |

</div>
<br/>


### Key idea

안토나치의 논문에서 언급하고 있듯이, 글로벌 채권시장에서 Dual momentum 전략으로 기대할 수 있는 효과는, **CAGR은 US High yield와 유사한 수준** 및 **변동성은 US High yield의 50%** 정도이다. 하지만 이걸로는 부족하다. 이 포스트에서는 채권시장 내 상대적으로 Risky 한 자산을 적극적으로 이용하여,  Dual momentum 고유의 기대치를 좀더 상향시키고자 한다. 

아이디어는 단순하다. **모멘텀 전략은 변동성이 큰 시장에서 효과적**이라는 연구결과가 있다. 즉 **채권시장보다는 주식시장, 대형주보다는 중소형주, 선진국보다는 이머징 국가**에서 모멘텀 전략이 잘 작동한다고 알려져있다. 위의 투자 유니버스에서 **TLT**(US Treasury Long)의 듀레이션은 17년 이상으로, 유니버스 내에서 가장 길고, 변동성도 높은 편이다. 따라서 이 자산의 독자적인 모멘텀(아래에서 정의한다)이 확인되는 경우, 원래의 Dual momentum 전략에 추가하여 포트폴리오를 강화한다. 즉, 

<center><big><b>GFAM = Dual momentum + Reinforcement</b></big></center>


<br/>

### Simulation setup
* **모멘텀**: 안토나치는 각 자산별 모멘텀 지표를 해당 자산의 12개월 수익률로 정의하였으나, 여기에서는 최근의 모멘텀을 일부 반영하기 위해 다음의 정의를 사용하였다. 
<center><b>모멘텀 = 12개월 수익률의 100% + 6개월 수익률의 50% + 3개월 수익률의 25%</b></center>

* **중장기 추세**: 위의 모멘텀 지표와는 별도로, 어떤 자산의 **3개월 이평선> 12개월 이평선** 일때, 해당 자산의 중장기 추세가 존재한다고 가정한다. 이는 Reinforcement에 사용된다. 
* 백테스트 기간: 2002.12.31 ~ 2018.03.31 (약 15년)
* Monthly rebalancing
* 매매비용 10bp
* Gross exposure 99% (매매비용 인출을 감안)
* Risky asset: **TLT** (US Treasury Long)
* Cash asset: **AGG** (US Aggregate)

참고로 이 포스트에서 AGG는 벤치마크와 유사한 의미로 취급하고 있다. 따라서 Absolute momentum이 약해진 경우 최소한 벤치마크는 따라가기 위해 Cash asset을 AGG로 설정하였다. **BIL**(US T-bill)을 Cash asset으로 정해도 크게 상관은 없으며, 이 경우 아래에 기술되는 백테스트 결과의 CAGR이 소폭 낮아지게 된다. 


<br/>

### Trading rules
1. 투자의사결정: 매월 마지막 영업일
	* **Dual momentum**
		* Absolute momentum: 모멘텀이 (-) 인 종목을 유니버스에서 제거한다. 
		* Relative momentum: 모멘텀 상위 **N**개의 종목을 선택한다. **N**은 추후 결정. 
	* **Reinforcement**: Dual momentum에서 선택한 종목의 수가 충분치 않을 때에는, 
		* Risky asset의 모멘텀이 (+)이거나 중장기 추세가 존재하는 경우, 해당 Risky asset으로 나머지 종목 수를 채운다. 
		* 그렇지 않은 경우에는 Cash asset으로 대신한다. 
	* **Rank-based weighting**: 선택된 종목들의 모멘텀 Ranking에 따라 포트폴리오 비중을 결정한다. 즉 모멘텀이 큰 종목의 비중을 높게, 모멘텀이 작은 종목의 비중을 낮게 조절한다. 
	
2. 매매: 매월 첫번째 영업일
	* 전일 투자의사결정된 포트폴리오가 전월의 포트폴리오와 다른 경우에 한해, 당일 종가(Adjusted)로 매매한다. 
	* 만약 어떤 종목이 시장에서 아직 거래되지 않는다면, 해당 종목의 Underlying index를 이용하여 그 종목의 시장가격을 역으로 추정한다. 

<br/>

## Backtest

### Absolute vs. Relative vs. Dual
우선 **N=1** 인 경우에 대해, Dual momentum을 구성하는 두 전략, 즉 Absolute momentum과 Relative momentum의 성과를 비교해보자. 다음은 각 전략별 누적수익률 차트이다. Reinforcement 전략은 아직 적용하지 않았다. 

<center><img src="https://gem763.github.io/assets/img/20180505/cum_compare_mode.png" alt="cum_compare_mode"/></center>

<center><img src="https://gem763.github.io/assets/img/20180505/stats_compare_mode.png" alt="stats_compare_mode"/></center>


* Absolute momentum (파랑)은 여차하면 현금으로 이동하므로, 상대적으로 안정적인 성과를 보여준다. 하지만 CAGR은 가장 작다. 
* Relative momentum (주황)은 시장상황에 따라 변동성이 매우 큰 편이지만, 상대적으로 높은 수익률이 기대된다.  
* Dual momentum (빨강) 성과는 상대적으로 안정적(특히 MDD 측면에서)이면서도, 수익을 낼 때에는 Relative momentum 처럼 확실한 수익성을 보인다. 
* 즉 안토나치의 주장과 동일하다. 
* 그렇다면 몇 개의 종목을 선택하는 것이 좋을까?

<br/>

### Calibration
**N**을 1부터 10까지 변화시켜가며 Dual momentum 전략의 성과를 측정해보았다. 

<center><img src="https://gem763.github.io/assets/img/20180505/stats_npicks_1_10.png" alt="stats_npicks_1_10"/></center>


* **N**이 커질수록 CAGR은 낮아지는 경향이 있다. 
* 하지만 연변동성(및 MDD)은 훨씬 빠른 속도로 작아진다.  
* 이에따라 Sharpe는 **N=5** 이후로 Saturation 된다. 
* 종목 1개만 선택하는 것은, 수익률은 좋을 지 모르나 운용 안정성이 매우 떨어진다. 다음의 누적수익률 차트를 보면, **N=5** 부근에서 안정적으로 Saturation 되는 것을 확인할 수 있다.  

<center><img src="https://gem763.github.io/assets/img/20180505/cum_npicks_1_5.png" alt="cum_npicks_1_5"/></center>

<br/>

### Reinforcement
**N=5** 인 경우의 Dual momentum에 대해, Reinforcement 전략을 적용해보자. Dual momentum을 기본 골격으로 하되, Dual momentum에 의해 선택된 종목의 수가 부족한 경우, 상대적으로 Risky한 US Treasury Long의 추가편입 여부를 결정하였다. 

<center><img src="https://gem763.github.io/assets/img/20180505/cum_compare_reinforce.png" alt="cum_compare_reinforce"/></center>

* Dual momentum 전략만 적용 (파랑): US High yield(주황)와 거의 유사한 CAGR과 상대적으로 낮은 변동성이 관측된다. 이는 안토나치 논문과 동일한 결과이다. 
* **Dual momentum + Reinforcement** (빨강): 글로벌 채권 유니버스 내에서 상대적으로 Risky한 자산의 익스포져를 확대한 결과, **Dual momentum만 적용한 것 보다 우월한 결과**가 도출되었다. 대신 변동성은 좀더 커진다. 

<br/>

### Evaluation

이제 위의 Trading rule (= Dual momentum + Reinforcement)에 따라 운용하는 전략을 **GFAM** (Global Fixed-income Allocation Model)이라고 부르자. 그리고 **N=5**에 대해서 해당 전략의 성과를 측정해보자. 글로벌 주식시장인 **ACWI** (MSCI All country)와 채권시장인 **AGG** (US Aggregate), 주식시장과 채권시장의 중간격인 **HYG** (US High yield)를 참고용으로 추가하였다. 

<center><b>Cumulative return</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/cum_base.png" alt="cum_base"/></center>

<br/>

<center><b>Statistics</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/stats_base.png" alt="stats_base"/></center>

각 성과지표에 관한 설명은 [투자성과의 측정](https://gem763.github.io/investment%20base/%ED%88%AC%EC%9E%90%EC%84%B1%EA%B3%BC%EC%9D%98-%EC%B8%A1%EC%A0%95.html)을 참고하기 바란다. GFAM의 성과 중 특이할 만한 것들만 추려서 얘기하자면, 

* **중장기: CAGR 9.1%, 변동성 7.8%, Sharpe 1.17**
백테스트 기간동안 **CAGR은 주식시장과 유사**한 수준이었으나, 변동성은 낮았기 때문에, 다른 자산들과 비교하여 **매우 양호한 Sharpe**(주식시장의 약 2.5배)를 기록하였다. 

* **단기: CAGR 7.5%, 변동성 6.8%, Sharpe 1.16**
반면 1년 단위로 Rolling 하며 관측하면, GFAM이 주식시장 보다 더 낫다고 보기는 힘들었다. 주식시장의 1년 Rolling CAGR의 중간값(Median)은 매우 높은 수준이었으며, 그 만큼 변동성도 굉장히 컸다. 

* **MDD 14%**
MDD는 채권시장과 유사한 수준으로 낮았다. 반면 주식시장의 MDD는 극히 높았다. 리만 때 무려 60% 가까이를 깨먹었다. 참고로 (안토나치의 저서 뿐만 아니라) Dual momentum 전략과 관련된 각종 Article에서, Dual momentum의 가장 큰 유용함으로 꼽는 것들 중 하나가 바로 **낮은 MDD**이다. 

* 주식시장과의 베타는 (-) 영역이었으며, 채권시장 보다도 소폭 낮았다. 

* **1년 단위의 손실 가능성**(즉 GFAM에 1년 동안 투자했을 때의 손실 가능성)은 **약 3%로 지극히 낮았다**. 참고로 주식시장의 1년 단위 손실가능성은 20%가 넘었으며, 채권시장 조차도 10%에 육박하는 손실가능성이 있었다. 
* 아래 차트는 유니버스 전 종목의 Risk-return profile 이다. 검은 실선의 기울기는 GFAM(빨강)의 Sharpe를 의미한다. **유니버스 내에 있는 대부분의 종목보다 Sharpe가 높다**는 사실을 알 수 있다. 


<center><b>Risk-return profile</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/profile_base.png" alt="profile_base"/></center>


그렇다면, GFAM의 연도별 성과는 어땟을까. 다음 차트는 연도별로 GFAM의 누적수익률(전년도 마지막 영업일=1.0)을 나타낸다. 2018년도는 1~3월 까지의 수익률만 나타내었다. 

<center><b>Yearly cumulative returns</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/cum_yearly_all.png" alt="cum_yearly_all"/></center>


GFAM의 성과가 몇몇 해(2003, 2008, 2009, 2011년)에 다소 과도하게 좋았기 때문에, 다른 해의 성과가 시각적으로 좀 묻히는 경향이 있다(y축이 동일 스케일이므로). 해당 구간들을 제거한 연도별 누적수익률 차트도 확인해보자.


<center><b>Yearly cumulative returns excluding 2003, 2008, 2009, 2011</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/cum_yearly_removed.png" alt="cum_yearly_removed"/></center>

이제 좀더 명확하게 보인다. 2007년과 2010년을 제외한 대부분의 해에서 US Aggregate보다 나은 성과가 나오고 있다. 변동성이 좀 크긴 하다. 

한편 GFAM에 투자했을 때의 **단기적(1년)인 기대효과**는 어떨까. 아래는 위의 백테스트 기간(약 15년)동안 얻을수 있는 1년 성과를 모두 추출해서, 분포를 그려본 것이다. 

<center><b>1-Year Rolling Distributions</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/dist_base.png" alt="dist_base"/></center>


* 붉은색 수직선은 각 분포의 중간값(Median)을 의미한다. 맨 위쪽의 Statistics 차트에서 CAGR(Rolling 1Y), Standard dev(Rolling 1Y), Sharpe(Rolling 1Y)의 대표값으로 각각 사용되었다. 

* **CAGR(Rolling 1Y)에서 0 이하 부분의 넓이는 1년 단위의 손실가능성을 의미**한다. 위에서 언급했듯이 GFAM(빨강)의 1년 단위 손실가능성은 매우 낮은 편인데, 이는 Absolute momentum 전략을 통해 하락장의 손실을 사전에 차단했기 때문인 것으로 보인다. 

* 주식시장(회색)의 무서운 점은, 1년 단위의 성과라도 매우 넓게 분포되어 있다는 사실이다. 주식시장의 CAGR(Rolling 1Y)을 보면, 중간값은 다른 자산들에 비해 높은  편이나, 왼쪽으로 꼬리가 길게 늘어져있어서 1년 손실가능성이 매우 크다. 주식시장의 Standard dev(Rolling 1Y)도 마찬가지이다. 잘 보면 변동성이 50%에 육박하는 구간도 존재했었다. 

* 반면 **GFAM의 분포는 다른 자산들에 비해 좁은 편**(즉 첨도가 높다)이다. 이는 통계적으로 볼때, 
**성과통계의 신뢰도가 높다**고 해석될 수 있다. 


GFAM의 성과를 종목별로 분해해보자. 아래 왼쪽 차트는 GFAM의 **1일 수익률을 100이라고 했을 때, 각 종목이 평균적으로 몇 %를 기여했는 지**를 나타낸다. 오른쪽 차트는 각 종목이 선택된 총 월수를 의미한다. 

<center><b>Performance breakdown</b></center>
<center><img src="https://gem763.github.io/assets/img/20180505/perf_breakdown.png" alt="perf_breakdown"/></center>


* GFAM의 성과에 가장 큰 기여를 한 종목은 **TLT**(US Treasury Long)였고, 그 다음으로는 **PFF**(US Preferred stock)였다. 반면 **LQD**(US Investment grade)에서는 손실이 발생했다. 

* **TLT**의 성과 기여도(평균 20%)가 큰 것은, **Reinforcement 전략에 따른 익스포져 확대 및 높은 변동성** 등의 이유가 크게 작용했을 것으로 추정된다. 

* **LQD**가 GFAM 전체 수익률의 10%를 까먹었다는 의미는 아니다. 총 투자 월수가 워낙 적었고 공교롭게도 그 기간동안 약간의 손실이 발생했기 때문에, 평균적으로 손실 기여도가 큰 것처럼 보일 뿐이다. 

* 총 투자기간과 성과기여도 사이에 의미있는 상관관계가 있는 것 같지는 않다. 다만, 총 투자기간이 길수록 해당 성과기여도 값의 신뢰도가 높다고 이해하면 된다. 


<br/>

> **NOTE**
> 2003년 이후로의 백테스트 결과를 잘 살펴보면, 꽤나 신경쓰이는 구간이 있다. 2008년과 2009년이다. 해당 구간동안 GFAM 성과의 변동폭이 컸고, 결과적으로는 (+) 요인으로 작용하면서, **GFAM 고유의 성과를 다소 왜곡했을 것이라는 주장이 가능**하다. 과연 그럴까? 2010년 이후의 성과를 측정하여 확인해보자.
> 
> **Cumulative return since 2010**
> <center><img src="https://gem763.github.io/assets/img/20180505/cum_since2010.png" alt="cum_since2010"/></center>
> 
> **Statistics since 2010**
> <center><img src="https://gem763.github.io/assets/img/20180505/stats_since2010.png" alt="stats_since2010"/></center>
> 
> **GFAM(2003~) vs. GFAM(2010~)**
> <center><img src="https://gem763.github.io/assets/img/20180505/stats_compare_since2010.png" alt="stats_compare_since2010"/></center>
>
> * CAGR과 Sharpe가 소폭 낮아지긴 했지만, 전구간(2003~) 백테스트 결과보다 열등하다고 보긴 힘들었다. (물론 통계적인 유의도까지 검증한 건 아니다)
> * 오히려 **MDD는 큰 폭으로 개선**되었으며, **1년 단위 손실가능성은 1% 이하로 축소**되었다. 
>


### Portfolio history
아래 차트는 2015년 부터 2018년 4월까지 40개월 간의 포트폴리오 변동을 보여준다. 이중 5개월 동안 **TLT**(US Treasury Long)가 단독으로 편입되었으며, 5개월 동안 Cash asset인 **AGG**(US Aggregate, 흰색)가 단독 편입되었다.

<center><img src="https://gem763.github.io/assets/img/20180505/port_history.png" alt="port_history"/></center>

단독으로 편입되었다고 해서 실제로 그렇게 운용되지는 않을 것이다. 미국에는 유사한 Underlying index를 추종하는 ETF가 많다. 예를들어 **TLT**와 같은 US Treasury Long-term ETF인 **SPTL**가 상장되어 있고, 성과도 거의 동일하다. 


### Drawback: High turnover
**일반적으로 Dual momentum 전략은 매매회전율(Turnover ratio)이 낮은 편**이다. 안토나치는 그의 논문에서, 매매회전률이 평균적으로 200%를 넘지 않는다고 서술해 놓았다. 사실이다. 하지만 그가 제안한 백테스트에 한해서만 사실이다. 그의 전략은 대부분 3-4개의 종목에서 한 개를 골라내는 것이었으며, 모멘텀이 높은 종목의 추세는 한동안 지속될 가능성이 높았기 때문에, 매매회전률이 낮을 수밖에 없었다.

그러나 이 보고서에서는, **훨씬 넓은 범위의 투자 유니버스에서 여러 종목을 한꺼번에 선택**하는 전략을 쓰고 있기 때문에, 매매회전률이 높다. 당사의 매매회전률 산출공식(= 12개월 매도비율)에 근거하여 GFAM의 매매회전률 추이를 계산해보면 아래 차트와 같다. **평균적으로 600%에 육박**한다. 참고로, 월 리밸런싱하는 펀드(예를들어 퀀트운용본부의 퀀트MP펀드)의 당사 매매회전률 가이드라인은 500% 수준이다. 

<center><img src="https://gem763.github.io/assets/img/20180505/turnover.png" alt="turnover"/></center>


## Conclusions

개리 안토나치의 Dual momentum 전략을 바탕으로 GFAM(Global Fixed-income Allocation Model)을 설계해보았다. 기관투자자의 입장에서 유용한 GFAM 전략을 setup 하고 원래의 Dual momentum 을 보완하기 위해서, Reinforcement 전략을 새롭게 추가하였다. 그 과정에서, Momentum space에서는 변동성이 높은 자산이 상대적으로 유리하다는 사실을 활용하였다. 종목수는 5개 이상이 적절하였다. 

GFAM으로 기대되는 성과는 다음과 같다. 
* **중장기: CAGR 7-9%, 연변동성 7-8%, Sharpe 1.1-1.2**
* **단기: CAGR 7-8%, 연변동성 7%, Sharpe 1.1-1.2**
* **MDD 7-15%**
* **1년 단위 손실가능성 1-3%**
<br/>

한편 GFAM 포트폴리오는 다음과 같은 특징이 있었다. 
* US Treasury Long의 성과기여도가 높은 편이다. 
* 매매회전률이 높다. 


## Next agenda
후속작업으로 다음의 Allocation model 설계가 진행되고 있다.  

* Global Equity Allocation by countries
* US Equity Sector Allocation
* US Equity Factor(Style) Allocation
* Commodity Allocation
* Currency Allocation
* Korea Equity Sector Allocation
* Korea Equity Factor(Style) Allocation

