---
layout: post
title: 선형대수학의 기초
tags: [선형대수학]
excerpt_separator: <!--more-->

---

계량투자에 필요하다고 생각되는, 선형대수학의 기초적인 부분을 정리합니다. 이 포스트는 수시로 업데이트 될 수 있습됩니다. 
<!--more-->



## 행렬의 크기

행렬 ####TEST#### 에 하나의 실수를 대응시키는 연산을 의미하며, **Norm**, **Trace**, **Determinant** 등이 있다. 
<br/>

### Norm

####TEST#### 에 대해서, **####TEST####-Norm** 은 다음과 같이 정의된다. 

####TEST####

이 중 ####TEST#### 인 경우를 프로베니우스 놈 (Frobenius norm)이라고 하며, 다음과 같이 쓰기도 한다. 

####TEST####
<br/>

### Trace

정방행렬 ####TEST####에 대해서, Trace는 다음과 같이 정의된다. 

####TEST####

Trace는 아래와 같은 성질을 지닌다. 

* ####TEST####
* ####TEST####
* ####TEST####
* ####TEST####
* ####TEST#### : 트레이스 트릭(Trace trick)

이중 트레이스 트릭을 이용하면, 다음과 같은 이차형식(Quadratic form)의 미분을 쉽게 할 수 있다. 벡터 ####TEST#### 에 대해서, 

####TEST####
<br/>

### Determinant

행렬식이라고도 하며, 정방행렬 ####TEST####에 대해 코팩터 확장(Cofactor expansion)이라고 불리는 재귀적인 방법으로 다음과 같이 정의된다. 

####TEST####

* ####TEST#### : 코팩터(Cofactor)
* ####TEST#### : 마이너(Minor)라고 하며, ####TEST####에서 ####TEST####-행과 ####TEST####-열을 지워서 얻어진 행렬의 행렬식

Determinant는 다음과 같은 성질을 지닌다. 

* ####TEST####
* ####TEST####
* ####TEST####
* ####TEST####



## 역행렬

역행렬 ####TEST####은 정방행렬인 ####TEST####와 다음의 관계를 만족하는 행렬을 뜻한다.

####TEST####

역행렬과 Determinant는 다음의 관계를 가진다. 


####TEST####

####TEST####

여기서 

* ####TEST#### : 코팩터 행렬(Cofactor matrix, comatrix)
* ####TEST#### : Adjugate matrix 또는 Adjoint matrix 라고 불린다. 



## 행렬의 미분


### 스칼라를 벡터로 미분

####TEST#### 에 대하여, 

####TEST####

스칼라를 벡터로 미분한 결과값인 ####TEST#### 를 그레디언트 벡터(Gradient vector)라고 부른다. 
<br/>

### 벡터를 스칼라로 미분

####TEST####,  ####TEST#### 에 대하여,


####TEST####
<br/>

### 벡터를 벡터로 미분

####TEST####에 대하여, 


####TEST####

벡터를 벡터로 미분한 결과값인 ####TEST####를 **자코비안 행렬**(Jacobian matrix)이라고 한다. 특히 ####TEST####에 대하여, 다변수함수 ####TEST####의 이차도함수 ####TEST####를 **헤시안 행렬**(Hessian matrix)이라고도 한다. 


####TEST####
<br/>

### 스칼라를 행렬로 미분

####TEST####에 대하여, 

####TEST####
<br/>

### 주요 미분규칙

벡터 ####TEST####와 행렬 ####TEST####에 대하여, 

* ####TEST####

* ####TEST####

* ####TEST####

* ####TEST####









