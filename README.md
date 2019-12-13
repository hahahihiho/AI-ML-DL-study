# 머신러닝(ML) 종류,분류

* 사실 종류,분류는 그리 중요하지 않다.  
  분류를 하는 이유는 처음 쓰는 사람도 쉽게 쓰게 하기위해서인데,  
  결국 수학적 지식들을 기준에 따라 나눠놓은것   
  그냥 보기좋게 분류해보고 싶었..지만.. 쉽지않고 공부할건 멀었다..  
* 분류해놓으면 검색해서 찾아서 공부하기 편하니까..  
* 이 문서는 참조만 하시길.  

## Index
> 1. 학습방식에 따른 분류 
> 2. 머신러닝 종류
> 3. 유명한 딥러닝 알고리즘
> 4. 추가 잡다한 정보(추후 정리)

---



## 1. 학습방식에 따른 분류

> * 지도학습 (supervised learning)
>   > 분류(classification)
>   > 회귀(regression)
> * 비지도학습 (unsupervised learning)
>   
>   > 군집화(clustering)
> * 강화학습 (reinforcement learning)

---



## 2. 머신러닝 종류

* 지도학습

  > * regression(회귀) : 연속데이터
  >   - linear regression
  >   - NN(Neural Network)(ANN,DNN,perceptron)
  >   - Regulation(규제)
  >     - Ridge (L2)
  >     - Lasso (L1)
  > * classification(분류)
  >   - KNN(K Nearist Neigbor)
  >   - Decision Tree
  >   - SVM(Support Vector Machine)
  >   - logistic regression
  >   - Ensemble
  >     - Random Forest
  >     - Gradient Tree Boosting
  >   - Linear Discriminant Analysis (선형판별분석) 
  
* 비지도학습(cluster)

  > * K-means
  > * Hierarchical clustering
  > * DBSCAN(Density-based spatial clustering of applications with noise)
  
* 번외 : 여러가지 기법

  > * 통계적
  >   * Standardization
  >   * Normalization (Z-score standardization or Min-Max scaling)
  >   * PCA(Principal Component Analysis) (주성분분석) : 고차원 feature -> 저차원 feature
  > * 데이터 검증
  >   - GridSearchCV : depth와 split의 수를 여러가지 변수로 대입해서 정확도 확인
  >   - K-fold Cross Validation

----------


## 3. 유명한 딥러닝 알고리즘

* CNN (이미지 인식에 사용)
* RNN (자연어 처리에 사용) (NLP)
* LSTM (RNN의 변형버전)

 ### 1. CNN (Convolutional Neural Network)
>
>   > * many filters -> pooling (크기 줄임)
>   >
>   > * various model
>   >
>   >   > * VGG16,19( 16,19 is # of layers )
>   >   >
>   >   >   > many (3*3)filters+pooling -> features
>   >   >   >
>   >   >   > [more info] : https://bskyvision.com/504
>   >   >
>   >   > * GoogLeNet Inception ( V1 : (5x5) -> V2: (3x3)+(3x3) -> V3 : (1x3)+(3x1)+(1x3)+(3x1) )
>   >   >
>   >   >   > |  V1   |     V2      |           V3            |
>   >   >   > | :---: | :---------: | :---------------------: |
>   >   >   > | (5x5) | (3x3)+(3x3) | (1x3)+(3x1)+(1x3)+(3x1) |
>   >   >   >
>   >   >   > various size of filters -> features
>   >   >   >
>   >   >   > many small filter : to reduce calculation
>   >   >   >
>   >   >   > Auxiliary classifier : help gradiant vanishing problem
>   >   >   >
>   >   >   > [more info] : https://bskyvision.com/539
>   >   >   > 					 : https://m.blog.naver.com/laonple/220710707354

> ### 2. RNN (Recurrent NN)
>
> > * 앞의정보를 뒤에 전달

> ### 3. LSTM(Long Short Term Memory)
>
> > * RNN 이 앞의 정보가 점점 소실된다는 점에서 착안한(장기 기억 알고리즘)

---



## 4. 추가 잡다한 정보

### 1. 데이터 전처리(Preprocessing)

> 1. 이상값 처리
>    * NA처리
>    * outlier test(이상값 처리)
> 1. 데이터 인벨런스
>    * 적절한 학습을 위한 데이터의 균형
>    * Oversampling/Undersampling 등..(SMOTE..등)
> 1. 데이터 가공
>    * one hot encoding
>    * standardization/ normalization
> 1. 분석(EDA)
>    * 주성분 분석 (PCA - Principal component analysis)
>    * 시각화(visualization)

---

### 2. Tensor-Flow

- 가중치 : tensor

- keras - framework

---

### 3. logit, sigmoid, softmax

- regression 을 classification으로 해주는 함수들  
  
  | logit | sigmoid | softmax |
  | ----- | ------- | ------- |
  | ![logit](https://latex.codecogs.com/gif.latex?y%20%3D%20ln%7Bp%20%5Cover%201-p%7D) | <img src="https://latex.codecogs.com/svg.latex?\;p={1\over1-e^{-y}}" />  |   ![softmax](https://latex.codecogs.com/gif.latex?f%28y%29_i%20%3D%20%5Cfrac%7Be%5E%7By_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BN%7D%20e%5E%7By_k%7D%7D) |
  
  <img src="https://latex.codecogs.com/svg.latex?\;y=a_0+\sum_{i=1}^{n}a_ix_i" />  (n = # of features)

-----------------

