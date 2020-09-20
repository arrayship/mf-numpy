# Matrix Factorization from Scratch with NumPy

Tested on MovieLens 20M dataset.

|  | start | end |
|-|-|-|
| train | 2010.01.01 | 2013.12.31 |
| validation | 2014.01.01 | 2014.06.30 |
| test | 2014.07.01 | 2014.12.31 |

## Run

### Command

최초 실행 시 --download True로 raw data 저장. \
최초 실행 시 --process True로 preprocess 및 결과 저장.

~~~shell
> python main.py --download True --process True
~~~

### Arguments

| name | choices | default | help |
|-|-|-|-|
| \-\-download | \(bool\) | False | dataset download 여부 |
| \-\-process | \(bool\) | False | preprocess 및 결과 저장 여부 |
| \-\-modeltype | \(str\) mf | mf | model type 설정 |
| \-\-f\_dim | \(int\) | 32 | latent factor 벡터의 차원 |
| \-\-pc | \(float\) | 0\.5 | regularizer 페널티 상수 |
| \-\-lr | \(float\) | 0\.01 | learning rate |
| \-\-max\_epoch | \(int\) | 20 | 최대 epoch |

## Directory Structure

~~~
├── main.py
├── README.md
├── modules
│   ├── ds_handler.py     <- dataset handling functions
│   ├── metrics.py        <- evaluation metric functions
│   └── mf.py             <- model class and functions
└── ds
    ├── raw               <- raw data
    └── processed         <- preprocessed data
~~~

## Matrix Factorization

[modules/mf.py](modules/mf.py)

기본적인 matrix factorization model에 bias, ridge regularizer를 추가한 model.\
stochastic gradient descent로 학습.

~~~python
class MF():
    """
    matrix factorization (with bias, ridge regularizer) model class
    bu, bm: trainable bias vector for user, movie
    Lu, Lm: trainable latent factor matrix for user, movie
    """
    def __init__(self, n_u, n_m, f_dim, train_avg):
        """
        initialize latent factor matrices

        n_u, n_m: # of users, movies
        f_dim: latent factor space dimension
        train_avg: average ratings for bias and cold start
        """
        self.train_avg = train_avg
        self.bu = np.zeros(n_u)
        self.Lu = np.random.rand(n_u, f_dim)
        self.bm = np.zeros(n_m)
        self.Lm = np.random.rand(n_m, f_dim)
~~~

Notations
- $D_{train}$, $D_{test}$: train, test dataset
- $n^{user}_{train}$, $n^{movie}_{train}$: train dataset의 user, movie 수
- $(u, m, r) \in D_{train}, D_{test}$: userId가 $u$인 user가 movieId가 $m$인 movie를 $r$점으로 평가
- $(ui, mi, r) \in D^e_{train}, D^e_{test}$: \
$u \rightarrow ui \in \{- 1, \cdots, n^{user}_{train} - 1 \}$, $m \rightarrow mi \in \{- 1, \cdots, n^{movie}_{train} - 1 \}$로 인코딩한 결과 \
$u$, $m$이 $D_{train}$에 없는 경우 $- 1$로 인코딩
- $\mu$: $r \in D^e_{train}$의 전체 평균
- $\mu_{ui}$, $\mu_{mi}$: user가 $ui$인 경우, movie가 $mi$인 경우 $r \in D^e_{train}$의 평균
- $f$: latent factor 벡터의 차원
- $(b^{user}_{ui} \in \mathbb{R}, L^{user}_{ui} \in \mathbb{R}^{f})$, $(b^{movie}_{mi} \in \mathbb{R}, L^{movie}_{mi} \in \mathbb{R}^{f})$: user $ui$, movie $mi$의 bias, latent factor
- $(b^{user} \in \mathbb{R}^{n^{user}_{train}}, L^{user} \in \mathbb{R}^{n^{user}_{train} \times f})$, $(b^{movie} \in \mathbb{R}^{n^{movie}_{train}}, L^{movie} \in \mathbb{R}^{n^{movie}_{train} \times f})$: 학습 parameter \
$D^e_{train}$에 존재하는 모든 user, movie의 bias를 concatenate한 벡터, latent factor를 concatenate한 행렬
- $\lambda$: ridge regularizer 페널티 상수
- $\gamma$: learning rate

### Prediction

$\hat{r}(ui, mi) =
\begin{cases}
\mu + b^{user}_{ui} + b^{movie}_{mi} + L^{user}_{ui} \cdot L^{movie}_{mi}, & & u \in D_{train}, m \in D_{train}\\
\mu_{ui}, & & u \in D_{train}, m \notin D_{train}\\
\mu_{mi}, & & u \notin D_{train}, m \in D_{train}\\
\mu, & & u \notin D_{train}, m \notin D_{train}\\
\end{cases}$

$D_{train}$에 존재하지 않는 (인코딩 값이 $- 1$인) user나 movie의 $r$을 예측하는 경우 bias와 latent factor가 존재하지 않으므로 $D_{train}$에서의 평균으로 예측.

~~~python
    def predict(self, ui, mi):
        """
        predict score from user, movie pair
        
        ui, mi: encoded index of user, movie
        """
        if (ui == - 1) and (mi == - 1):
            return self.train_avg[(- 1, - 1)]
        elif (ui == - 1):
            return self.train_avg[(- 1, mi)] + self.bm[mi]
        elif (mi == - 1):
            return self.train_avg[(ui, - 1)] + self.bu[ui]
        else:
            return self.train_avg[(- 1, - 1)] +\
                    self.bu[ui] + self.bm[mi] +\
                    np.dot(self.Lu[ui], self.Lm[mi])
~~~

### Error (sgd train)

$e = r - \hat{r}(ui, mi) = r - (\mu + b^{user}_{ui} + b^{movie}_{mi} + L^{user}_{ui} \cdot L^{movie}_{mi})$

~~~python
    def error(self, r, r_pred):
        """        
        r: rating of mi rated by ui
        r_pred: prediction of r
        """
        return r - r_pred
~~~

### Loss (sgd train)

$l = e^2 + \lambda \left({b^{user}_{ui}}^2 + {b^{movie}_{mi}}^2 + {|L^{user}_{ui}|}^2 + {|L^{movie}_{mi}|}^2\right)$

Error의 제곱에 ridge regularizer를 더하여 loss를 정의.

### Gradient (sgd train)

$grad(b^{user}_{ui}) = - 2 e + 2 \lambda b^{user}_{ui}$\
$grad(L^{user}_{ui}) = - 2 e L^{movie}_{mi} + 2 \lambda L^{user}_{ui}$\
$grad(b^{movie}_{mi}) = - 2 e + 2 \lambda b^{movie}_{mi}$\
$grad(L^{movie}_{mi}) = - 2 e L^{user}_{ui} + 2 \lambda L^{movie}_{mi}$

~~~python
    def gradient(self, b_ui, l_ui, b_mi, l_mi, e, pc):
        """
        b_ui, b_mi: bias for ui, mi
        l_ui, l_mi: latent factor vector for ui, mi
        pc: regulizer penalty constant
        """
        g_ui = (((- 2) * e) + (2 * pc * b_ui),
                ((- 2) * e * l_mi) + (2 * pc * l_ui))
        g_mi = (((- 2) * e) + (2 * pc * b_mi),
                ((- 2) * e * l_ui) + (2 * pc * l_mi))
        return g_ui, g_mi
~~~

### Update (sgd train)

$(b, L) \leftarrow (b, L) - \gamma (grad(b), grad(L))$

~~~python
    def update(self, ui, mi, g_ui, g_mi, lr):
        """
        g_ui, g_mi: gradient corresponding to ui, mi
        lr: learning rate
        """
        self.bu[ui] -= (lr * g_ui[0])
        self.Lu[ui] -= (lr * g_ui[1])
        self.bm[mi] -= (lr * g_mi[0])
        self.Lm[mi] -= (lr * g_mi[1])
~~~

### Step (sgd train)

예측, error 계산, gradient 계산, parameter 업데이트를 한 step으로 구성.

~~~python
    def step(self, ui, mi, r, pc, lr):
        """
        sgd one step
        """
        r_pred = self.predict(ui, mi)
        e = self.error(r, r_pred)
        g_ui, g_mi = self.gradient(
                self.bu[ui], self.Lu[ui], self.bm[mi], self.Lm[mi], e, pc
                )
        self.update(ui, mi, g_ui, g_mi, lr)
~~~

## Results

~~~shell
> python main.py --download True --process True --f_dim 32 --pc 1.0 --lr 0.003 --max_epoch 10
Epoch:   1, Train RMSE:  0.933, Val RMSE:  1.001, Elapsed:   2.6min
Epoch:   2, Train RMSE:  0.877, Val RMSE:  0.951, Elapsed:   2.6min
Epoch:   3, Train RMSE:  0.867, Val RMSE:  0.937, Elapsed:   2.5min
Epoch:   4, Train RMSE:  0.863, Val RMSE:  0.931, Elapsed:   2.5min
Epoch:   5, Train RMSE:  0.862, Val RMSE:  0.927, Elapsed:   2.5min
Epoch:   6, Train RMSE:  0.861, Val RMSE:  0.925, Elapsed:   2.5min
Epoch:   7, Train RMSE:  0.861, Val RMSE:  0.924, Elapsed:   2.5min
Epoch:   8, Train RMSE:  0.860, Val RMSE:  0.923, Elapsed:   2.7min
Epoch:   9, Train RMSE:  0.860, Val RMSE:  0.923, Elapsed:   2.5min
Epoch:  10, Train RMSE:  0.860, Val RMSE:  0.922, Elapsed:   2.6min
Train RMSE: 0.860, Test RMSE: 0.951
~~~
