+++
date = '2025-10-04T16:07:12+08:00'
title = '一个关于GNN在Tensor Core上的实现的数学问题'
+++

## 完整的问题

可以看[我和Gemini的对话](https://g.co/gemini/share/ad1bd7ed4a63)

## 抽象出的问题

<blockquote>

假设在mxn的矩阵中全为0，现在进行两个操作：
1. 以avgdeg/n的概率将矩阵中的值变为1；
2. 如果矩阵中有一列有一个1，那就把这一列都变成1。

现在计算avgdeg'为矩阵中1的个数/(mn)，记avgdeg=d, avgdeg'=d'

</blockquote>


## Kimi的回答

我们来逐步分析这个问题，并给出数学推导。



### 问题设定

我们有一个 $ m \times n $ 的矩阵，初始时所有元素为 0。

进行两个操作：

1. **随机置 1**：以概率 $ \frac{\text{avgdeg}}{n} $ 将每个元素独立地变为 1（否则保持 0）。  
   记 $ d = \text{avgdeg} $，所以每个元素变为 1 的概率是 $ p = \frac{d}{n} $。

2. **列传播**：如果某一列中**至少有一个 1**，则将该列**所有元素都变为 1**。

最终，我们计算新的平均度：

\[
d' = \frac{\text{矩阵中 1 的总数}}{mn}
\]

我们的目标是求 $ d' $ 关于 $ d $ 的表达式，即 $ d' = f(d) $。



### 步骤 1：计算某一列在操作 1 后**至少有一个 1**的概率

在操作 1 后，每个元素独立地以概率 $ p = \frac{d}{n} $ 变为 1。

某一列有 $ m $ 个元素，全为 0 的概率是：

\[
(1 - p)^m = \left(1 - \frac{d}{n}\right)^m
\]

所以，该列**至少有一个 1** 的概率是：

\[
P_{\text{active}} = 1 - \left(1 - \frac{d}{n}\right)^m
\]



### 步骤 2：操作 2 后，每一列要么全为 1，要么全为 0

因为在操作 2 中，只要某一列有一个 1，整列就变成 1；否则保持全 0。

所以，最终矩阵中，每一列要么是全 1，要么是全 0。



### 步骤 3：计算期望的 1 的总数

每一列以概率 $ P_{\text{active}} $ 变成全 1（即贡献 $ m $ 个 1），以概率 $ 1 - P_{\text{active}} $ 保持全 0（即贡献 0 个 1）。

所以，**期望的 1 的总数**为：

\[
\mathbb{E}[\text{total 1s}] = n \cdot m \cdot P_{\text{active}} = mn \left(1 - \left(1 - \frac{d}{n}\right)^m\right)
\]



### 步骤 4：计算 $ d' $

\[
d' = \frac{\mathbb{E}[\text{total 1s}]}{mn} = 1 - \left(1 - \frac{d}{n}\right)^m
\]



### 最终答案

$\boxed{d' = 1 - \left(1 - \frac{d}{n}\right)^m}$

这就是 $ d' $ 关于 $ d $ 的显式表达式。


## 计算我们所需的结果

- $m=8 / m=16$
- $d$ 的分布统计量 $\mu=5.856, \sigma =0.133$, 这里取 $6$
- $n$ 的典型值为 5000~50000

得到结果:
```
m=8,n=5000,d'=0.009559776622987126,d'*n=47.798883114935634
m=8,n=50000,d'=0.0009595968967535118,d'*n=47.979844837675586
m=16,n=5000,d'=0.01902816391689277,d'*n=95.14081958446386
m=16,n=50000,d'=0.0019182729673028476,d'*n=95.91364836514238
```

## 极限

### 问题
如果n趋于无穷，nd'的极限是多少？m和d保留

### Kimi的回答

我们已有

\[
d' = 1 - \left(1 - \frac{d}{n}\right)^m.
\]

令 $n\to\infty$ 而 $m,d$ 固定，考察

\[
n d' = n\left[1 - \left(1 - \frac{d}{n}\right)^m\right].
\]

利用极限

\[
\lim_{n\to\infty} n\left[1 - \left(1 - \frac{d}{n}\right)^m\right]
= \lim_{n\to\infty} n\left[1 - \exp\left(m\ln\!\Bigl(1 - \frac{d}{n}\Bigr)\right)\right].
\]

对 $\ln(1 - x)$ 做泰勒展开：$\ln(1 - x) = -x - \frac{x^2}{2} - \dots$，取首项即可：

\[
m\ln\!\Bigl(1 - \frac{d}{n}\Bigr) = -\frac{md}{n} + O\!\left(\frac{1}{n^2}\right).
\]

于是

\[
\exp\left(m\ln\!\Bigl(1 - \frac{d}{n}\Bigr)\right)
= \exp\!\left(-\frac{md}{n} + O\!\left(\frac{1}{n^2}\right)\right)
= 1 - \frac{md}{n} + O\!\left(\frac{1}{n^2}\right).
\]

代回：

\[
n d' = n\left[1 - \left(1 - \frac{md}{n} + O\!\left(\frac{1}{n^2}\right)\right)\right]
= n\left[\frac{md}{n} + O\!\left(\frac{1}{n^2}\right)\right]
= md + O\!\left(\frac{1}{n}\right).
\]

因此

\[
\boxed{\lim_{n\to\infty} n d' = md}.
\]  

## 感悟

这么简单的数学题我居然都要用AI了, 看来上班后数学能力大幅下滑.