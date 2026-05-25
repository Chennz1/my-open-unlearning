我建议把理论从原稿里的 **“same-size retain vector naturally aligns with forget vector”** 改成更稳的表述：

> TVC is a constrained compensation problem in a retain/forget functional subspace.
> 原始 TVC 是这个问题在 (K=1)、单位度量、强 alignment 假设下的特例。

这样比直接证明“1:1 随机 retain subset 足够”更合理，也能正面回应 decision 里对 scale-matching 机制和理论过度声称的批评。原文目前的核心公式是 task vector (\tau=\theta_{\mathrm{ft}}-\theta_{\mathrm{base}})、direct task-vector unlearning，以及 TVC 的 (\theta_{\mathrm{unlearned}}=\theta_{\mathrm{target}}-\alpha\tau_{\mathrm{fg}}+\beta\tau_{\mathrm{comp}})；同时原文引用 Cheng et al. 的结论，认为低资源 fine-tuning 下线性层 task vector 近似落在输入样本张成的线性子空间中。   任务向量的加减法基础来自 task arithmetic：task vector 由 fine-tuned 权重减去 pre-trained 权重得到，并可通过加法/取负来改变模型行为。([arXiv][1]) Cheng et al. 进一步把线性层 task vector 与输入子空间联系起来，说明 task vector 可以被看成携带训练数据分布信息的参数空间表征。([arXiv][2])

下面是我认为可以改进成 EMNLP 版主方法或理论小节的公式推导。

---

# 1. 先把原始 TVC 写成一个近似目标，而不是直接设 (\alpha=\beta=1)

设目标模型是：

[
\theta_{\mathrm{target}}
========================

\theta_{\mathrm{base}}
+
\tau_{\mathrm{full}},
]

其中完整 fine-tuning 数据为：

[
D_{\mathrm{full}}
=================

D_{\mathrm{forget}}
\cup
D_{\mathrm{retain}}.
]

原文 TVC 从 base model 上重学 forget set 得到：

[
\theta_{\mathrm{fg}}
====================

\mathrm{FT}(\theta_{\mathrm{base}},D_{\mathrm{forget}}),
\quad
\tau_f
======

\theta_{\mathrm{fg}}-\theta_{\mathrm{base}},
]

并从 retain subset 得到：

[
\theta_{\mathrm{c}}
===================

\mathrm{FT}(\theta_{\mathrm{base}},D'_{\mathrm{retain}}),
\quad
\tau_c
======

\theta_{\mathrm{c}}-\theta_{\mathrm{base}}.
]

原始 TVC 更新是：

[
\theta_{\mathrm{TVC}}
=====================

## \theta_{\mathrm{target}}

\alpha\tau_f
+
\beta\tau_c.
]

如果设 (\alpha=\beta=1)，就是：

[
\theta_{\mathrm{TVC}}
=====================

## \theta_{\mathrm{target}}

\tau_f
+
\tau_c.
]

问题在于，这里隐含了一个很强的假设：

[
\tau_c
\approx
\tau_{\mathrm{shared}}(\tau_f),
]

即 retain compensation vector 能够近似恢复从 (\tau_f) 中误删掉的 shared/general component。原稿把这个解释成 “same-size natural magnitude alignment”，但 reviewer 的主要质疑正是：**matching sample size 不等于 matching direction / distribution / functional effect**。decision 也明确说需要更清楚的 scale-matching 理论解释。 

所以更合理的推导应当是：不要假设 (\tau_c) 天然正确，而是把 compensation 写成一个优化出来的投影。

---

# 2. 从 fine-tuning 动力学推导：为什么“同样样本数”不够

设单样本 loss 为 (\ell_i(\theta))，数据集 (D) 上的平均 loss 为：

[
L_D(\theta)
===========

\frac{1}{|D|}
\sum_{i\in D}
\ell_i(\theta).
]

从 (\theta_0=\theta_{\mathrm{base}}) 开始 fine-tuning (T) 步：

[
\theta_{t+1}
============

## \theta_t

\eta_t\nabla L_D(\theta_t).
]

因此 task vector 是：

[
\tau_D
======

# \theta_T-\theta_0

*

\sum_{t=0}^{T-1}
\eta_t\nabla L_D(\theta_t).
]

在低学习率、短程 fine-tuning、局部线性近似下：

[
\nabla L_D(\theta_t)
\approx
\nabla L_D(\theta_0),
]

于是：

[
\tau_D
\approx
-------

\bar{\eta}T
\cdot
\frac{1}{|D|}
\sum_{i\in D}
g_i,
\quad
g_i=\nabla_\theta \ell_i(\theta_0).
]

令：

[
\mu_D
=====

\mathbb E_{i\sim D}[g_i],
\quad
\Sigma_D
========

\mathrm{Cov}_{i\sim D}(g_i).
]

则 task vector norm 的期望近似为：

[
\mathbb E|\tau_D|_2^2
\approx
\bar{\eta}^2T^2
\left(
|\mu_D|_2^2
+
\frac{1}{|D|}
\mathrm{Tr}(\Sigma_D)
\right).
]

这个式子说明：

[
|D_1|=|D_2|
\not\Rightarrow
|\tau_{D_1}|\approx |\tau_{D_2}|.
]

只有当下面三个条件同时近似成立时，same-size 才可能推出 magnitude alignment：

[
\mu_{D_1}\approx \mu_{D_2},
\quad
\Sigma_{D_1}\approx \Sigma_{D_2},
\quad
T_1,\eta_1 \approx T_2,\eta_2.
]

因此，EMNLP 版里最好不要再说“1:1 retain sampling theoretically guarantees scale matching”。更合理的说法是：

[
|D'*{\mathrm{retain}}|
\approx
|D*{\mathrm{forget}}|
]

只是控制了 variance term 的一个因素；真正需要匹配的是 gradient / activation distribution。

---

# 3. 更合理主方法一：Retain-Subspace Projection TVC

我建议把主方法从：

[
\theta_{\mathrm{target}}
------------------------

\tau_f
+
\tau_c
]

升级为：

[
\theta_{\mathrm{target}}
------------------------

\tau_f
+
P_{\mathcal S_R}(\tau_f),
]

其中 (P_{\mathcal S_R}(\tau_f)) 表示把 forget vector 中“会伤害 retain behavior 的部分”投影回 retain task-vector 子空间。

## 3.1 构造 retain task-vector basis

不要只采一个 retain subset。把 retain data 分成 (K) 个小 shard：

[
D_{\mathrm{retain}}^{(1)},\ldots,D_{\mathrm{retain}}^{(K)}.
]

每个 shard 从 base model fine-tune 一次：

[
\theta_r^{(k)}
==============

\mathrm{FT}(\theta_{\mathrm{base}},D_{\mathrm{retain}}^{(k)}),
]

[
\tau_r^{(k)}
============

## \theta_r^{(k)}

\theta_{\mathrm{base}}.
]

定义 retain task-vector 子空间：

[
\mathcal S_R
============

\mathrm{span}
\left{
\tau_r^{(1)},\ldots,\tau_r^{(K)}
\right}.
]

把这些向量组成矩阵：

[
T_R
===

\left[
\tau_r^{(1)},\ldots,\tau_r^{(K)}
\right]
\in
\mathbb R^{d\times K}.
]

这里 (K=1) 时就退化成原文的 single compensation vector；(K>1) 时，方法不再依赖一次随机采样，而是估计 retain subspace。

---

## 3.2 用 retain 度量定义“误删的 shared component”

为了定义“(\tau_f) 中哪些部分会影响 retain behavior”，引入一个 retain metric：

[
M_R\succeq 0.
]

最简单可以取单位矩阵：

[
M_R=I.
]

更合理的是取 retain Fisher / diagonal Fisher：

[
M_R
===

# F_R

\mathbb E_{(x,y)\sim D_{\mathrm{retain}}}
\left[
\nabla_\theta \ell(x,y;\theta)
\nabla_\theta \ell(x,y;\theta)^\top
\right].
]

如果 full Fisher 太贵，可以用 diagonal Fisher：

[
F_R^{\mathrm{diag}}
===================

\mathrm{diag}
\left(
\mathbb E_{(x,y)\sim D_{\mathrm{retain}}}
[
g(x,y)\odot g(x,y)
]
\right).
]

这样 inner product 变成：

[
\langle u,v\rangle_{M_R}
========================

u^\top M_Rv.
]

该度量的直觉是：如果某个参数方向在 retain data 上 Fisher 大，说明这个方向对 retain behavior 重要；误删这部分会损伤 utility。这个表述也能接上 tangent-space task arithmetic，因为 Ortiz-Jimenez et al. 认为 task arithmetic 的效果与 tangent/linearized space 中的 weight disentanglement 有关，linearized fine-tuning 可以增强这种 disentanglement。([arXiv][3])

---

## 3.3 投影目标

我们希望找到一个 retain-subspace compensation：

[
c
=

T_Rw
]

来恢复 (\tau_f) 中 retain-relevant 的部分。最直接的 ridge projection 是：

[
w^*
===

\arg\min_w
\left|
T_Rw-\tau_f
\right|_{M_R}^2
+
\lambda
|w|_2^2.
]

展开：

[
\left|
T_Rw-\tau_f
\right|_{M_R}^2
===============

(T_Rw-\tau_f)^\top
M_R
(T_Rw-\tau_f).
]

对 (w) 求导：

[
2T_R^\top M_R(T_Rw-\tau_f)
+
2\lambda w
==========

0.

]

得到闭式解：

[
\boxed{
w^*
===

\left(
T_R^\top M_RT_R+\lambda I
\right)^{-1}
T_R^\top M_R\tau_f
}
]

于是 compensation vector 为：

[
\boxed{
c^*
===

T_Rw^*
}
]

最终模型为：

[
\boxed{
\theta_{\mathrm{RSP}}
=====================

## \theta_{\mathrm{target}}

\tau_f
+
T_R
\left(
T_R^\top M_RT_R+\lambda I
\right)^{-1}
T_R^\top M_R\tau_f
}
]

这可以命名为：

> Retain-Subspace Projection TVC, RSP-TVC.

这个推导的好处是：compensation 不再是“随机 same-size retain vector”，而是 **forget vector 在 retain subspace 上的投影**。换句话说，你不是盲目加一个 retain vector，而是只把 (\tau_f) 中对 retain behavior 重要的成分加回来。

---

# 4. 加入 forget leakage penalty：避免 compensation 把忘掉的东西加回来

上面的投影只考虑 retain preservation，但没有显式防止 (c^*) 在 forget data 上重新激活遗忘内容。为此可以引入 forget metric：

[
M_F\succeq 0.
]

例如：

[
M_F
===

# F_F

\mathbb E_{(x,y)\sim D_{\mathrm{forget}}}
[
g(x,y)g(x,y)^\top
].
]

新的目标是：

[
w^*
===

\arg\min_w
\underbrace{
\left|
T_Rw-\tau_f
\right|*{M_R}^2
}*{\text{recover retain-relevant component}}
+
\gamma
\underbrace{
\left|
T_Rw
\right|*{M_F}^2
}*{\text{avoid forget reactivation}}
+
\lambda
|w|_2^2.
]

展开求导：

[
2T_R^\top M_R(T_Rw-\tau_f)
+
2\gamma T_R^\top M_FT_Rw
+
2\lambda w
==========

0.

]

所以：

[
\boxed{
w^*
===

\left(
T_R^\top M_RT_R
+
\gamma T_R^\top M_FT_R
+
\lambda I
\right)^{-1}
T_R^\top M_R\tau_f
}
]

最终更新为：

[
\boxed{
\theta_{\mathrm{IA\text{-}TVC}}
===============================

## \theta_{\mathrm{target}}

\tau_f
+
T_Rw^*
}
]

可以命名为：

> Interference-Aware TVC, IA-TVC.

这个版本比原始 TVC 更容易说服 reviewer，因为它正面优化两个目标：

[
\text{retain recovery}
\quad
\text{and}
\quad
\text{forget non-reactivation}.
]

原始 TVC 是下面这个特例：

[
K=1,\quad
T_R=\tau_c,\quad
M_R=I,\quad
M_F=0,\quad
\lambda=0.
]

此时：

[
w^*
===

\frac{
\langle \tau_c,\tau_f\rangle
}{
|\tau_c|_2^2
}.
]

如果进一步假设：

[
\langle \tau_c,\tau_f\rangle
\approx
|\tau_c|_2^2,
]

那么：

[
w^*\approx 1,
]

就得到原始 TVC 的：

[
\theta_{\mathrm{target}}-\tau_f+\tau_c.
]

这就把原文的 (\beta=1) 从“默认设定”改成了“一个特定条件下的近似解”。

---

# 5. 单 compensation vector 下的 (\beta^*) 推导

如果你们不想增加多个 retain vectors，也可以保留一个 compensation vector (\tau_c)，但不要固定 (\beta=1)。设：

[
a=\tau_f,
\quad
b=\tau_c.
]

固定 (\alpha=1)，考虑目标：

[
J(\beta)
========

\underbrace{
|\beta b|*{M_F}^2
}*{\text{compensation should not reactivate forget}}
+
\lambda
\underbrace{
|-a+\beta b|*{M_R}^2
}*{\text{retain perturbation should be small}}
+
\rho\beta^2.
]

展开：

[
J(\beta)
========

\beta^2 b^\top M_F b
+
\lambda
(-a+\beta b)^\top M_R(-a+\beta b)
+
\rho\beta^2.
]

对 (\beta) 求导：

[
2\beta b^\top M_Fb
+
2\lambda
\left(
\beta b^\top M_Rb
-----------------

b^\top M_Ra
\right)
+
2\rho\beta
==========

0.

]

因此：

[
\boxed{
\beta^*
=======

\frac{
\lambda\langle b,a\rangle_{M_R}
}{
|b|*{M_F}^2
+
\lambda|b|*{M_R}^2
+
\rho
}
}
]

这比固定 (\beta=1) 更合理，因为它明确表达了三件事：

[
\langle b,a\rangle_{M_R}
]

越大，说明 (\tau_c) 越能恢复 (\tau_f) 误删的 retain-relevant 成分，(\beta) 应该越大；

[
|b|_{M_F}^2
]

越大，说明 (\tau_c) 越可能在 forget data 上重新激活被遗忘内容，(\beta) 应该越小；

[
\rho
]

控制过补偿，防止添加过大的 retain vector。

这个公式可以直接解释原文 Table 3 / Table 4 的现象：(\beta=0) retain utility 差，说明确实存在 retain-relevant damage；但 compensation 太多又会引入 interference，所以 sample ratio 超过 1:1 后 utility 开始下降。原文实验里 (\beta=0) 时 retain 指标明显下降，(\beta=1.0) 后 ES Re.、MU 和 R-RL 恢复；sampling ratio ablation 也显示 1:1 附近较优、过大比例会造成 utility 下降。 

---

# 6. 同时学习 (\alpha,\beta)：二维闭式解

如果想更完整，可以同时求 unlearning strength (\alpha) 和 compensation strength (\beta)。

仍设：

[
a=\tau_f,
\quad
b=\tau_c.
]

更新为：

[
\Delta
======

-\alpha a+\beta b.
]

更新后，forget contribution 的 residual 近似是：

[
a+\Delta
========

(1-\alpha)a+\beta b.
]

retain behavior 的 perturbation 是：

[
\Delta
======

-\alpha a+\beta b.
]

于是目标写成：

[
J(\alpha,\beta)
===============

\underbrace{
|(1-\alpha)a+\beta b|*{M_F}^2
}*{\text{forget residual}}
+
\lambda
\underbrace{
|-\alpha a+\beta b|*{M_R}^2
}*{\text{retain drift}}
+
\rho\beta^2.
]

定义：

[
A_F=|a|*{M_F}^2,\quad
B_F=|b|*{M_F}^2,\quad
C_F=\langle a,b\rangle_{M_F},
]

[
A_R=|a|*{M_R}^2,\quad
B_R=|b|*{M_R}^2,\quad
C_R=\langle a,b\rangle_{M_R}.
]

对 (\alpha,\beta) 分别求导，可以得到线性系统：

[
\boxed{
\begin{bmatrix}
A_F+\lambda A_R
&
-(C_F+\lambda C_R)
\
-(C_F+\lambda C_R)
&
B_F+\lambda B_R+\rho
\end{bmatrix}
\begin{bmatrix}
\alpha^*
\
\beta^*
\end{bmatrix}
=============

\begin{bmatrix}
A_F
\
-C_F
\end{bmatrix}
}
]

解出：

[
(\alpha^*,\beta^*)
==================

\left[
\begin{array}{cc}
A_F+\lambda A_R
&
-(C_F+\lambda C_R)
\
-(C_F+\lambda C_R)
&
B_F+\lambda B_R+\rho
\end{array}
\right]^{-1}
\left[
\begin{array}{c}
A_F
\
-C_F
\end{array}
\right].
]

这个推导可以作为 EMNLP 主文里的 “scale-matching analysis”。它的意义是：(\alpha,\beta) 不再是人工网格搜索的超参数，而是由 forget/retain functional geometry 给出的 closed-form coefficients。

---

# 7. Layer-wise activation-space 版本：更贴近 Cheng et al. 的理论

原文引用的 Cheng et al. 结论是：线性层 task vector 近似位于输入样本张成的子空间中；Cheng et al. 原文也把线性层 task vector 与对应 input subspace 和 interference 联系起来。 ([arXiv][2]) 所以最自然的推导不是在全参数空间里看 (|\tau|_2)，而是在每一层看 task vector 对 activation 的影响。

对线性层 (l)，设权重为 (W_l)，输入 activation 为 (h_l(x))。若参数扰动为 (\Delta W_l)，则一阶输出变化是：

[
\Delta h_{l+1}(x)
\approx
\Delta W_l h_l(x).
]

定义 forget / retain activation covariance：

[
M_{F,l}
=======

\frac{1}{|D_F|}
\sum_{x\in D_F}
h_l(x)h_l(x)^\top,
]

[
M_{R,l}
=======

\frac{1}{|D_R|}
\sum_{x\in D_R}
h_l(x)h_l(x)^\top.
]

于是 layer-wise functional norm 为：

[
|\Delta W_l|*{M*{D,l}}^2
========================

\mathrm{Tr}
\left(
\Delta W_l
M_{D,l}
\Delta W_l^\top
\right)
=======

\frac{1}{|D|}
\sum_{x\in D}
|\Delta W_lh_l(x)|_2^2.
]

这比参数空间 norm 更合理，因为它度量的是“这个参数更新在某个数据分布上的函数影响”。

---

## 7.1 Layer-wise compensation objective

设 forget vector 在第 (l) 层为：

[
\tau_{f,l}.
]

我们想找 compensation matrix (C_l)，使它：

1. 在 retain activation 上恢复 (\tau_{f,l}) 被减掉的影响；
2. 在 forget activation 上尽量不重新激活 forget behavior；
3. 本身不要太大。

目标：

[
C_l^*
=====

\arg\min_{C_l}
\left|
C_l-\tau_{f,l}
\right|*{M*{R,l}}^2
+
\gamma
\left|
C_l
\right|*{M*{F,l}}^2
+
\rho
|C_l|_F^2.
]

展开：

[
\left|
C_l-\tau_{f,l}
\right|*{M*{R,l}}^2
===================

\mathrm{Tr}
\left[
(C_l-\tau_{f,l})
M_{R,l}
(C_l-\tau_{f,l})^\top
\right].
]

对 (C_l) 求导：

[
2(C_l-\tau_{f,l})M_{R,l}
+
2\gamma C_lM_{F,l}
+
2\rho C_l
=========

0.

]

所以：

[
C_lM_{R,l}
----------

\tau_{f,l}M_{R,l}
+
\gamma C_lM_{F,l}
+
\rho C_l
========

0.

]

整理得：

[
C_l
\left(
M_{R,l}
+
\gamma M_{F,l}
+
\rho I
\right)
=======

\tau_{f,l}M_{R,l}.
]

闭式解：

[
\boxed{
C_l^*
=====

\tau_{f,l}
M_{R,l}
\left(
M_{R,l}
+
\gamma M_{F,l}
+
\rho I
\right)^{-1}
}
]

最终 layer-wise 更新：

[
\boxed{
W_{u,l}
=======

## W_{\mathrm{target},l}

\tau_{f,l}
+
C_l^*
}
]

这可以命名为：

> Activation-Projected TVC, AP-TVC.

这个版本的理论感更强，因为它不是说“同样大小的数据产生同样大小的向量”，而是说：

[
C_l^*
]

是在 retain activation subspace 上恢复 (\tau_{f,l})，同时在 forget activation subspace 上抑制 reactivation 的最优 closed-form compensation。

---

# 8. 如果想保持 task-vector 风格：把 (C_l) 限制在 retain vector span 中

上面的 (C_l^*) 是任意矩阵，可能让 reviewer 觉得不再是 task-vector method。可以把它限制成 retain vectors 的线性组合。

设第 (l) 层 retain basis 为：

[
T_{R,l}
=======

[
\tau_{r,l}^{(1)},\ldots,\tau_{r,l}^{(K)}
].
]

令：

[
C_l
===

\sum_{k=1}^K
w_{l,k}
\tau_{r,l}^{(k)}.
]

写成：

[
C_l=T_{R,l}w_l.
]

目标：

[
w_l^*
=====

\arg\min_{w_l}
\left|
T_{R,l}w_l-\tau_{f,l}
\right|*{M*{R,l}}^2
+
\gamma
\left|
T_{R,l}w_l
\right|*{M*{F,l}}^2
+
\lambda
|w_l|_2^2.
]

闭式解：

[
\boxed{
w_l^*
=====

\left(
T_{R,l}^{\top}M_{R,l}T_{R,l}
+
\gamma T_{R,l}^{\top}M_{F,l}T_{R,l}
+
\lambda I
\right)^{-1}
T_{R,l}^{\top}M_{R,l}\tau_{f,l}
}
]

最终更新：

[
\boxed{
W_{u,l}
=======

## W_{\mathrm{target},l}

\tau_{f,l}
+
T_{R,l}w_l^*
}
]

这是我最推荐放进主文的方法版本，因为它有三个优点：

1. 保留 TVC 的 task-vector arithmetic 形式；
2. 不再依赖单个随机 retain subset；
3. 给出了 closed-form scale matching / subspace matching 解释。

---

# 9. Distribution-matched retain subset：替代随机 1:1 采样

如果你们不想做 (K) 个 retain vectors，也可以至少把 retain subset 的选择从 random same-size 改成 distribution-matched coreset。

设 (\phi(x)) 是一个 representation，比如 base model 最后一层 hidden state 的 mean pooling。forget distribution 的均值：

[
\mu_F
=====

\frac{1}{|D_F|}
\sum_{x\in D_F}
\phi(x).
]

从 retain pool 中选加权样本，权重为 (q_i)，满足：

[
q_i\ge 0,
\quad
\sum_i q_i=1.
]

选择：

[
q^*
===

\arg\min_q
\left|
\sum_i q_i\phi(x_i^R)
---------------------

\mu_F
\right|_2^2
+
\rho|q|_2^2.
]

如果想加 covariance matching：

[
q^*
===

\arg\min_q
\left|
\mu_R(q)-\mu_F
\right|_2^2
+
\kappa
\left|
\Sigma_R(q)-\Sigma_F
\right|_F^2
+
\rho|q|_2^2.
]

其中：

[
\mu_R(q)
========

\sum_i q_i\phi(x_i^R),
]

[
\Sigma_R(q)
===========

\sum_i q_i
(\phi(x_i^R)-\mu_R(q))
(\phi(x_i^R)-\mu_R(q))^\top.
]

然后用这个 weighted retain subset fine-tune：

[
\tau_c(q^*)
===========

\mathrm{FT}
(\theta_{\mathrm{base}},D_{\mathrm{retain}},q^*)
------------------------------------------------

\theta_{\mathrm{base}}.
]

更新：

[
\theta_u
========

## \theta_{\mathrm{target}}

\tau_f
+
\beta^*\tau_c(q^*).
]

这个方法可以解释为：

> We match not only the number of retain examples, but also their representation-level distribution to the forget set, so that the compensation vector restores nearby shared directions rather than arbitrary retain knowledge.

这会直接回应 BNyP 的问题：为什么 same-size retain subset 能代表 full retain distribution？更准确的回答是：**same-size 本身不能保证，所以我们使用 representation/activation matching 来选择 retain subset。**

---

# 10. 可以写成论文里的 Proposition

可以在 EMNLP 版中加入一个温和的 proposition，不要写 theorem guarantee。

**Proposition: RSP-TVC is the optimal retain-subspace compensation under a local quadratic approximation.**

设 unlearning update 为：

[
\Delta
======

-\tau_f
+
T_Rw.
]

在 retain metric (M_R) 和 forget metric (M_F) 下，若局部函数变化由二次型近似，则最优 retain-subspace compensation 满足：

[
w^*
===

\arg\min_w
|T_Rw-\tau_f|*{M_R}^2
+
\gamma|T_Rw|*{M_F}^2
+
\lambda|w|_2^2.
]

其闭式解为：

[
w^*
===

\left(
T_R^\top M_RT_R
+
\gamma T_R^\top M_FT_R
+
\lambda I
\right)^{-1}
T_R^\top M_R\tau_f.
]

原始 TVC 是 (K=1)、(M_R=I)、(M_F=0)、(\lambda=0)、且 (\langle\tau_c,\tau_f\rangle\approx|\tau_c|^2) 时的近似特例。

这段非常适合放主文，因为它把 reviewer 认为“直觉 extension”的地方变成了一个明确的 constrained projection problem。

---

# 11. 最推荐的最终方法公式

如果只能选一个版本，我建议用下面这个作为 EMNLP 主方法：

[
\boxed{
\theta_{\mathrm{unlearned}}
===========================

## \theta_{\mathrm{target}}

\tau_f
+
T_R
\left(
T_R^\top M_RT_R
+
\gamma T_R^\top M_FT_R
+
\lambda I
\right)^{-1}
T_R^\top M_R\tau_f
}
]

其中：

[
\tau_f
======

## \mathrm{FT}(\theta_{\mathrm{base}},D_{\mathrm{forget}})

\theta_{\mathrm{base}},
]

[
T_R
===

[
\tau_r^{(1)},\ldots,\tau_r^{(K)}
],
]

[
\tau_r^{(k)}
============

## \mathrm{FT}(\theta_{\mathrm{base}},D_{\mathrm{retain}}^{(k)})

\theta_{\mathrm{base}}.
]

如果计算资源紧张，可以设：

[
K=1.
]

则：

[
\theta_{\mathrm{unlearned}}
===========================

## \theta_{\mathrm{target}}

\tau_f
+
\beta^*
\tau_c,
]

[
\boxed{
\beta^*
=======

\frac{
\langle\tau_c,\tau_f\rangle_{M_R}
}{
|\tau_c|*{M_R}^2
+
\gamma|\tau_c|*{M_F}^2
+
\lambda
}
}
]

这比原始：

[
\beta=1
]

更有理论说服力。

---

# 12. 写法建议

主文可以这样表述：

> The original TVC rule can be viewed as adding a retain vector whose scale is assumed to match the retain-relevant component removed by the forget vector. However, matching sample size alone does not guarantee matching functional effect. We therefore formulate compensation as a retain-subspace projection problem under a local quadratic approximation. The resulting closed-form coefficient recovers the original TVC rule as a special case, while explicitly penalizing forget reactivation.

这句话既承认原始 TVC 是合理 heuristic，又把新方法提升为更 principled 的公式。相比继续声称 “natural magnitude alignment guarantees (\alpha=\beta=1)”，这个版本更容易过审。

[1]: https://arxiv.org/abs/2212.04089?utm_source=chatgpt.com "Editing Models with Task Arithmetic"
[2]: https://arxiv.org/html/2503.08099v1 "Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors"
[3]: https://arxiv.org/abs/2305.12827?utm_source=chatgpt.com "Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models"
