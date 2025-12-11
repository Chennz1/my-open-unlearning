## 示例草稿区

## Methods

### 3. Unlearning with MSA

Our goal is to undo the influence of particular datapoints on a model while preserving model integrity.
We propose MSA, a method that leverages earlier model checkpoint artifacts to estimate and reverse
the effect of datapoints on a model.

**MSA procedure**

- **Input:** A model θD, a model checkpoint C (with weights θ0), and a set of datapoints Df.
- **Step 1:** Finetune C on Df to obtain a weight-space vector ⃗θf. This estimates the effect of Df.
	We hypothesize that using a checkpoint not yet exposed to the unlearning targets can result in more
	effective unlearning.
- **Step 2:** Apply the vector ⃗θf to model weights θD to obtain model θunlearn.
- **Output:** A model θunlearn that should approximate an ideal reference model θD\Df.

We finetune θ0 for ef epochs on the forget set Df, resulting in a new model with parameters θ1. The
resulting forget vector, denoted as ⃗θf := θ1 − θ0, captures the influence of the forget set in weight
space. The parameters of the resulting unlearned model, θunlearn, can then be expressed as:

```
θunlearn = θD − α ⃗θf
```

where α controls the magnitude of the update along the forget vector, effectively aiming to remove the
influence of the forget set while preserving the model’s overall performance.

Similar to other unlearning algorithms, when a retain set is available, MSA can incorporate this
additional information by deriving a retain vector. In this case, we continue finetuning the model with
parameters θ0 on the retain set for er epochs to obtain a model with parameters θ2. The retain vector is
then defined as ⃗θr := θ2 − θ0. Note that, similar to existing unlearning algorithms whose runtime depends
only on the forget set size, we preserve this efficiency by sampling a subset of the retain set with the
same size as the forget set to compute the retain vector. The final unlearned model can be computed as:

```
θunlearn = θD − α ⃗θf + β ⃗θr
```

where α and β control the influence of the forget and retain vectors, respectively. Importantly, we do not
constrain the forget and retain vectors to be derived from the target model that is being unlearned. Instead,
we can leverage earlier model states in training that may have not been exposed to the unlearning targets.
The unlearned model can be parameterized by α, β, ef, er, and the model checkpoint used to obtain the
unlearning vectors. Accordingly, we denote the unlearned model as MSAckpt,α,β,ef,er. For brevity, parameters
such as ef and er are often omitted from the model description as these values are fixed and specified in the
experimental setup. We use validation sets to tune α and β.

## 讨论板块



## Methods

### 3. Unlearning with MSA


我们的目标是从参数层面移除特定数据的影响，并尽可能维持模型的必要能力。

在本节，我们介绍XXX方法，（一种通过任务向量清除特定数据参数，从而xxxx的方法）。我们首先讨论核心假设与发现，随后给出实际操作方案。

#### 3.1

观察模型微调过程的不同阶段，可以发现，模型对只是的学习主要在第一个epoch完成，而后续epoch主要是完成参数的固化和输出稳定。（如何设计实验）

（实验图1）

#### 3.2 方案



Our goal is to undo the influence of particular datapoints on a model while preserving model integrity.
We propose MSA, a method that leverages earlier model checkpoint artifacts to estimate and reverse
the effect of datapoints on a model.

**MSA procedure**

- **Input:** A model θD, a model checkpoint C (with weights θ0), and a set of datapoints Df.
- **Step 1:** Finetune C on Df to obtain a weight-space vector ⃗θf. This estimates the effect of Df.
	We hypothesize that using a checkpoint not yet exposed to the unlearning targets can result in more
	effective unlearning.
- **Step 2:** Apply the vector ⃗θf to model weights θD to obtain model θunlearn.
- **Output:** A model θunlearn that should approximate an ideal reference model θD\Df.

We finetune θ0 for ef epochs on the forget set Df, resulting in a new model with parameters θ1. The
resulting forget vector, denoted as ⃗θf := θ1 − θ0, captures the influence of the forget set in weight
space. The parameters of the resulting unlearned model, θunlearn, can then be expressed as:

```
θunlearn = θD − α ⃗θf
```

where α controls the magnitude of the update along the forget vector, effectively aiming to remove the
influence of the forget set while preserving the model’s overall performance.

Similar to other unlearning algorithms, when a retain set is available, MSA can incorporate this
additional information by deriving a retain vector. In this case, we continue finetuning the model with
parameters θ0 on the retain set for er epochs to obtain a model with parameters θ2. The retain vector is
then defined as ⃗θr := θ2 − θ0. Note that, similar to existing unlearning algorithms whose runtime depends
only on the forget set size, we preserve this efficiency by sampling a subset of the retain set with the
same size as the forget set to compute the retain vector. The final unlearned model can be computed as:

```
θunlearn = θD − α ⃗θf + β ⃗θr
```

where α and β control the influence of the forget and retain vectors, respectively. Importantly, we do not
constrain the forget and retain vectors to be derived from the target model that is being unlearned. Instead,
we can leverage earlier model states in training that may have not been exposed to the unlearning targets.
The unlearned model can be parameterized by α, β, ef, er, and the model checkpoint used to obtain the
unlearning vectors. Accordingly, we denote the unlearned model as MSAckpt,α,β,ef,er. For brevity, parameters
such as ef and er are often omitted from the model description as these values are fixed and specified in the
experimental setup. We use validation sets to tune α and β.