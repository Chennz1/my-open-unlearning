说明
v1-v3: 基础steering_vector方式调整，不符unlearning整体效果和目的
v3.5-v6 改用自适应方式调整权重，期间尝试了ortho_init parameter_orthorize等等操作，意义不大，后期修改了eval_metric暴露了模型的问题。
v7 自适应 + 类GRU的机制，期望更好的定位需要修改的参数。