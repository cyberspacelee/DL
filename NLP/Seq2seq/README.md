## seq2seq

- utils.py：数据处理
- Seq2Seq.py：训练模型
- Seq2Seq_Attention.py：加入 attention

训练结果：

![](img/seq2seq-attention.png)

模型：

- seq2seq：

![](img/seq2seq-model.png)

- seq2seq with attention:

  ![](img/seq2seq-attention-model.png)

[attention paper](https://arxiv.org/pdf/1508.04025.pdf)

![](img/attention.png)

仅实现了第一种 dot