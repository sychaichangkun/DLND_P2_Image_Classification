# 关于模型的设定：

## 模型架构中节点的参数数量
毫无疑问，越多的节点数量意味着模型的表达能力越强。

## 如何确定模型的参数被完全训练

你观察训练模型阶段的 loss 值，你发现到了 一定数量的 epoch 时，loss 值就基本不下降了，说明模型已经被完全训练了。
这个时候你可以：

o 在之后的训练中降低 epoch 数（若不改变节点参数，那么这么训练也没有意义）

o 增加模型的节点数（提升模型参数规模）

o 尝试使用早停的策略（不过该技巧目前已经不常用了）

## 卷积层的参数设定
个人认为，卷积层的本质实际上是一个个可训练滤波器的集合，执行的功能是对图像进行采样。因而一些比较基础的设计卷积层的思想包括：

o 各卷积层之间，卷积核的尺度应逐步增大，例如 2 - 4 - 6 这样的形式（当然也有一直使用小核的情况）

o 卷积的输出通道数出也类似如上的表述，可以是 32 - 64 - 128 这样的形式

o Stride 越小，那么图像保留的细节就越多；不过也有可能会带来过多的细节，不利于训练，所以需要权衡。一般取 1～3 之类的数值

o Pooling 层的逻辑大致相同，不过一般我们取比较小的 pooling 核尺寸，避免特征被过度的舍弃。

## 全连接层的节点数设定

个人的观点是，此处全连接层的作用就是一个分类器，即把卷积层提取的特征进行分类。
那么这边层数的设定应当根据你 flatten 层的到的 tensor 的规模来，然后逐步的缩小，你可以考虑类似 512 - 256 -10 的设定。
## 参考资料

[How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
[Why not set dropout in cnn layers](https://www.reddit.com/r/MachineLearning/comments/42nnpe/why_do_i_never_see_dropout_applied_in/)
