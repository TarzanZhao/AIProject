###TODO

2020/11/28
- max-min 更改加速计算score的方式。
- 思考、实现一下，更改神经网络，从8*8和4*4的小棋盘学习到一些概率，然后利用到15*15的网络里面去。
- reward 是否可以设计实现出来一种新的方式。
- 把所有的tensor能转化为list，（numpy），可能能提高速度，把getboardtensor换为getboardlist。np.random.randint 也是非常慢的。


- 检查boardlist的流动。
- 修改实现 min-max search。
- 每次选下一步，按照到现有点的距离排序得下。