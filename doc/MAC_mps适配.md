## mps

需要检查所有 use_cuda这个参数的地方，默认为1就是cuda，为0就是cpu；

第一个地方是
networks.py的init方法
其余地方通过搜索检查
![](./cuda检索.png)