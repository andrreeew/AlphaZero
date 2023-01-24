# AlphaZero


&ensp;&ensp;&ensp;&ensp;复现AlphaZero应用在反黑白棋上面，网络结构以及输入与论文中基本一致，主要区别是规模更小，论文中输入记录了过去8步，这里只记录了过去2步，以及网络的残差块只有两块，同时channel数从论文中256改为128。并且搜索c部分的$c_{cput}$为2不同于论文中的5。

&ensp;&ensp;&ensp;&ensp;net.py为网络部分，mcts.py为蒙特卡洛搜索部分，game.py为游戏部分，compete.py为对战部分，train.py为训练部分，后缀为'.params'为模型参数，src_net.py用于将游戏部分转为网络需要的输入。应该说game.py与网络是解耦的。后续需要应用于其它游戏(如五子棋，黑白棋)上面只需修改game.py即可。

&ensp;&ensp;&ensp;&ensp;AlphaZero的复现是容易的，但是想要得到一个好的AI比较困难。重点还是在训练部分，生成自我博弈数据是蛮耗时的，最好是能够将生成数据与梯度下降像论文中那样异步地来做。想要在一个普通的PC机上面就串行地跑训练十几个小时得到比较好的AI感觉不太可能。个人尝试下来采取<https://github.com/junxiaosong/AlphaZero_Gomoku>中的训练方式：每生成一把就采样部分数据做五次梯度下降。能够比较快得到一个能打败贪心的AI，但采取此方法loss不够小，或许是训练时间不够。
也尝试过<https://github.com/suragnair/alpha-zero-general>中的训练方式：每生成100把将所有的数据做一次梯度下降。这个方法效果可能会更好些，但这样每次迭代时间太长。

&ensp;&ensp;&ensp;&ensp;主要是我没什么耐心，一个模型跑五六天你还不知道是否真的有好效果，真的有点折磨人。而且我实验发现loss低不一定代表更强，loss不变也可能在变强。'reversi6.params'是我PC机(Intel i7, 2.3GHz)一晚上的训练结果，相比没有训练是有点效果，但效果也不太好。后续能使用更好的机子再来训练吧。

### 参考文献
1. Mastering the game of Go without human knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. <https://github.com/junxiaosong/AlphaZero_Gomoku>
4. <https://github.com/suragnair/alpha-zero-general>


 
