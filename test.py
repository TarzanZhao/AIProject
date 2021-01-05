import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(8, 8))
# X = [300, 900, 1500]
# pure = [0, 0.033,0.1]
# advisor = [0.833,1.0,1.0]
# plt.plot(X, pure, color = "blue", label = "policy distribution")
# plt.plot(X, advisor, color = "pink", label = "onehot policy")
# plt.title("compare models trained on different number of play", fontsize=25)
# plt.xlabel("number of play for training", fontsize=17)
# plt.ylabel("win rate competing with min-max search", fontsize=17)
# plt.show()
# fig.sivefig("./compare models trained on different number of play.png")
# plt.close(fig)


#导入扩展包
# import matplotlib.pyplot as plt
# import numpy as np
#
# #构造数据
# width = 0.4
# seperate = width * 6
# y = [0,1,2,3,4]
# x = [0,seperate,seperate*2,seperate*3,seperate*4]
# #["policy distribution:300 play", "policy distribution:900 play", "policy distribution:1500 play", "onehot policy:300", "onehot policy:900", "onehot policy:1500"]
#
# #绘图
# plt.figure(figsize=(20,4))
#
# plt.bar(x=x,height=y,width=width,label='Data1')
# #plt.bar(x=x+width,height=y2,width=width,label='Data2')
#
# #添加数据标签
# # for x_value,y_value in zip(x,y1):
# #     plt.text(x=x_value,y=y_value,s=y_value)
#
# # for x_value,y_value in zip(x,y2):
# #     plt.text(x=x_value+width,y=y_value,s=y_value)
#
# #添加图标题和图例
# # plt.rcParams["font.sans-serif"] = ["SimHei"]
# # plt.rcParams["axes.unicode_minus"] = False
# plt.title('并列柱状图')
# plt.legend()
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 生成数据，创建 DataFrame
data = np.array([ [0.01,0.01,0.133,0.976,1, 1],
                  [0.01,0.01,0.033, 0.9,1, 1],
                  [0.01,0.033,0.1, 0.833,1, 1],
                  [0.01,0.01,0.01, 0.3,0.633, 0.8],
                  [0.01,0.01,0.01,0.367, 0.3, 0.6]
                  ])
index = [0, 1, 2, 3, 4]
Metrics = ["softmax distribution:300 play", "softmax distribution:900 play", "softmax distribution:1500 play", "onehot distribution:300 play", "onehot distribution:900 play", "onehot distribution:1500 play"]
df = pd.DataFrame(data, index=index, columns=pd.Index(Metrics, name='models trained from'))

# 设置图形属性及布局
plt.style.use('ggplot')
fig = plt.figure('win rate for competing with min-max-search')
# axes = fig.subplots(nrows=1, ncols=1)
# ax1 = axes.ravel()
ax1 = fig.subplots(nrows=1, ncols=1)

# 在第 1 个坐标系创建竖直条形图
df.plot(kind='bar', ax=ax1, alpha=0.7, title='Our models v.s. Min-max-search' )
plt.setp(ax1.get_xticklabels(), rotation=0, fontsize=15)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=15)
ax1.set_xlabel('search depth of min-max-search', fontsize=20), ax1.set_ylabel('win rate',fontsize=20)
# 调整图形显示位置
# fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95,
#                     top=0.95, hspace=0.1, wspace=0.1)

plt.show()
