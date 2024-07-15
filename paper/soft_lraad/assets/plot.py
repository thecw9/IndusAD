import matplotlib.pyplot as plt
import numpy as np


# 定义函数
def f(x):
    # return np.maximum(0, 1 - x)
    return np.exp(-x)


# 生成x数据
x = np.linspace(-1, 2, 400)

# 生成y数据
y = f(x)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="max(0, 1-x)", color="blue")
plt.title("Graph of max(0, 1-x)")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

# 设置纵横比为等距
plt.gca().set_aspect("equal", adjustable="box")

plt.savefig("plot.pdf")

plt.show()
