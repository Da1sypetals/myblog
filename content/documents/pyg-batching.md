+++
date = '2025-10-16T14:57:00+08:00'
title = 'PyG Batching'
+++

内容来自[这里](https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html).

高级 Mini-Batching（Mini-Batching） 

创建 mini-batching 对于让深度学习模型的训练扩展到海量数据至关重要。mini-batch 不会一个接一个地处理样本，而是将一组样本分组到一个统一的表示中，从而可以高效地并行处理。在图像或语言领域，这个过程通常是通过将每个样本重新缩放或 padding 到一组等大小的形状来实现的，然后将样本分组到一个额外的维度中。这个维度的长度等于分组在一个 mini-batch 中的样本数量，通常称为 batch_size。

由于图是一种最通用的数据结构，可以包含任意数量的节点（nodes）或边（edges），因此上述两种方法要么不可行，要么可能导致大量不必要的内存消耗。在 PyG 中，我们采用另一种方法来实现对大量样本的并行化。在这里，adjacency matrices 以对角线方式堆叠（创建一个包含多个孤立子图的巨大图），并且节点和目标特征（features）简单地沿节点维度进行拼接，即：

$$
A = \begin{bmatrix} A_1 & & \\ & \ddots & \\ & & A_n \end{bmatrix}, \quad
X = \begin{bmatrix} X_1 \\ \vdots \\ X_n \end{bmatrix}, \quad
Y = \begin{bmatrix} Y_1 \\ \vdots \\ Y_n \end{bmatrix}.
$$

与其他 batching 过程相比，此过程具有一些关键优势：

* 依赖于 message passing scheme 的 GNN operators 不需要修改，因为属于不同图的两个节点之间仍然不能交换消息。
* 没有计算或内存开销。例如，此 batching 过程完全不需要对节点或边的特征进行任何 padding。请注意，adjacency matrices 没有额外的内存开销，因为它们以稀疏（sparse）方式保存，只包含非零项，即边。

PyG 借助 `torch_geometric.loader.DataLoader` 类自动将多个图 batch 成一个巨大的图。在内部，`DataLoader` 只是一个常规的 PyTorch torch.utils.data.DataLoader，它重写了其 `collate()` 功能，即定义如何将样本列表分组在一起。因此，所有可以传递给 PyTorch DataLoader 的参数也可以传递给 PyG `DataLoader`，例如 worker 数量 `num_workers`。

在其最一般的形式中，PyG `DataLoader` 会自动将 `edge_index` tensor 增加到当前处理图之前已聚合（collated）的所有图的累积节点数，并将 `edge_index` tensors（形状为 `[2, num_edges]`）在第二个维度上进行拼接。`face` tensors，即 mesh 中的 face 索引，也是如此。所有其他 tensors 将仅在第一个维度上进行拼接，而不会进一步增加其值。

然而，存在一些特殊用例（如下所述），用户需要根据自己的需求主动修改此行为。PyG 允许通过重写 `torch_geometric.data.Data.__inc__()` 和 `torch_geometric.data.Data.__cat_dim__()` 功能来修改底层的 batching 过程。在没有任何修改的情况下，它们在 `Data` 类中定义如下：

```python
def __inc__(self, key, value, *args, kwargs):
    if 'index' in key:
        return self.num_nodes
    else:
        return 0

def __cat_dim__(self, key, value, *args, kwargs):
    if 'index' in key:
        return 1
    else:
        return 0
```

我们可以看到 `__inc__()` 定义了两个连续图属性之间的增量计数。默认情况下，只要属性名称包含子字符串 `index`（出于历史原因），PyG 就会将属性增加节点数量 `num_nodes`，这对于 `edge_index` 或 `node_index` 等属性非常方便。但是请注意，这可能会导致属性名称包含子字符串 `index` 但不应增加的属性出现意外行为。为确保正确，最佳实践是始终仔细检查 batching 的输出。此外，`__cat_dim__()` 定义了相同属性的图 tensors 应在哪个维度上进行拼接。这两个函数都会为存储在 `Data` 类中的每个属性调用，并将它们的特定 `key` 和 `value` 项作为参数传递。

接下来，我们将介绍一些可能绝对需要修改 `__inc__()` 和 `__cat_dim__()` 的用例。

### 图对（Pairs of Graphs）

如果您想在单个 `Data` object 中存储多个图，例如用于图匹配（graph matching）等应用，则需要确保所有这些图的 batching 行为正确。例如，考虑在 `Data` 中存储两个图：一个源图 $G_s$ 和一个目标图 $G_t$，例如：

```python
from torch_geometric.data import Data

class PairData(Data):
    pass

data = PairData(x_s=x_s, edge_index_s=edge_index_s,  # Source graph.
                x_t=x_t, edge_index_t=edge_index_t)  # Target graph.
```

在这种情况下，`edge_index_s` 应该增加源图 $G_s$ 中的节点数，例如 `x_s.size(0)`，而 `edge_index_t` 应该增加目标图 $G_t$ 中的节点数，例如 `x_t.size(0)`：

```python
class PairData(Data):
    def __inc__(self, key, value, *args, kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, kwargs)
```

我们可以通过设置一个简单的测试脚本来测试我们的 `PairData` batching 行为：

```python
from torch_geometric.loader import DataLoader
import torch

x_s = torch.randn(5, 16)  # 5 nodes.
edge_index_s = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])

x_t = torch.randn(4, 16)  # 4 nodes.
edge_index_t = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])

data = PairData(x_s=x_s, edge_index_s=edge_index_s,
                x_t=x_t, edge_index_t=edge_index_t)

data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
# >>> PairDataBatch(x_s=[10, 16], edge_index_s=[2, 8], x_t=[8, 16], edge_index_t=[2, 6])
print(batch.edge_index_s)
# >>> tensor([[0, 0, 0, 0, 5, 5, 5, 5],
#             [1, 2, 3, 4, 6, 7, 8, 9]])
print(batch.edge_index_t)
# >>> tensor([[0, 0, 0, 4, 4, 4],
#             [1, 2, 3, 5, 6, 7]])
```

到目前为止一切顺利！即使 $G_s$ 和 $G_t$ 使用不同数量的节点，`edge_index_s` 和 `edge_index_t` 也能正确地 batch 到一起。然而，`batch` 属性（将每个节点映射到其各自的图）丢失了，因为 PyG 无法识别 `PairData` object 中的实际图。这就是 `DataLoader` 的 `follow_batch` 参数发挥作用的地方。在这里，我们可以指定要为哪些属性维护 batch 信息：

```python
loader = DataLoader(data_list, batch_size=2, follow_batch=['x_s', 'x_t'])
batch = next(iter(loader))

print(batch)
# >>> PairDataBatch(x_s=[10, 16], edge_index_s=[2, 8], x_s_batch=[10], x_t=[8, 16], edge_index_t=[2, 6], x_t_batch=[8])
print(batch.x_s_batch)
# >>> tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
print(batch.x_t_batch)
# >>> tensor([0, 0, 0, 0, 1, 1, 1, 1])
```

正如所见，`follow_batch=['x_s', 'x_t']` 现在成功地为节点特征 `x_s` 和 `x_t` 分别创建了分配向量 `x_s_batch` 和 `x_t_batch`。现在可以使用这些信息在单个 `Batch` object 中对多个图执行 reduce 操作，例如 global pooling。

### 二分图（Bipartite Graphs）

bipartite graph 的 adjacency matrix 定义了两种不同节点类型之间的关系。通常，每种节点类型的节点数量不必匹配，从而导致一个形状为 $A \in \{0, 1\}^{N \times M}$ 且可能 $N \ne M$ 的非方阵 adjacency matrix。在 bipartite graphs 的 mini-batching 过程中，`edge_index` 中边的源节点应以不同于 `edge_index` 中边的目标节点的方式增加。为了实现这一点，考虑一个介于两种节点类型之间的 bipartite graph，分别具有相应的节点特征 `x_s` 和 `x_t`：

```python
from torch_geometric.data import Data

class BipartiteData(Data):
    pass

data = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
```

对于 bipartite graphs 中正确的 mini-batching 过程，我们需要告诉 PyG 它应该独立地增加 `edge_index` 中边的源节点和目标节点：

```python
class BipartiteData(Data):
    def __inc__(self, key, value, *args, kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, kwargs)
```

在这里，`edge_index[0]`（边的源节点）增加了 `x_s.size(0)`，而 `edge_index[1]`（边的目标节点）增加了 `x_t.size(0)`。我们可以再次通过运行一个简单的测试脚本来测试我们的实现：

```python
from torch_geometric.loader import DataLoader
import torch

x_s = torch.randn(2, 16)  # 2 nodes.
x_t = torch.randn(3, 16)  # 3 nodes.

edge_index = torch.tensor([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
])

data = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
# >>> BipartiteDataBatch(x_s=[4, 16], x_t=[6, 16], edge_index=[2, 8])
print(batch.edge_index)
# >>> tensor([[0, 0, 1, 1, 2, 2, 3, 3],
#             [0, 1, 1, 2, 3, 4, 4, 5]])
```

再次，这正是我们想要的行为！

### 沿新维度进行 Batching（Batching Along New Dimensions）

有时，`Data object` 的属性应该通过获得一个新的 batch dimension 来进行 batching（如经典 mini-batching 中那样），例如对于图级别的属性或目标。具体来说，形状为 `[num_features]` 的属性列表应作为 `[num_examples, num_features]` 返回，而不是 `[num_examples * num_features]`。PyG 通过在 `__cat_dim__()` 中返回一个 `None` 的拼接维度来实现这一点：

```python
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class MyData(Data):
    def __cat_dim__(self, key, value, *args, kwargs):
        if key == 'foo':
            return None
        return super().__cat_dim__(key, value, *args, kwargs)

edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1],
])

foo = torch.randn(16)
data = MyData(num_nodes=3, edge_index=edge_index, foo=foo)
data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
# >>> MyDataBatch(num_nodes=6, edge_index=[2, 8], foo=[2, 16])
```

如预期，`batch.foo` 现在由两个维度描述：batch dimension 和 feature dimension。