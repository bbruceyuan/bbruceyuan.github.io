---
title: Python 类型体操训练（一）-- 基础篇
id: 37
date: 2023-12-08 23:57:00
description: Python 类型体操训练（一）-- 基础篇，本篇文章介绍了 Python 基础类型、容器类型、 Python function 如何写类型注释，此外详细讲解了常见 Python Type 常见的关键字，包括 Union, Any, Optional, TypeAlias(type), NewType, Final，通过这 5 个最常用的关键字类型增强我们类型注释的表达能力。
category: 
  - python-type-challenge
permalink: /post/python-type-challge-basic.html
---

## 阅读提示
- 面向读者群体
    - 有一定Python基础，需要进阶开发中大型项目
    - 有其他静态类型语言开发经验的人，需要快速了解 Python 类型注释（type hint）
- 你能学到什么？
    - Python 基础变量如何写类型注释（type hint）？
    - Python 容器变量如何写类型注释？
    - Python 函数如何写类型注释？
    - Python 类型有哪些常见的关键词？
- 建议
    - 注释部分非常的重要，阅读过程中请关注代码注释部分

这篇文章按照 [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges)[1]库的划分，一共分为四个部分。
- [Python 类型体操训练（一）-- 基础篇](https://bbruceyuan.com/post/python-type-challge-basic.html) （**本篇文章**）
- [Python 类型体操训练（二）-- 中级篇](https://bbruceyuan.com/post/python-type-challge-intermediate.html)
- [Python 类型体操训练（三）-- 高级篇] TODO
- [Python 类型体操训练（四）-- 究极篇] TODO
<!-- 
- [Python 类型体操训练（三）-- 高级篇](https://bbruceyuan.com/post/python-type-challge-advanced.html)
- [Python 类型体操训练（四）-- 究极篇](https://bbruceyuan.com/post/python-type-challge-extreme.html) -->

## Python为什么要写类型
大家都知道 Python 语言的灵活性，能写出非常简单灵活的代码，似乎在 Python 里面不需要类型推断，但是代码的灵活性恰好是重构和维护的杀手。 **绝大部分开源的 Python package，都是写了类型注释**。因为代码复杂之后，不可避免的会出现各类错误，而类型提示就可以很好的提前暴露相关的问题，从而在一开始就把问题扼杀在摇篮之中。

!!!最重要的一点：有类型提示，**写代码更简单**（IDE 提示更智能），**降低使用函数的心智负担**。

强烈推荐亲手完成 laike9m 的 [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges)库里面的习题，提升对Python类型的了解。一般推荐大家完成 高级训练即可，究极训练非常的难，更推荐大家在日常实践中学习掌握。

> ! NOTE: 
> 因为语言在慢慢发展，为了让大家体验到更好更高级的内容，这个系列的文章和Repo 一样，基于 Python 3.12 进行介绍。
## Python 类型基础

在 Python 中有很多基础类型，主要包括 `int`, `float`, `str`, `bool`, `bytes`, `None`,  `list`, `tuple`, `dict`, `set`, `frozenset`，可以分为简单变量（simple variable）和容器变量（container variable）。

### 简单变量
简单变量指的是： `int`, `float`, `str`, `bool`, `bytes`, `None` 等类型变量

- 建议
    - 简单变量类型申明不要有压力，最好能写，不想写省略也没问题
    - 简单变量类型不写也可以很好的被 IDE 推断
    - 如果这个变量作为一个返回值，而 func 又没有定义类型，建议写上

```python
# good case (推荐)
a: int = 1
b: float = 1.2
c: str = 'hello chaofa'
d: bool = True
e: bytes = b'hello chaofa'
f: None = None  # 这种很少见到有人这么写
# f = None # it is also ok. 
# 注意：str 和 byte 的 区别 
# assert c.encode("utf-8") == e, "两者类型一样"

# bad case (不推荐)
a = 1
b = 1.2
c = 'hello chaofa'
d = True
e = b'hello chaofa'
# 尽管是申明一些简单的变量，也推荐写上变量申明
```

### 容器变量
容器变量指的是： `list`, `tuple`, `dict`, `set` 等

- 建议
    - 容器变量强烈建议写上类型
    - 这样可以让代码更可读

```python
# good case (推荐)
int_arr: list[int] = [1, 2, 3, 4]
str_arr: list[str] = ['h', 'e', 'l', 'l', 'o']
float_arr: list[float] = [1.2, 1.3, 3.14]

str_set: set[str] = {'chaofa', 'bbruceyuan'}
int_set: set[int] = {1, 3}

two_value_tuple: tuple[int, float] = (1, 3.14)
three_value_tuple: tuple[int, float, str] = (1, 3.14, 'PI')

# 表示这是一个 str -> str 的 dict
url_map: dict[str, str] = {"chaofa": "bbruceyuan.com"}
embedding_lookup: dict[str, list[float]] = {"chaofa": [1.2, 3.4, 5.6]}

# bad case (不推荐)
from typing import List, Set, Tuple, Dict

int_arr: List[int] = [1, 2, 3, 4]
str_arr: List[str] = ['h', 'e', 'l', 'l', 'o']
float_arr: List[float] = [1.2, 1.3, 3.14]

str_set: List[str] = {'chaofa', 'bbruceyuan'}
int_set: List[int] = {1, 3}

two_value_tuple: Tuple[int, float] = (1, 3.14)
three_value_tuple: Tuple[int, float, str] = (1, 3.14, 'PI')

# 表示这是一个 str -> str 的 dict
url_map: Dict[str, str] = {"chaofa": "bbruceyuan.com"}
embedding_lookup: Dict[str, list[float]] = {"chaofa": [1.2, 3.4, 5.6]}

# 从python3.9 开始， List, Set, Tuple, Dict 等内置类型 支持小写表示，见 good case
# List/Set/Tuple/Dict等计划在 3.14 将被标记为 deprecated. 未来一定会被移除
# 所以我们尽量不要使用它，尊重一个语言发展的过程。
# 详情见：https://peps.python.org/pep-0585/

```

### 函数使用
Python 写类型注释，更重要的使用场景是函数。当你写一个函数（类）的时候，说明你需要抽象一些东西，意味着你的场景更复杂。

- 建议
    - 建议写清楚每一个函数的 入参，类型返回值
    - 这样有助于后续自己 debug。
    - 请相信：绝大部分时候我们不需要考虑输入可能是多种类型，所以前期我觉得可以勇敢的写上类型。

```python
# good case (推荐)
def foo(a: int, b: str) -> tuple[str, int]:
    return (b, a)

# bad case (不推荐)
from typing import Any

def foo(a: Any, b: Any) -> Any:
    return (b, a)

# Any 在 typing 表示任意类型，上面这种写法，写了等于没写
```

## Type 常见关键词

### Union
`Union` 单词的意思很好理解，表示`联合, 合并`。在 Python 集合操作中，表示把两个集合放在一起，两者都保留。在 `typing` 中意义非常的接近，使用 `Union` 表示两个类型都可以。一般用法为： `a: Union[int, str]`

```python
# good case 1
from typing import Union
a: Union[int, str] = 3
b: Union[int, str] = 'chaofa'

# good case 2 （推荐）
# 在 python3.10 之后， Union 可以通过 | 代替
a: int | str = 3
b: int | str = 'chaofa'

# !!! 强烈推荐这种写法，很直观，而且 TS 也是这样的写法
```

### Optional
`Optional` 表示 `可选的`，`Optional[X]` 和 `X | None` (`Union[X, None]`) 是一样的意思，表示这个变量可能是 `None` 或者是一个 `X` 类型。

```python
# 语法解读
a: Optional[str] = None
b: str | None = None
# a / b 两个变量申明的类型是一样的

# !!!注意：一般用于 设置默认值
```

### TypeAlias (type)
`TypeAlias` 表示对某个类型创建一个**别名**，类型没有发生变化，一般是为了方便自己理解。

```python
# 假设要创建一个 Vector 类型，是一个 只有 float 类型的数组

# bad case (不推荐)
Vector = list[float]
# 虽然语法支持，但是不推荐这么做，看上去不是很直观

# good case 1
from typing import TypeAlias

Vector: TypeAlias = list[float]

# good case 2 （推荐）
type Vector = list[float]

# Python 3.12 语法心支持 type 定义类型，非常的直观，清晰，比 good case1 又简单
```

### NewType
`NewType` 表示创建一个新的类型，这个类型和原有的类型不是同一个类型了。
> `TypeAlias` 或者 `type` 标记的类型只是创建一个别名，`NewType` 是直接得到一个新的变量

```python
from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(524313)

def get_user_name(user_id: UserId) -> str:
    ...

# 可以通过测试
user_a = get_user_name(UserId(42351))

# 不可以通过测试，因为 UserId 是一个新的类型，已经不是 int 了
user_b = get_user_name(-1)
```


### Final
`Final` 表示这个变量不能再被重新赋值(assign)了。

```python
# 用法
from typing import Final

my_list: Final = []
my_list.append(1) # 成功
my_list = []      # 失败，因为这里重新对 my_list 进行了赋值
```


## 小结
通过上面的学习，我们知道了 Python 基础类型、容器类型怎么写类型注释，也知道怎么在一个 Python function 里面加上类型注释，通过一些简单的类型标记就可以减少我们函数依赖错误，提前发现代码问题。最后我们还额外了解一些常见 Python Type 常见的关键字，包括 `Union`, `Optional`, `TypeAlias(type)`, `NewType`, `Final`，通过这 5 个最常用的关键字类型增强我们类型注释的表达能力。

可以说阅读完本节内容，我们就可以非常轻易地在工作学习中用上，而且一定会极大的减少一些因为粗心带来的错误。


>  `TypeDict`, `Literal`, `Generic` 等更高级的关键字将在 中级教程 介绍。

## Reference
- [1]. [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges)
- [2]. [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)
