---
title: Python 类型体操训练（二）-- 中级篇
id: 38
date: 2023-12-10 12:57:00
description: python typing tutorial 中级篇。这篇文章介绍了 Python Class Variable 可以使用 ClassVar 定义一个只能由 Class 修改的类变量，并且介绍了 Self 类型；此外，着重了解 TypedDict 如何定义特定 Key 的字典类型，了解 Required 和 NotRequired 的区别；然后介绍 Python Generic Type（泛型）的写法，明确了 Python 3.12 之后方括号 [T] 注释的写法，以及如何在 3.12 版本以前通过 TypeVar 定义通用类型；最后介绍了 Literal 和 Callable 两个重要且基础的 Python 类型。
category: python-type-challenge
tag:
  - python-typing-tutorial
  - python-type-tutorial
permalink: /post/python-type-challenge-intermediate.html
publish: true
---

## 阅读提示

- 面向读者群体
  - 有一定Python基础，需要进阶开发中大型项目
  - 有其他静态类型语言开发经验的人，需要快速了解 Python 类型注释（type hint）
  - 如果没有太多基础，可以先阅读 [Python 类型体操训练（一）-- 基础篇](/post/python-type-challenge-basic.html)
- 你能学到什么？
  - Python **类变量**如何写类型注释（type hint）？
  - Python **字典类型**如何写类型注释？
  - Python **通用类型（Generic）** 如何写类型注释？
  - Python 类型中的一些高级关键字（Literal, Callable）？
- 结论
  - 完成这篇文章的阅读，基本上已经可以适应 Python 日常项目的开发需求
  - 强烈推荐自己进行[类型训练](https://github.com/laike9m/Python-Type-Challenges)

这篇文章按照 [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges)[^1]库的划分，一共分为四个部分。

- [Python 类型体操训练（一）-- 基础篇](/post/python-type-challenge-basic.html)
- [Python 类型体操训练（二）-- 中级篇](/post/python-type-challenge-intermediate.html) （**本篇文章**）
- [Python 类型体操训练（三）-- 高级篇](/post/python-type-challenge-advanced.html)
- [Python 类型体操训练（四）-- 究极篇] TODO
<!-- -
- [Python 类型体操训练（四）-- 究极篇](/post/python-type-challenge-extreme.html) -->

## 类相关类型

### 基础类类型

在 Python 里面，任何一个类本质上都是对象，因此无论是 Python 内部类还是自定义类，都需要进行类型注释。

- Example 1, Python 内部类

```python
# 以 Python pathlib 内部类 为例
from pathlib import Path

def get_cur_path() -> Path:
    cur_path: Path = Path('.')
    return cur_path

pwd_path = get_cur_path()

# 这里 pwd_path 的类型可能是 pathlib.PosixPath 或者 pathlib.WindowsPath, 但是两者都是 Path 的子类。

# 注意：int/str/float等也是内部的类
```

- Example 2，自定义类

```python
class Person:
    name: str
    age: int

class FakePerson:
    name: str
    age: int

def print_user_info(user: Person) -> None:
    print(user.name, str(user.age))

user1 = Person('chaofa', '28')
user2 = FakePerson('chaofa', '25')

print_user_info(user1)   # 通过检查
print_user_info(user2)   # 无法通过检查（失败）
```

### 类变量 (ClassVar)

在 Python 内部，`class` 属性分为**类变量**和成员变量，类变量可以同时被类和实例反问，因此类变量可能被实例修改，但一般情况下，我们不希望实例修改类变量的值，

```python
from typing import ClassVar

class Person:
    # 类变量，只能被类修改，不能被实例修改
    name: ClassVar[str] = "chaofa"
    # 实例变量，instance variable，可以被实例修改
    age: int = 28

p = Person()   # Person 类的实例

Person.name = "bbruceyuan"  # ok, 成功
p.age = 25                  # ok, 成功，instace var 可以被实例修改
p.name = "other name"       # error, 失败（因为这是一个类变量，不能被实例修改）
```

### Self 类型

`self` 在 Python 里面是一个特殊的名字，尽管可以写成其他的名字，但是一般会按照约定把 Python Class Method 里面的第一个参数设置为 `self`。

- 用法 1： 希望 current class 和 sub class 返回不同的类型

```python
from typing import Self, reveal_type

class Foo:
    def return_self(self) -> Self:
        ...
        return self

class SubclassOfFoo(Foo):
    pass

reveal_type(Foo().return_self())  # 最终类型是 "Foo"
reveal_type(SubclassOfFoo().return_self())  # 最终类型是 "SubclassOfFoo"

```

- 用法 2： 希望 current class 和 sub class 返回**相同的类型**

```python
from typing import TypeVar

Self = TypeVar("Self", bound="Foo")

class Foo:
    def return_self(self: Self) -> Self:
        ...
        return self

class SubclassOfFoo(Foo):
    pass

reveal_type(Foo().return_self())  # 最终类型是 "Foo"
reveal_type(SubclassOfFoo().return_self())  # !!!!!最终类型是 "Foo"
```

- 用法3： 希望 current class 和 sub class 返回**相同的类型**

```python
# 用法 3 和 用法 2 使用场景稍微有一点不同
# 用法 2 中，返回的是 当前 class
# 用法 3 中，返回的一直是 Foo 的实例

class Foo:
    def return_self(self) -> 'Foo':
        ...
        return Foo()

class SubclassOfFoo(Foo):
    pass

reveal_type(Foo().return_self())  # 最终类型是 "Foo"
reveal_type(SubclassOfFoo().return_self())  # !!!!!最终类型是 "Foo"
```

## 字典类型（TypedDict）

在上一篇文章 [Python 类型体操训练（一）-- 基础篇](/post/python-type-challenge-basic.html)，介绍了 `dict[key_type, value_type]`，定义一个字典，拥有特定的 `key_type` 和 `value_type`，这个字典可以拥有无数的 `key`。`TypedDict` 是为了定义【**拥有特定 key**】的字典类型，`key` 的数量是确定的。

### TypedDict-基础用法

基础定义，定义一个字典类型，叫做 `Programer`（程序员），有三个 `key`，分别是 `name`, `age`, `github` 分别是 `str`, `int`, `str`类型。

```python
from typing import TypedDict

class Programer(TypedDict):
    name: str
    age: int
    github: str

# a 正确，所有类型匹配
a: Programer = {"name": "chaofa", "age": 28, "github": "github.com/bbruceyuan"}
# b 错误，因为 age 类型不匹配
b: Programer = {"name": "bbruceyuan", "age": 12.3, "github": "github.com/bbruceyuan"}
# c 错误，因为缺少 key github
c: Programer = {"name": "chaofa", "age": 12}
# d 错误，因为缺少 key age
d: Programer = {"name": 'bruce', "github": "github.com/bbruceyuan"}

# 用法场景
# 一般用于 固定 key 的**字典**，而且明确是一个字典类型
# 现在有 dataclasses 之后，我个人觉得用一个 dataclass 会是更好的选择
```

### TypedDict-NotRequired

还是上面这个例子，想定义一个程序员类（Programer），但是并不是每一个程序员都有自己的 `Github` 账号，这个时候 `github` 就不是一个必填的 key。这时候就可以使用 `NotRequired` 关键字。

```python
from typing import TypedDict, NotRequired

class Programer(TypedDict):
    name: str
    age: int
    github: NotRequired[str]

# a 正确，所有类型匹配
a: Programer = {"name": "chaofa", "age": 28, "github": "github.com/bbruceyuan"}
# b 错误，因为 age 类型不匹配
b: Programer = {"name": "bbruceyuan", "age": 12.3, "github": "bruceyuan.com"}
# c **正确**，因为 key github 是可选 key
c: Programer = {"name": "chaofa", "age": 12}
# d 错误，因为缺少 key age
d: Programer = {"name": 'bruce', "github": "github.com/bbruceyuan"}

```

### TypedDict-Required

还是上面这个例子，想定义一个程序员类（Programer），有 `name`, `age`, `github`, `address`, `email` 这些 `Key`, 但只有 `name` 是必须得，其他的都属于隐私不想公开，如果按照上面 `NotRequired` 的方式，就需要写很多 `NotRequired`。因此可以使用 `Required` 替代，具体见 case。

```python
from typing import TypedDict, Required

class Programer(TypedDict, total=False):
    name: Required[str]
    age: int
    github: str
    address: str
    email: str

# a 正确，所有类型匹配
a: Programer = {"name": "chaofa", "age": 28, "github": "github.com/bbruceyuan", 'address': 'address', 'email': 'email'}
# b 错误，因为 age 类型不匹配
b: Programer = {"name": "bbruceyuan", "age": 12.3, "github": "bruceyuan.com"}
# c **正确**，因为 其他 key 都是可选的
c: Programer = {"name": "chaofa"}
# d **错误**，因为缺少 name key, 缺少了一定需要的 name key
d: Programer = {"age": 28, "github": "github.com/bbruceyuan", 'address': 'address', 'email': 'email'}
```

### TypedDict-继承

`TypeDict` 定义的类可以和普通的 `class` 一样，通过继承来实现组合类型。

```python
from typing import TypedDict

class Programer(TypedDict):
    name: str
    age: int

class GoogleProgramer(Programer):
    work_base: str   # 工作地 BASE

# a 正确，所有类型匹配
a: GoogleProgramer = {"name": "chaofa", "age": 28, "work_base": "china"}
# b 错误，因为 缺少 work_base 这个 key
b: Programer = {"name": "bbruceyuan", "age": 25}
```

## 通用类型（Generic）

写过 `C++/Java`的同学可能知道 泛型的概念，一般会用一个 `T` 来表示这个变量可能是任意类型。`C++`语法结构为：`template<class T> void func(T var) {...}` ，`Java` 语法结构为 `public class Hello<T> {...}`。

而 [Python 的写法](https://docs.python.org/3/library/typing.html#typing.Generic)和 [Scala 语言的泛型](https://docs.scala-lang.org/zh-cn/tour/generic-classes.html)更为接近，语法是几乎是一样的，用 `[T]` 来表示泛型，方括号 `[]` 是用来接收泛型参数，`T` 是一个通用的参数标识符。

以下为 泛型参数的基本语法讲解，更高级用法见下一篇文章[Python 类型体操训练（三）-- 高级篇](/post/python-type-challenge-advanced.html)。

### 推荐写法 --方括号语法 (Python >= 3.12)

- 例子 1，定义一个函数，**输入和输出都是一个类型**，不指定具体类型

```python
# !!! good case （推荐）
def foo1[T](a: T, b: T) -> T:
    ...

# bad case (不推荐，实际上就是错的)
def foo2(a: Any, b: T) -> Any:
    ...

# 在这个例子看起来 T 有点像 Any 的用法
# foo1 表示 参数 和 返回值 的一样的类型
# foo2 参数 和 返回值 可能是不一样的类型
```

- 例子 2， 如果需要输入和输出都是一个类型，这个类型就是 `str` 类型，那么可以写成

```python
def foo1[T: str](a: T, b: T) -> T:
    ...
# 表示 T 可以是 str
```

- 例子 3，定义一个函数，输入和输出都是一个类型，这个类型只能是 `int` 或者 `str`

```python
def foo1[T: (int, str)](a: T, b: T) -> T:
    ...
# 表示 T 可以是 int or str
```

- 例子 4，如果 T 需要是一个函数呢？

```python
from collections.abc import Callable

def decorator[T: Callable](func: T) -> T:
    ...
# 表示 T 可以是 函数, 这也是定义装饰器的方法
```

- 例子 5， 类中使用 泛型，比如 stack 可以接受任意类型

```python
class Stack[T]:
    def __init__(self) -> None:
        self.items: list[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self):
        return self.items.pop()
```

### Python 3.12 之前的写法

在 Python 3.12 之前，还不支持方括号 `[T]` 语法，因此需要使用一个叫做 `TypeVar` 的函数定义泛型，具体见[链接](https://docs.python.org/3/library/typing.html#typing.TypeVar)。

- 例子 1

```python
from typing import TypeVar

T = TypeVar("T")

def foo1[T](a: T, b: T) -> T:
    ...
```

- 例子 2

```python
from typing import TypeVar

# 表示可以是 str 的 subclass
T = TypeVar("T", bound=str)

def foo1[T](a: T, b: T) -> T:
    ...
```

- 例子 3

```python
from typing import TypeVar

# 表示 T 可以是 int or str
T = TypeVar("T", str, int)

def foo1[T: (int, str)](a: T, b: T) -> T:
    ...
```

- 例子 4

```python
from collections.abc import Callable
from typing import TypeVar

# 表示 T 可以是 函数, 这也是定义装饰器的方法
T = TypeVar("T", bound=Callable)

def decorator[T: Callable](func: T) -> T:
    ...
```

- 例子 5

```python
from typing import TypeVar

T = TypeVar("T")

class Stack:
    def __init__(self) -> None:
        self.items: list[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self):
        return self.items.pop()
```

## 其他

### Literal

`Literal` 是字面变量的意思，表示【只能是字面的取值】。

```python
# 假设有一条路，只能往左或者往右，那么 direction 只能取值为：`left` 或者 `right`
from typing import Literal

def go(direction: Literal['left', 'right']):
    ...

# Literal 就是把所有可能得候选都写上，一般用于候选比较少的情况
# 比如 SQL JOIN 只能取值 inner, left, right, full 等
```

### Callable

`Callable` 表示一个可调用对象，一般是一个函数。具体用法为：
`Callable[[函数参数1, 函数参数2, ..., 函数参数n], 函数返回值]`

```python
from collections.abc import Callable

def foo1(name: str) -> None:
    print(name)

def foo2(name: int) -> None
    print(name)

def accept_a_func(func: Callable[[str], None]) -> None:
    name = 'chaofa'
    func(name)

accept_a_func(foo1)   # 成功，符合 Callable 类型定义
accpet_a_func(foo2)   # 失败，因为 Callable 定义了，func 参数应该接受一个 str 类型
```

## 小结

通过阅读这一篇文章，可以知道 Python Class Variable 可以使用 `ClassVar` 定一个一个只能由 `Class` 修改的类变量，并且介绍了 `Self` 类型（这里已经看到了 前向注释的影子，具体可以参考[下一篇](/post/python-type-challenge-advanced.html)）；此外，着重了解 `TypedDict` 如何定义特定 `Key` 的字典类型，了解 `Required` 和 `NotRequired` 的区别；然后介绍 Python Generic Type（泛型）的写法，明确了 Python 3.12 之后方括号 `[T]` 注释的写法，以及如何在 3.12 版本以前通过 `TypeVar` 定义通用类型；最后介绍了 `Literal` 和 `Callable` 两个重要且基础的 Python 类型。

## Reference

[^1]. [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges)
