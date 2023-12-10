---
title: Python 类型体操训练（三）-- 高级篇
id: 39
date: 2023-12-11 14:54:00
description: python typing tutorial 高级篇。这篇文章介绍了 Python 类型的一些高级用法，包括 protocol, override, overload, forwardref, generator... 通过实际案例解释了每一种类型的使用场景和使用建议。
category: python-type-challenge
tag:
  - python-typing-tutorial
  - python-type-tutorial
permalink: /post/python-type-challenge-advanced.html
publish: true
---
## 阅读提示
- 面向读者群体
	- 有一定Python基础，需要进阶开发中大型项目
	- 有其他静态类型语言开发经验的人，需要快速了解 Python 类型注释（type hint）
	- 如果没有太多基础，可以**先阅读前两篇文章**
- 你能学到什么？
	- Python 如何定义 `protocol`
	- Python 如何重载类方法和函数签名
	- Python 前向推导、生成器、Nerver等类型的使用
	- ...
	- 推荐自己完成 [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges) 上面的练习。

这篇文章按照 [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges)[1]库的划分，一共分为四个部分。
- [Python 类型体操训练（一）-- 基础篇](/post/python-type-challenge-basic.html)
- [Python 类型体操训练（二）-- 中级篇](/post/python-type-challenge-intermediate.html) 
- [Python 类型体操训练（三）-- 高级篇](/python-type-challenge-advanced.html)（**本篇文章**）
- [Python 类型体操训练（四）-- 究极篇]  博主自己暂时还没学会
<!-- - 
- [Python 类型体操训练（四）-- 究极篇](challenge/post/python-type-challenge-extreme.html) -->

## Python Type 高级类型
### Protocol - 协议
`Protocol` 定义方式有点像 `abc`类，表示这个类型下面有某些方法。

- 看例子学习，Duck 类下面有一个方法 `quack`
```python
from typing import Protocol

# SupportsQuack 是一个有 quack 方法的类型
class SupportsQuack(Protocol):
    def quack(self) -> None:
        ...

class Duck:
	def quack(self) -> None:
		print('quack!')

duck: SupportsQuack = Duck()  # 正确

class Dog:
	def bark(self) -> None:
	    print("bark!")
dog: SupportsQuack = Dog()   # 错误，因为 dog 类没有 `quack` 方法
```

### 重载 
#### override - 类方法重载
重载这个特性在其他语言里面是被大量使用的，表示子类需要重载父类的方法。直接看 
- 例子 
```python
class Animal:
	def say(self) -> str:
		return 'hello world'

class Dog(Animal):
	# 正确
	def say(self) -> str:
		return "bake bake!!"

class Duck(Animal):
	# 这里可能写错了方法的名字，类型检查器也不会报错
	def sey(self) -> str:
		return "quack quack!!"

animal1: Animal = Dog()
animal1.say()  # 返回 'bake bake!!'，对 say 方法进行重载了

animal2: Animal = Duck()
animal2.say()  # 返回 'hello world'，因为没有正确的对 say 重载
```

而现在有了 `override` 关键字之后，就不会发生上面的问题了
```python
class Animal:
	def say(self) -> str:
		return 'hello world'

class Dog(Animal):
	# 正确，重载 Animal.say 方法
	@override
	def say(self) -> str:
		return "bake bake!!"

class Duck(Animal):
	# !!!!这里会报错，因为 Animal 类里面没有 sey 方法
	@override
	def sey(self) -> str:
		return "quack quack!!"
```

#### overload -函数签名重载
这里的重载并不是真正的函数重载，因为**重载的时候并不需要做真正的实现**，而仅仅是重载签名。
- 下面的 snippet code 来自于 [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges/blob/main/challenges/advanced-overload/solution.py)
	- `process` 方法并没有真正的重写
	- `overload` 要在 `process` 实现之前

```python
from typing import overload

@overload
def process(response: None) -> None:
    ...

@overload
def process(response: int) -> tuple[int, str]:
    ...

@overload
def process(response: bytes) -> str:
    ...

def process(response: int | bytes | None) -> str | None | tuple[int, str]:
    ...


from typing import assert_type

assert_type(process(b"42"), str)
assert_type(process(42), tuple[int, str])
assert_type(process(None), None)

assert_type(process(42), str)  # expect-type-error
assert_type(process(None), str)  # expect-type-error
assert_type(process(b"42"), tuple[int, str])  # expect-type-error
assert_type(process(None), tuple[int, str])  # expect-type-error
assert_type(process(42), str)  # expect-type-error
assert_type(process(None), str)  # expect-type-error
```

### ForwardRef -前向推导类型
- Example 1, 我们使用一个类型的时候，可能这个类型还没有完成定义，但是我们又想定义内部的返回值。这个时候就需要使用前向推导，语法为「引号包裹变量名」，比如下面的 `copy` 方法返回 `"MyClass"` 。

```python
class MyClass:
    def __init__(self, x: int) -> None:
        self.x = x

    def copy(self) -> "MyClass":
        copied_object = MyClass(x=self.x)
        return copied_object

from typing import assert_type
inst = MyClass(x=1)
assert_type(inst.copy(), MyClass)  
# 这两个是同一个类型
# 前向推导一般使用 引号将类名 包裹起来，从而达到前向推导的目的。
```

- Example 2, 循环定义类型。定义一个 名叫 `Tree` 的字典，key 是 str, value 还是 `Tree`
```python
type Tree = dict[str, "Tree"]
```

### Generator - 生成器
用法: `Generator[YieldType, SendType, ReturnType]`，详情见例子
```python
def echo_round() -> Generator[int, float, str]:
    sent = yield 0
    while sent >= 0:
        sent = yield round(sent)
    return 'Done'
# 解释：
# yield 后都是返回 int 类型
# round 函数接受一个 float 类型
# 最终的 return 是 `Done`，类型是 str
```

### Never
这通常用于表示一个函数永远不会被调用或者一个函数没有返回值。
- Example 1, 永远不会被调用
```python
from typing import Never

def never_call_me(arg: Never) -> None:
    pass

def int_or_str(arg: int | str) -> None:
    never_call_me(arg)  # type checker error
    match arg:
        case int():
            print("It's an int")
        case str():
            print("It's a str")
        case _:
            never_call_me(arg)  # OK, arg is of type Never
```
- Example 2， 没有返回值
```python
from typing import Never

def stop() -> Never:
    raise RuntimeError("")

from typing import assert_never

assert_never(stop())
```

### TypeGuard
一般用于把 Python 类型缩窄。用 `TypeGuard` 定义会告诉类型检查器两个信息
- 返回值是一个布尔类型（boolean）
- 如果返回 `True` ，说明类型是 `TypeGuard` 内的类型。
```python
from typing import Any, TypeGuard

def is_string(value: Any) -> TypeGuard[str]:
    return isinstance(value, str)
```

### TupleVar
`Generic`（泛化）的高级用法，表示接受多个参数化泛化。
```python
def move_first_element_to_last[T, *Ts](tup: tuple[T, *Ts]) -> tuple[*Ts, T]:
    return (*tup[1:], tup[0])

# T 被绑定为 int, Ts 被绑定为 ()
# 最终类型是 tuple[int], 返回值为 (1, )
move_first_element_to_last(tup=(1,))

# T 被绑定为 int, Ts 被绑定为 (str, )
# 返回值为 ('spam', 1), 返回类型为 tuple[str, int]
move_first_element_to_last(tup=(1, 'spam'))

# T 绑定为 int, Ts 绑定为 (str, float)
# 返回值为 ('spam', 3.0, 1), 返回类型为 tuple[str, float, int]
move_first_element_to_last(tup=(1, 'spam', 3.0))

# 类型检查和运行都会出错，至少需要一个值
# tuple[()] 和 tuple[T, *Ts] 是不同的
move_first_element_to_last(tup=())
```
### ParamSpec
这也是 `Generic` 的高级用法，一般用于参数的传递。常用于**高阶函数的参数传递、修改**，比如 `decorator` 输入是一个函数，具体例子如下

```python
from collections.abc import Callable
import logging

def add_logging[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    '''A type-safe decorator to add logging to a function.'''
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        logging.info(f'{f.__name__} was called')
        return f(*args, **kwargs)
    return inner

@add_logging
def add_two(x: float, y: float) -> float:
    return x + y
```

如果没有 `ParamSpec` 就只能写成 `Callable[..., Any]`，这样的注释只能知道这是一个函数，不知道推断出函数的具体类型。

## Reference
- [1]. [Python-Type-Challenges](https://github.com/laike9m/Python-Type-Challenges)
