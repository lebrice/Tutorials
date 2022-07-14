from typing_extensions import TypeGuard


def is_str_list(l: list) -> TypeGuard[list[str]]:
    return all(isinstance(v, str) for v in l)


def foo(l: list):
    if is_str_list(l):
        print(l)
    else:
        print(l)
