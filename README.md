## Reference
1. [micrograd](https://github.com/karpathy/micrograd/tree/master)
2. [micrograd-c](https://github.com/KhawajaAbaid/micrograd_c)

## Quick start

```bash
mkdir build
cd build
cmake ..
make demo
../bin/demo
```

## Dev env

```bash
pip install pre-commit
pre-commit install 
# this will install a pre-commit hook in .git/hooks, then using clang-format to check your code style
```

If you want to test if pre-commit work
```
git commit
```

If not work
```
pre-commit uninstall
pre-commit install
```

or contact me: zincsabian@gmail.com
