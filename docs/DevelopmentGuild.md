# Development Guild

## Overview

This document outlines key conventions and best practices for developers contributing to **MedVoiceQAReasonDataset**. It covers:

1. **Package Structure & Exports**
2. **LangGraph Integration**

---

## 1  Package Structure & Exports

* **Use `__init__.py` only for exports**

  * Every Python package directory (e.g., `pipeline/`, `nodes/loader/`, etc.) **must** contain an `__init__.py` that:

    1. Imports and exposes the public API of that module.
    2. Does **not** contain business logic or function definitions.
  * Example:

    ```python
    # nodes/loader/__init__.py
    from .loader import run_loader, LoaderConfig

    __all__ = ["run_loader", "LoaderConfig"]
    ```
  * All internal functions and classes live in submodules (e.g., `loader.py`) and are **not** imported automatically.

* **Keep modules small and focused**

  * One class or one related function group per `.py` file.
  * Avoid circular imports by only referencing sibling modules in functions, not in the package-level scope.
