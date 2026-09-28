"""Stable fingerprints of callables, so the cache can tell an edited eval (or an
edited ``run``) from the one that produced a cached result.

The cache used to key an eval outcome on the eval's LABEL. Editing
``checks.contains("X1")`` to ``checks.contains("ZZZ")`` kept the label, so a
cached run served the OLD verdicts: a baseline that should fail came back valid.
Two evals sharing a label (``--check contains:8080 --check contains:BANANA``)
shared one cached outcome.

A fingerprint hashes what a callable actually does: its code (recursively,
including nested functions), defaults, closure values, the simple module
globals it reads, and — for callable objects such as a metric wrapper — its
type and attribute values. Two design rules:

* **Prefer a miss over a collision.** When something can't be summarized
  exactly (an opaque object), its type plus an address-stripped ``repr`` is fed
  in, so a change usually changes the fingerprint. A miss costs an API call; a
  collision serves a wrong verdict.
* **Stable when nothing changed** (across processes on the same Python), so a
  re-run still hits: memory addresses are stripped, sets and dicts are ordered.

Set ``cache_version`` on an eval (any string) to force a new fingerprint when
it depends on something muteval can't see (a file it reads, a remote rubric).
"""

from __future__ import annotations

import functools
import hashlib
import re
import types
from typing import Any

_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_MAX_DEPTH = 6
_MAX_REPR = 2000


def fingerprint(obj: Any) -> str:
    """A short, stable hash of ``obj``'s behavior-relevant content."""
    h = hashlib.sha256()
    _feed(h, obj, 0, set())
    return h.hexdigest()[:24]


def _put(h: "hashlib._Hash", tag: str, text: str = "") -> None:
    h.update(tag.encode("utf-8"))
    h.update(b"\x1f")
    h.update(text.encode("utf-8", "backslashreplace"))
    h.update(b"\x1e")


def _opaque(h: "hashlib._Hash", obj: Any) -> None:
    try:
        text = repr(obj)
    except Exception:  # noqa: BLE001 - a broken __repr__ still gets a type tag
        text = "<unrepresentable>"
    _put(h, "repr", type(obj).__qualname__ + ":" + _ADDR.sub("0x", text)[:_MAX_REPR])


def _feed(h: "hashlib._Hash", obj: Any, depth: int, seen: set) -> None:
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes)):
        _put(h, "v", repr(obj))
        return
    if depth > _MAX_DEPTH:
        _opaque(h, obj)
        return
    oid = id(obj)
    if oid in seen:
        _put(h, "cycle", type(obj).__qualname__)
        return
    seen.add(oid)
    try:
        version = getattr(obj, "cache_version", None)
        if isinstance(version, str):
            _put(h, "cache_version", version)
        if isinstance(obj, (list, tuple)):
            _put(h, type(obj).__name__, str(len(obj)))
            for item in obj:
                _feed(h, item, depth + 1, seen)
        elif isinstance(obj, (set, frozenset)):
            _put(h, "set", str(len(obj)))
            for item in sorted(obj, key=repr):
                _feed(h, item, depth + 1, seen)
        elif isinstance(obj, dict):
            _put(h, "dict", str(len(obj)))
            for k in sorted(obj, key=repr):
                _feed(h, k, depth + 1, seen)
                _feed(h, obj[k], depth + 1, seen)
        elif isinstance(obj, types.CodeType):
            _feed_code(h, obj, depth, seen)
        elif isinstance(obj, functools.partial):
            _put(h, "partial")
            _feed(h, obj.func, depth + 1, seen)
            _feed(h, obj.args, depth + 1, seen)
            _feed(h, obj.keywords, depth + 1, seen)
        elif isinstance(obj, types.MethodType):
            _put(h, "method")
            _feed(h, obj.__func__, depth + 1, seen)
            _feed(h, obj.__self__, depth + 1, seen)
        elif isinstance(obj, types.FunctionType):
            _feed_function(h, obj, depth, seen)
        elif isinstance(obj, (types.BuiltinFunctionType, types.ModuleType, type)):
            _put(h, "named", getattr(obj, "__module__", "") or "")
            _put(
                h, "qualname", getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
            )
        elif callable(obj) or hasattr(obj, "__dict__"):
            # A callable object (a metric wrapper, a class-based eval) or plain
            # object held in a closure: its type, its __call__ code, its state.
            cls = type(obj)
            _put(h, "object", f"{cls.__module__}.{cls.__qualname__}")
            # The class's own (or inherited) __call__ FUNCTION, looked up on the
            # MRO so we get the code object, not a bound method.
            call = next(
                (k.__dict__["__call__"] for k in cls.__mro__ if "__call__" in k.__dict__),
                None,
            )
            if isinstance(call, types.FunctionType):
                _feed(h, call, depth + 1, seen)
            state = getattr(obj, "__dict__", None)
            if isinstance(state, dict):
                _feed(h, state, depth + 1, seen)
            else:
                _opaque(h, obj)
        else:
            _opaque(h, obj)
    finally:
        seen.discard(oid)


def _feed_code(h: "hashlib._Hash", code: types.CodeType, depth: int, seen: set) -> None:
    _put(h, "code", code.co_name)
    h.update(code.co_code)
    _put(h, "names", ",".join(code.co_names))
    for const in code.co_consts:
        _feed(h, const, depth + 1, seen)


def _feed_function(
    h: "hashlib._Hash", fn: types.FunctionType, depth: int, seen: set
) -> None:
    _put(h, "fn", f"{fn.__module__}.{fn.__qualname__}")
    _feed_code(h, fn.__code__, depth, seen)
    _feed(h, fn.__defaults__, depth + 1, seen)
    _feed(h, fn.__kwdefaults__, depth + 1, seen)
    for cell in fn.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:  # an empty cell
            _put(h, "cell", "<empty>")
            continue
        _feed(h, value, depth + 1, seen)
    # Module-level globals the code reads: simple values (THRESHOLD = 0.7) and
    # helper functions change behavior without changing this function's code.
    globs = fn.__globals__
    for name in _all_names(fn.__code__):
        if name in globs:
            value = globs[name]
            if isinstance(value, types.ModuleType):
                _put(h, "module", name)
            elif isinstance(value, type) or isinstance(value, types.BuiltinFunctionType):
                _put(h, "global", name)
            else:
                _put(h, "global", name)
                _feed(h, value, depth + 1, seen)


def _all_names(code: types.CodeType) -> list:
    names = list(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names.extend(_all_names(const))
    return sorted(set(names))
