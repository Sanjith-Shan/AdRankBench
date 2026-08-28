"""Optional NVTX range annotation, for making an Nsight Systems timeline readable.

An unannotated Nsight Systems timeline of this model is a wall of kernel names.
The whole run is one undifferentiated stripe of CUDA activity, and working out
which part of it is the embedding gather and which part is the multilayer
perceptron means reading kernel names one at a time and guessing. NVTX fixes
that. An NVTX range is a named interval pushed onto a stack on the host, and
Nsight Systems draws it as a labelled bar on its own row above the CUDA rows, so
the timeline reads as embedding gather, then concatenate, then linear one, then
linear two, rather than as a hundred anonymous kernels.

That matters for this workload specifically. `docs/INFERENCE.md` predicts that
this model is memory bandwidth bound on the embedding gather rather than compute
bound on the multilayer perceptron. The evidence for or against that prediction
is the share of the wall clock that the gather occupies, which is exactly the
number an NVTX range around the embedding lookup gives directly and which is
tedious to reconstruct from kernel names.

This module is a helper and not a change to the benchmark. Nothing in
`scripts/run_inference_benchmark.py` imports it and nothing has to. Instrumenting
a model is one call.

    from tools.nvtx import instrument, range as nvtx_range

    handle = instrument(model)          # a named range per submodule forward
    with nvtx_range("timed pass"):      # a range around anything at all
        run_the_batches()
    handle.remove()

Everything here is inert when NVTX is not available, which is every machine
without a CUDA build of PyTorch, including the Apple Silicon machine this
project is developed on. The ranges become no ops with no import error and no
warning, so a script that annotates itself still runs unchanged on a laptop.

Nothing in this module has been executed against a real Nsight Systems capture,
because that needs an NVIDIA GPU and this project has none. The no op path is
exercised on this machine and the NVTX path is not. See
`docs/BENCHMARK_AUTOMATION.md`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator, List, Optional, TypeVar

__all__ = [
    "backend",
    "available",
    "push",
    "pop",
    "mark",
    "range",
    "annotate",
    "instrument",
    "InstrumentHandle",
]

F = TypeVar("F", bound=Callable[..., Any])


def _detect_backend() -> str:
    """Pick the NVTX implementation to use, or report that there is none.

    PyTorch is tried first because it is already a dependency of this project
    and its CUDA builds ship NVTX bindings, so on a GPU box the annotation costs
    no extra install. The standalone `nvtx` package is the fallback for a
    process that has NVTX available without a CUDA build of torch. Neither being
    present is the normal case on a laptop and is not an error.
    """
    try:
        import torch

        if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
            # Touch the api once, because the bindings exist on some builds that
            # cannot actually call into the NVTX library.
            torch.cuda.nvtx.range_push("nvtx probe")
            torch.cuda.nvtx.range_pop()
            return "torch.cuda.nvtx"
    except Exception:  # noqa: BLE001  any failure here means the path is unusable
        pass

    try:
        import nvtx as _candidate

        # This module is itself named nvtx, so running it as a script puts its
        # own directory first on sys.path and `import nvtx` finds this file
        # again. Checking for the api rather than for the import is what tells
        # the real package apart from that, and it also rejects a partially
        # installed one.
        if _candidate.__name__ == __name__ or getattr(_candidate, "__file__", None) == __file__:
            return "disabled"
        if hasattr(_candidate, "push_range") and hasattr(_candidate, "pop_range"):
            return "nvtx"
    except Exception:  # noqa: BLE001
        pass

    return "disabled"


backend = _detect_backend()


def available() -> bool:
    """Return True when a real NVTX implementation was found."""
    return backend != "disabled"


def push(name: str) -> None:
    """Open a named range. Every push needs a matching pop."""
    if backend == "torch.cuda.nvtx":
        import torch

        torch.cuda.nvtx.range_push(name)
    elif backend == "nvtx":
        import nvtx as _nvtx

        _nvtx.push_range(name)


def pop() -> None:
    """Close the innermost open range."""
    if backend == "torch.cuda.nvtx":
        import torch

        torch.cuda.nvtx.range_pop()
    elif backend == "nvtx":
        import nvtx as _nvtx

        _nvtx.pop_range()


def mark(message: str) -> None:
    """Drop an instantaneous marker on the timeline.

    Useful for the boundary between warmup and timed passes, which is otherwise
    invisible in a profile and is the single most common way a profile ends up
    including startup cost that a serving process pays once.
    """
    if backend == "torch.cuda.nvtx":
        import torch

        torch.cuda.nvtx.mark(message)
    elif backend == "nvtx":
        import nvtx as _nvtx

        _nvtx.mark(message=message)


@contextlib.contextmanager
def range(name: str) -> Iterator[None]:  # noqa: A001  the name is the point
    """Wrap a block of code in a named NVTX range.

    The pop runs in a finally block, because a range left open by an exception
    corrupts every range after it and turns the rest of the timeline into
    nonsense.
    """
    push(name)
    try:
        yield
    finally:
        pop()


def annotate(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorate a function so every call to it becomes a named range."""

    def decorator(func: F) -> F:
        label = name or getattr(func, "__qualname__", "function")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with range(label):
                return func(*args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", "wrapper")
        wrapper.__qualname__ = getattr(func, "__qualname__", "wrapper")
        wrapper.__doc__ = func.__doc__
        return wrapper  # type: ignore[return-value]

    return decorator


class InstrumentHandle:
    """Undo handle for `instrument`, so annotation can be removed again.

    Instrumentation adds host side work to every forward call. It is small, but
    it is not nothing, and it must not be left in place around a timing loop
    whose numbers are going to be published. Profile with it and measure without
    it.
    """

    def __init__(self, handles: Optional[List[Any]] = None) -> None:
        self._handles = handles or []

    @property
    def count(self) -> int:
        """How many hooks were installed. Zero when NVTX was not available."""
        return len(self._handles) // 2

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001  removing twice is not worth raising for
                pass
        self._handles = []


def instrument(
    module: Any,
    prefix: str = "",
    leaves_only: bool = True,
) -> InstrumentHandle:
    """Put an NVTX range around every submodule forward of a torch module.

    The ranges are installed as forward pre hooks and forward hooks rather than
    by rewriting the model, so the model is unchanged, the instrumentation is
    removable, and nothing about this has to live in the benchmark script.

    `leaves_only` defaults to True because nesting a range around every
    container as well as every leaf produces a deep stack of bars that mostly
    restate the module tree. The leaves are where the kernels are. Pass False to
    get the containers too, which is the view to use when the question is how
    much of the time is inside the embedding block as a whole rather than which
    individual table is expensive.

    Returns a handle whose `count` is zero when NVTX is not available, which is
    the signal that the call did nothing rather than that the model has no
    submodules.
    """
    if not available():
        return InstrumentHandle()

    try:
        import torch.nn as nn
    except ImportError:
        return InstrumentHandle()

    if not isinstance(module, nn.Module):
        raise TypeError("instrument expects a torch.nn.Module")

    handles: List[Any] = []
    for name, child in module.named_modules():
        if not name:
            continue
        if leaves_only and any(True for _ in child.children()):
            continue
        label = f"{prefix}{name} [{type(child).__name__}]"

        def make_pre(label: str) -> Callable[..., None]:
            def pre_hook(_module: Any, _inputs: Any) -> None:
                push(label)

            return pre_hook

        def post_hook(_module: Any, _inputs: Any, _output: Any) -> None:
            pop()

        handles.append(child.register_forward_pre_hook(make_pre(label)))
        handles.append(child.register_forward_hook(post_hook))

    return InstrumentHandle(handles)


def status() -> str:
    """Return a one line description of what this module would do right now."""
    if backend == "disabled":
        return (
            "NVTX is not available in this process, so every range in "
            "tools.nvtx is a no op. That is expected on a machine with no cuda "
            "build of torch and no nvtx package, and it means an annotated "
            "script still runs here unchanged."
        )
    return f"NVTX is available through {backend}, so ranges will appear in a capture."


if __name__ == "__main__":
    print(status())
