"""SonicMoE dataflow & memory visualization suite."""

try:
    from visualization.sonicmoe_dataflow import generate_all  # noqa: F401
except ModuleNotFoundError:
    generate_all = None  # type: ignore[assignment]
