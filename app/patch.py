try:
    import pkg_resources  # noqa: F401
except ImportError:
    import sys
    import types
    import importlib.metadata

    pkg = types.ModuleType("pkg_resources")

    def get_distribution(name):
        class Dist:
            def __init__(self, n):
                try:
                    self.version = importlib.metadata.version(n)
                except Exception:
                    self.version = "0.0.0"
        return Dist(name)

    class DistributionNotFound(Exception):
        pass

    pkg.get_distribution = get_distribution
    pkg.DistributionNotFound = DistributionNotFound
    sys.modules["pkg_resources"] = pkg
