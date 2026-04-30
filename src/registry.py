class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items = {}

    def __call__(self, name: str):
        def decorator(obj):
            if name in self._items:
                raise ValueError("Duplicate {} registry key '{}'".format(self.kind, name))
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(
                "Unknown {} '{}'. Available: {}".format(
                    self.kind, name, ", ".join(sorted(self._items.keys()))
                )
            )
        return self._items[name]

    def names(self):
        return sorted(self._items.keys())


DATASETS = Registry("dataset")
FILTRATIONS = Registry("filtration")
MODELS = Registry("model")
ENCODERS = Registry("encoder")
