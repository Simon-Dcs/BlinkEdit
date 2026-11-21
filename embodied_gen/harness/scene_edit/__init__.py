"""Transactional natural-language scene editing using the native spatial API."""

__all__ = ["SceneEditor"]


def __getattr__(name):
    if name == "SceneEditor":
        from .runtime import SceneEditor

        return SceneEditor
    raise AttributeError(name)
