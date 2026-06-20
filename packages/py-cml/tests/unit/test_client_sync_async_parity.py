"""Guard: the sync and async SDK clients must expose the same public methods.

Catches the real risk in the sync/async twins — one drifting from the other
(e.g. a method added to async but not sync, the read_stream bug class) — without
an unasync codegen pipeline to maintain.

ponytail: parity test instead of sync/async codegen; build the generator only if
the twins ever need to diverge structurally.
"""

import inspect

from cml.async_client import AsyncCognitiveMemoryLayer, AsyncNamespacedClient
from cml.client import CognitiveMemoryLayer, NamespacedClient


def _public_methods(cls: type) -> set[str]:
    return {name for name, _ in inspect.getmembers(cls, callable) if not name.startswith("_")}


def test_sync_async_client_method_parity():
    assert _public_methods(CognitiveMemoryLayer) == _public_methods(AsyncCognitiveMemoryLayer)


def test_sync_async_namespaced_method_parity():
    assert _public_methods(NamespacedClient) == _public_methods(AsyncNamespacedClient)


if __name__ == "__main__":
    test_sync_async_client_method_parity()
    test_sync_async_namespaced_method_parity()
    print("client sync/async parity OK")
