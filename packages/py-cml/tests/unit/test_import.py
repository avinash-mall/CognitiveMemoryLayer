"""Verify package and public API imports."""


def test_import_cml() -> None:
    """Importing cml should succeed and expose __version__."""
    import cml
    from cml._version import __version__ as expected

    assert cml.__version__ == expected


def test_import_public_api() -> None:
    """Public API should be importable from cml."""
    from cml import (
        AsyncCognitiveMemoryLayer,
        CMLConfig,
        CMLError,
        CognitiveMemoryLayer,
        __version__,
    )
    from cml._version import __version__ as expected

    assert __version__ == expected
    assert CognitiveMemoryLayer is not None
    assert AsyncCognitiveMemoryLayer is not None
    assert CMLConfig is not None
    assert CMLError is not None


def test_import_models() -> None:
    """Models subpackage should expose MemoryType and response models."""
    from cml.models import HealthResponse, MemoryType

    assert MemoryType.EPISODIC_EVENT is not None
    assert MemoryType.SEMANTIC_FACT is not None
    assert HealthResponse is not None


def test_import_exceptions() -> None:
    """Exception hierarchy should be importable."""
    from cml.exceptions import (
        AuthenticationError,
        CMLError,
        ConnectionError,
        RateLimitError,
        ValidationError,
    )

    assert issubclass(AuthenticationError, CMLError)
    assert issubclass(ConnectionError, CMLError)
    assert issubclass(RateLimitError, CMLError)
    assert issubclass(ValidationError, CMLError)


def test_client_instantiation() -> None:
    """Sync and async clients can be instantiated with minimal args."""
    from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer

    sync_client = CognitiveMemoryLayer(api_key="sk-test", base_url="http://localhost:8000")
    assert sync_client._config.api_key == "sk-test"
    assert sync_client._config.base_url == "http://localhost:8000"
    assert sync_client._transport is not None

    async_client = AsyncCognitiveMemoryLayer(api_key="sk-test")
    assert async_client._config.api_key == "sk-test"
    assert async_client._transport is not None
