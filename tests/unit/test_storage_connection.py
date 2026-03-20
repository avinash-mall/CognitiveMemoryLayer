from src.storage.connection import _neo4j_host_allows_empty_password


def test_neo4j_host_allows_empty_password_for_local_loopbacks() -> None:
    assert _neo4j_host_allows_empty_password("bolt://localhost:7687")
    assert _neo4j_host_allows_empty_password("bolt://127.0.0.1:7687")
    assert _neo4j_host_allows_empty_password("bolt://[::1]:7687")


def test_neo4j_host_requires_exact_local_host_match() -> None:
    assert not _neo4j_host_allows_empty_password("bolt://localhost.attacker.invalid:7687")
    assert not _neo4j_host_allows_empty_password("bolt://db-localhost.internal:7687")
    assert not _neo4j_host_allows_empty_password("bolt://neo4j:7687")
