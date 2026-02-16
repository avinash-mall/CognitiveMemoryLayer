import asyncio
import os
from neo4j import AsyncGraphDatabase

# Mock settings or use env vars
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


async def debug_neo4j():
    print(f"Connecting to {NEO4J_URL} as {NEO4J_USER}...")
    driver = AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

    async with driver.session() as session:
        # 1. Count all nodes
        result = await session.run("MATCH (n) RETURN count(n) as count")
        record = await result.single()
        print(f"Total nodes: {record['count']}")

        # 2. Count nodes with Entity label
        result = await session.run("MATCH (n:Entity) RETURN count(n) as count")
        record = await result.single()
        print(f"Nodes with :Entity label: {record['count']}")

        # 3. Count relationships
        result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
        record = await result.single()
        print(f"Total relationships: {record['count']}")

        # 4. Check tenants
        result = await session.run("MATCH (n:Entity) RETURN DISTINCT n.tenant_id as tenant_id")
        tenants = [record["tenant_id"] async for record in result]
        print(f"Tenants found: {tenants}")

        # 6. Simulate getGraphOverview
        tenant_id = "lp-0"
        scope_id = "lp-0"
        print(f"\nSimulating getGraphOverview for tenant={tenant_id}, scope={scope_id}...")

        # 6a. Find center node (FIXED syntax)
        query_center = """
        MATCH (n:Entity {tenant_id: $tenant_id, scope_id: $scope_id})
        WITH n, COUNT { (n)--() } AS deg
        ORDER BY deg DESC
        LIMIT 1
        RETURN n.entity AS entity, deg
        """
        result = await session.run(query_center, tenant_id=tenant_id, scope_id=scope_id)
        record = await result.single()
        if record:
            center_entity = record["entity"]
            degree = record["deg"]
            print(f"Center entity: '{center_entity}' (degree={degree})")

            # 6b. Get neighbors (simplified version of dashboard query)
            query_neighbors = """
            MATCH path = (start:Entity {
                tenant_id: $tenant_id, scope_id: $scope_id, entity: $entity
            })-[*1..2]-(neighbor:Entity)
            WHERE neighbor.tenant_id = $tenant_id AND neighbor.scope_id = $scope_id
            RETURN count(neighbor) as tumor
            """
            result = await session.run(
                query_neighbors, tenant_id=tenant_id, scope_id=scope_id, entity=center_entity
            )
            record = await result.single()
            print(f"Neighbors count: {record['tumor']}")

        else:
            print("No center entity found.")

    await driver.close()


if __name__ == "__main__":
    asyncio.run(debug_neo4j())
