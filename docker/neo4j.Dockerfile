# Neo4j with the Graph Data Science plugin baked in.
#
# Why a Dockerfile instead of NEO4J_PLUGINS=["graph-data-science"]: that env var makes the
# official entrypoint download the plugin from the internet on *every* container start,
# which breaks the project's offline posture (CLAUDE.md rule 1) and makes startup depend
# on a third-party host being up. Downloading at build time is the same trade already
# made for pip wheels and the embedding model.
#
# Without this plugin `personalized_pagerank` silently takes its fallback path — a
# path-count proximity score, not PageRank — so the graph prong's ranking was never what
# the docstring claimed.
#
# Version pin: GDS 2.13.x is the line that supports Neo4j 5.26. Bumping the base image
# means checking the GDS compatibility matrix, not just taking the newest jar.
FROM neo4j:5.26-community

ARG GDS_VERSION=2.13.4
ARG GDS_SHA256=10e072f73992224f1159f246c9d6a89da5f3b3434aeffa5be42647edda13a8d8

USER root
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    curl -fsSL -o /var/lib/neo4j/plugins/neo4j-graph-data-science.jar \
      "https://github.com/neo4j/graph-data-science/releases/download/${GDS_VERSION}/neo4j-graph-data-science-${GDS_VERSION}.jar"; \
    echo "${GDS_SHA256}  /var/lib/neo4j/plugins/neo4j-graph-data-science.jar" | sha256sum -c -; \
    chown neo4j:neo4j /var/lib/neo4j/plugins/neo4j-graph-data-science.jar; \
    apt-get purge -y --auto-remove curl; \
    rm -rf /var/lib/apt/lists/*
# No trailing USER: the stock image runs as root and its entrypoint chowns /data and
# /logs before dropping to neo4j itself. An earlier version ended with `USER neo4j`,
# which survived CI and throwaway containers only because those had fresh volumes — on
# an existing store it risks a permissions failure at startup, which looks like data
# loss. Match the base image's contract instead of inventing a new one.
