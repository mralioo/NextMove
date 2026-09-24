import os

os.environ["NEO4J_MIRROR"] = "off"      # unit tests never write to the real Neo4j
