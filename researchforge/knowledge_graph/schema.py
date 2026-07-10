"""Knowledge graph schema definitions."""

NODE_LABELS = [
    "Paper",
    "Dataset",
    "Model",
    "Task",
    "Metric",
    "Benchmark",
    "Repository",
    "Author",
]

EDGE_TYPES = [
    "uses_dataset",
    "evaluates_on",
    "beats",
    "extends",
    "cites",
    "implements",
    "trained_on",
    "authored",
]

NEO4J_SCHEMA_CYPHER = """
CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (n:Paper) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (n:Dataset) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT model_id IF NOT EXISTS FOR (n:Model) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT task_id IF NOT EXISTS FOR (n:Task) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT metric_id IF NOT EXISTS FOR (n:Metric) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT benchmark_id IF NOT EXISTS FOR (n:Benchmark) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT repository_id IF NOT EXISTS FOR (n:Repository) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT author_id IF NOT EXISTS FOR (n:Author) REQUIRE n.id IS UNIQUE;
""".strip()

KUZU_SCHEMA_DDL = """
CREATE NODE TABLE Paper(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE NODE TABLE Dataset(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE NODE TABLE Model(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE NODE TABLE Task(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE NODE TABLE Metric(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE NODE TABLE Benchmark(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE NODE TABLE Repository(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE NODE TABLE Author(id STRING, title STRING, source STRING, PRIMARY KEY (id));
CREATE REL TABLE authored(FROM Author TO Paper);
CREATE REL TABLE implements(FROM Repository TO Paper);
CREATE REL TABLE cites(FROM Paper TO Paper);
""".strip()
