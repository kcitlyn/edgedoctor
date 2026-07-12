"""Rule knowledge base — curated error → cause → fix mappings.

Each YAML file contains rules for one backend. A rule fires when all of its
`requires` fact kinds are present in the parsed Facts. The match is
deterministic, fast, and needs no API key — the k8sgpt model.
"""
