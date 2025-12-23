# Phase-2 Evaluation Data

This directory contains evaluation queries and expected evidence for testing the RAG system.

## File Format

### queries.jsonl

Each line is a JSON object with:
- `query_id`: Unique identifier for the query
- `query`: The question text
- `expected_chunks`: List of expected chunk ID patterns (fuzzy matched)
- `expected_acts`: List of acts that should appear in results
- `category`: Query category (definition, procedure, scope, etc.)

Example:
```json
{"query_id": "q1", "query": "What is the definition of employer?", "expected_chunks": ["section_2"], "expected_acts": ["Minimum Wages Act"], "category": "definition"}
```

### expected_sources.jsonl

Reference data for expected evidence chunks:
- `chunk_id`: Pattern to match chunk IDs
- `act`: Act name
- `section`: Section number
- `text_snippet`: Key text that should appear
- `relevance`: Why this chunk is relevant

## Creating Evaluation Queries

1. **Identify key questions** your system should answer
2. **Determine expected evidence** by manually reviewing documents
3. **Use fuzzy patterns** for chunk IDs (e.g., "section_2" matches any chunk from Section 2)
4. **Specify expected acts** to enable act-level filtering

## Updating Evaluation Data

To add new queries:

```bash
# Add to queries.jsonl
echo '{"query_id": "q11", "query": "Your question?", "expected_chunks": ["pattern"], "expected_acts": ["Act Name"], "category": "type"}' >> eval/queries.jsonl
```

## Notes

- **Fuzzy matching** is enabled by default - expected chunk patterns are matched against actual chunk IDs and content
- **Act-level matching** helps when exact chunk IDs are unknown
- **Categories** help analyze performance by query type
