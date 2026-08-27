# Public release data note

The runtime semantic index contains 268 chunks derived from six University of Warwick public policy pages. The original downloaded HTML snapshots are intentionally excluded from this public repository because a saved webpage can contain signed-in account or session metadata that is unrelated to the policy text.

The public release contains:

- `chunks.json`: the processed policy passages and metadata;
- `embeddings.npy`: the frozen MiniLM embeddings;
- `metadata.json`: the semantic-index configuration;
- `docs/policy_data_manifest.csv`: titles, official source URLs, access dates and original source hashes.

The original frozen `chunks.json` SHA-256 was:

```text
303b0da73dd810ac4fa02fec281e7bb80e43893d4a456d2ee3440e399f0de1a2
```

The public `chunks.json` SHA-256 is:

```text
76b70f529ee1b445e5ad4880f87e42dda621b7d877bdc899ff0f45ea5eea52d4
```

The difference is limited to replacing absolute Windows `source_path` values with relative `data/raw/<filename>` paths. Chunk text, chunk order, section identifiers and all other fields are unchanged.

The frozen embeddings remain byte-for-byte unchanged:

```text
3b0b57fe5498a3f92c7198d0c91cb72a37d8a016ec6cd8cb9805115cb1f70ef0
```

The official University pages linked in the manifest remain the authoritative policy sources.
