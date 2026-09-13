# Unified Build, Search, and Convert Binaries

This document describes the current CLI surface for:
- `hnsw_build`
- `hnsw_search`
- `hnsw_rerank_search_dense`
- `ivf_build`
- `ivf_search`

`hnsw_rerank_search` is documented separately in [MultiVectorUsage.md](MultiVectorUsage.md).

All binaries are behind the `cli` feature (`hnsw_rerank_search` additionally needs `multivec`),
so a plain `cargo build` produces none of them:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
```

All examples and option names below are aligned with the current binaries.

## `hnsw_build`

```bash
Usage: hnsw_build [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE> --dataset-type <DATASET_TYPE>

Options:
  -d, --data-file <DATA_FILE>
  -o, --output-file <OUTPUT_FILE>
      --dataset-type <DATASET_TYPE>      [possible values: dense, sparse]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f16, f32, fixedu8, fixedu16]
      --component-type <COMPONENT_TYPE>  [default: u16] [possible values: u16, u32]
      --encoder <ENCODER>                [default: plain] [possible values: plain, pq, dotvbyte, rabitq, rabitq-ext]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
      --m <M>                            [default: 16]
      --ef-construction <EF_CONSTRUCTION> [default: 150]
      --distance <DISTANCE>              [default: dotproduct]
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
      --nbits <NBITS>                    [default: 8] (ignored by vectorium PQ)
      --sample-size <SAMPLE_SIZE>        [default: 100000] (ignored by vectorium PQ)
      --rabitq-seed <RABITQ_SEED>        [default: 42] Seed for the random orthogonal rotation
      --rabitq-no-rotate                 Skip the rotation: faster, but expect recall to drop
      --rabitq-total-bits <BITS>         [default: 4] Document code width, rabitq-ext only [2, 4, 8]
      --rabitq-exact-quant               Per-vector rescale search instead of the trained constant
```

## `hnsw_search`

```bash
Usage: hnsw_search [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE> --dataset-type <DATASET_TYPE> --distance <DISTANCE>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>
  -o, --output-path <OUTPUT_PATH>
      --dataset-type <DATASET_TYPE>      [possible values: dense, sparse]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f16, f32, fixedu8, fixedu16]
      --component-type <COMPONENT_TYPE>  [default: u16] [possible values: u16, u32]
      --encoder <ENCODER>                [default: plain] [possible values: plain, pq, dotvbyte, rabitq, rabitq-ext]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
      --distance <DISTANCE>
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
  -k, --k <K>                            [default: 10]
      --ef-search <EF_SEARCH>            [default: 40]
      --query-bits <QUERY_BITS>          [default: 1] Query code width for --encoder rabitq (1..=8)
      --early-termination <EARLY_TERMINATION> [default: none] [possible values: none, distance-adaptive]
      --lambda <LAMBDA>                  [default: 1]
      --num-runs <NUM_RUNS>              [default: 1]
```

## `hnsw_rerank_search_dense`

Two-stage dense search: a compressed HNSW first stage over PQ / RaBitQ / RaBitQ-ext codes,
reranked against a plain `f16` copy of the same collection. The dense counterpart of
`hnsw_rerank_search`, which is sparse-multivector search.

The graph is always built over `f16` and only its *dataset* is replaced by codes, so the compressed representation never takes part in construction. The rerank file is the same collection at plain `f16`, in the original vector order.

```bash
Usage: hnsw_rerank_search_dense [OPTIONS] --index-file <INDEX_FILE> --rerank-file <RERANK_FILE> --query-file <QUERY_FILE> --encoder <ENCODER>

Options:
  -i, --index-file <INDEX_FILE>
  -r, --rerank-file <RERANK_FILE>
  -q, --query-file <QUERY_FILE>
  -o, --output-path <OUTPUT_PATH>
      --encoder <ENCODER>                [possible values: pq, rabitq, rabitq-ext]
      --graph-type <GRAPH_TYPE>          [default: permuted] [possible values: standard, permuted, streamvbyte]
      --distance <DISTANCE>              [default: dotproduct]
      --pq-subspaces <PQ_SUBSPACES>      [default: 0]
      --rabitq-query-bits <RABITQ_QUERY_BITS>  [default: 1]
  -k, --k <K>                            [default: 10]
      --k-candidates <K_CANDIDATES>      [default: 100]
      --ef-search <EF_SEARCH>            [default: 100]
      --alpha <ALPHA>
      --beta <BETA>
      --early-termination <EARLY_TERMINATION> [default: none] [possible values: none, distance-adaptive]
      --lambda <LAMBDA>
      --num-runs <NUM_RUNS>              [default: 1]
```

`--k-candidates` is the frontier knob: it sets how many first-stage candidates are reranked.
`--ef-search` sizes the first-stage candidate list and is usually a multiple of
`--k-candidates`. `--alpha` prunes candidates whose first-stage score falls outside a relative
slack of the k-th best; `--beta` is rerank early exit and is off unless given.

`--pq-subspaces` must match what the index was built with, for `--encoder pq`: it names the
monomorphization to load the index as. There is no matching flag for `--encoder rabitq-ext` — the
document code width is stored with the encoder and comes back with the index. `--rabitq-query-bits`
applies to `--encoder rabitq` and describes only how the query is processed, so it can be varied
without rebuilding.

## `ivf_build`

Builds an inverted-file index. Dense only.

```bash
Usage: ivf_build [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE>

Options:
  -d, --data-file <DATA_FILE>
  -o, --output-file <OUTPUT_FILE>
      --distance <DISTANCE>              [default: euclidean] [possible values: euclidean, dotproduct]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f32, f16]
      --n-clusters <N_CLUSTERS>          [default: 1024]
      --kmeans-n-iter <KMEANS_N_ITER>    [default: 25]
      --kmeans-n-redo <KMEANS_N_REDO>    [default: 1]
      --kmeans-sample-size <SIZE>        (optional; defaults to the whole dataset)
      --kmeans-hnsw                      Use an HNSW index to speed up k-means assignment
      --kmeans-spherical                 L2-normalize centroids each iteration
      --residuals                        Encode vectors as residuals from their centroid
      --hnsw                             Index the centroids with HNSW instead of scanning them
      --m-hnsw <M_HNSW>                  [default: 32] (alias: --m) Only with --hnsw
      --ef-construction <EF_CONSTRUCTION> [default: 200] Only with --hnsw
      --m-pq <M_PQ>                      (optional) PQ-encode the vectors with this many subspaces
```

## `ivf_search`

```bash
Usage: ivf_search [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>
  -o, --output-path <OUTPUT_PATH>
  -k, --k <K>                            [default: 10]
      --n-probe <N_PROBE>                [default: 32] Clusters visited per query
      --distance <DISTANCE>              [default: euclidean] [possible values: euclidean, dotproduct]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f32, f16]
      --hnsw                             Must match the index
      --m-pq <M_PQ>                      Must match the index
      --ef-search <EF_SEARCH>            [default: 40] Only with --hnsw
      --lambda <LAMBDA>                  [default: 0]
      --num-runs <NUM_RUNS>              [default: 1]
```

As with HNSW, `--distance`, `--value-type`, `--hnsw` and `--m-pq` are properties baked into the
index at build time and must be repeated identically at search time: each one names part of the
concrete type the index has to be loaded as. `ivf_build --residuals` has no search-time
counterpart — residual-vs-plain is runtime state stored in the index, so it is read back from the
index rather than restated.

## Examples

Dense plain:

```bash
./hnsw_build --data-file data.npy --output-file index.bin \
  --dataset-type dense --encoder plain --value-type f32 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Dense PQ:

```bash
./hnsw_build --data-file data.npy --output-file index.bin \
  --dataset-type dense --encoder pq --pq-subspaces 16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse plain with explicit sparse component type:

```bash
./hnsw_build --data-file data.bin --output-file index.bin \
  --dataset-type sparse --encoder plain --value-type f16 --component-type u16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse DotVByte:

```bash
./hnsw_build --data-file data.bin --output-file index.bin \
  --dataset-type sparse --encoder dotvbyte --component-type u16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse DotVByte search:

```bash
./hnsw_search --index-file index.bin --query-file queries.bin \
  --dataset-type sparse --encoder dotvbyte --component-type u16 \
  --distance dotproduct --k 10 --ef-search 40 --output-path results.tsv
```

Compressed graph (see [GraphCompression.md](GraphCompression.md)):

```bash
./hnsw_build --data-file data.npy --output-file index_svb.bin \
  --dataset-type dense --encoder plain --value-type f32 --graph-type streamvbyte \
  --m 16 --ef-construction 150 --distance dotproduct

./hnsw_search --index-file index_svb.bin --query-file queries.npy \
  --dataset-type dense --encoder plain --value-type f32 --graph-type streamvbyte \
  --distance dotproduct --k 10 --ef-search 40 --output-path results.tsv
```

IVF:

```bash
./ivf_build --data-file data.npy --output-file ivf.bin \
  --distance euclidean --value-type f32 --n-clusters 1024

./ivf_search --index-file ivf.bin --query-file queries.npy \
  --distance euclidean --value-type f32 --n-probe 32 --k 10 --output-path results.tsv
```

## Validation Rules

The binaries reject invalid combinations:

1. `pq` is dense-only.
2. `rabitq` and `rabitq-ext` are dense-only.
3. `dotvbyte` is sparse-only.
4. `fixedu8` and `fixedu16` value types are sparse-only.
5. `component-type` is sparse-only.
6. `dotvbyte` requires `component-type = u16`.
7. `pq-subspaces` must be one of `4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256` and must divide the vector dimensionality.
8. For PQ, `--nbits` and `--sample-size` are accepted for compatibility but ignored by vectorium.
9. `--rabitq-total-bits` must be 2, 4 or 8; for a 1-bit code use `--encoder rabitq` instead.
10. `--graph-type streamvbyte` caps the ground level at 256 neighbors per node, so it requires `--m` of at most 128. This is checked before the build starts, not after.

Beyond these, `dataset-type`, `value-type`, `component-type`, `encoder`, `distance` and `graph-type` are all baked into the index at build time. `hnsw_search` cannot detect them — index files carry no header — so passing a different value produces a decode error rather than a helpful message. Repeat the build flags exactly.

`hnsw_search --query-bits` is the exception: it scalar-quantizes the *query* for `--encoder rabitq`
and is not stored in the index, so one index serves every setting and it can be varied without
rebuilding. Every other encoder ignores it, `rabitq-ext` included — that one's query width follows
the document code width stored with the index, which is why there is no search-time
`--rabitq-total-bits`.
