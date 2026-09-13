//! Two-stage dense search: a compressed HNSW first stage over PQ / RaBitQ / RaBitQ-ext codes,
//! reranked against a plain `f16` dataset.
//!
//! This is the dense counterpart of `hnsw_rerank_search`, which is multi-vector only and lives
//! behind the `multivec` feature.
//!
//! The first-stage index is whatever `hnsw_build --encoder pq|rabitq|rabitq-ext` produced; the
//! graph is built over `f16` and only its *dataset* is compressed, so the codes never take part
//! in construction. The rerank dataset is the same collection at plain `f16`.
//!
//! The frontier knob is `--k-candidates`; `--alpha` is the candidate-pruning threshold. `--beta`
//! (rerank early exit) is available, and off unless it is given.
//!
//! Example:
//! ```text
//! hnsw_rerank_search_dense \
//!   --index-file idx_pq32.bin --rerank-file train.npy --query-file queries.npy \
//!   --encoder pq --pq-subspaces 32 --distance dotproduct --graph-type permuted \
//!   -k 10 --k-candidates 50 --ef-search 50 --alpha 0.3 --output-path results.tsv
//! ```

use std::fs::File;
use std::io::Write;
use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;

use kannolo::graph::graph::Graph as GenericGraph;
use kannolo::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};

use vectorium::core::rerank_index::RerankIndex;
use vectorium::core::vector::DenseVectorView;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::readers::read_npy_f32;
use vectorium::{
    Dataset, DenseDataset, IndexSerializer, PlainDenseDataset, PlainDenseQuantizer,
    RabitqExtQuantizer, RabitqQuantizer, RabitqQueryParams,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum EncoderArg {
    Pq,
    Rabitq,
    RabitqExt,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum GraphTypeArg {
    Standard,
    Permuted,
    Streamvbyte,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum EarlyTerminationArg {
    None,
    DistanceAdaptive,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The compressed dense HNSW index produced by `hnsw_build --encoder pq|rabitq|rabitq-ext`.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The `.npy` collection used for reranking, read as f32 and stored as f16.
    #[clap(short, long, value_parser)]
    rerank_file: String,

    /// The `.npy` query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The output file to write the results (query_id, doc_id, rank, score).
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The encoder the first-stage index was built with.
    #[clap(long, value_enum)]
    encoder: EncoderArg,

    /// Adjacency layout of the first-stage graph. `standard` and `permuted` share a
    /// representation on disk.
    #[clap(long, value_enum, default_value_t = GraphTypeArg::Permuted)]
    graph_type: GraphTypeArg,

    /// The distance metric ("euclidean" or "dotproduct").
    #[clap(long, value_parser, default_value_t = String::from("dotproduct"))]
    distance: String,

    /// PQ subspace count, for `--encoder pq`. Must match the index.
    #[clap(long, value_parser, default_value_t = 0)]
    pq_subspaces: usize,

    /// Query code width, for `--encoder rabitq` (1..=8).
    #[clap(long, value_parser, default_value_t = 1)]
    rabitq_query_bits: u32,

    /// Final number of results per query.
    #[clap(short, long, value_parser, default_value_t = 10)]
    k: usize,

    /// Candidates retrieved from the first stage and passed to reranking. The frontier knob.
    #[clap(long, value_parser, default_value_t = 100)]
    k_candidates: usize,

    /// First-stage candidate list size. Usually a multiple of k_candidates.
    #[clap(long, value_parser, default_value_t = 100)]
    ef_search: usize,

    /// Candidate-pruning threshold (CP). Omit for no pruning.
    #[clap(long, value_parser)]
    alpha: Option<f32>,

    /// Rerank early-exit parameter (EE). Omit to disable early exit.
    #[clap(long, value_parser)]
    beta: Option<usize>,

    /// First-stage early termination.
    #[clap(long, value_enum, default_value_t = EarlyTerminationArg::None)]
    early_termination: EarlyTerminationArg,

    /// Lambda for `--early-termination distance-adaptive`.
    #[clap(long, value_parser)]
    lambda: Option<f32>,

    /// Timed passes over the query set.
    #[clap(long, value_parser, default_value_t = 1)]
    num_runs: usize,
}

fn early_termination(args: &Args) -> EarlyTerminationStrategy {
    match args.early_termination {
        EarlyTerminationArg::None => EarlyTerminationStrategy::None,
        EarlyTerminationArg::DistanceAdaptive => EarlyTerminationStrategy::DistanceAdaptive {
            lambda: args.lambda.unwrap_or_else(|| {
                eprintln!("Error: --early-termination distance-adaptive requires --lambda.");
                process::exit(1);
            }),
        },
    }
}

fn write_results_to_file(output_path: &str, results: &[(f32, usize)], k: usize) {
    let mut file = File::create(output_path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}

/// Load the rerank collection at `f16`, the precision the plain graph stores.
///
/// The `.npy` is f32 on disk, so the f32 buffer is resident until the f16 one is built.
fn read_rerank_dataset<D>(path: &str) -> DenseDataset<PlainDenseQuantizer<f16, D>>
where
    D: vectorium::ScalarDenseSupportedDistance,
{
    let plain: PlainDenseDataset<f32, D> =
        read_npy_f32::<D>(path).expect("failed to read the rerank .npy");
    let dim = plain.input_dim();
    let n_vecs = plain.len();
    let values: Vec<f16> = plain.values().iter().map(|&x| f16::from_f32(x)).collect();
    drop(plain);
    DenseDataset::from_raw(
        values.into_boxed_slice(),
        n_vecs,
        PlainDenseQuantizer::<f16, D>::new(dim),
    )
}

/// The timed loop, shared by every (encoder, metric, graph) instantiation.
///
/// Split out of the dispatch macro so the search body exists once: the macro's job is only to
/// name the fully monomorphized first-stage type.
fn run<FS, D>(
    first_stage: FS,
    rerank: DenseDataset<PlainDenseQuantizer<f16, D>>,
    queries: PlainDenseDataset<f32, D>,
    args: &Args,
    first_stage_params: FS::SearchParams,
) where
    D: vectorium::ScalarDenseSupportedDistance + Distance + From<f32>,
    FS: for<'q> vectorium::core::index::Index<Query<'q> = DenseVectorView<'q, f32>, Distance = D>,
{
    let index = RerankIndex::new(first_stage, rerank);
    let num_queries = queries.len();

    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);
    let mut total_time_search = 0u128;

    for _ in 0..args.num_runs {
        results.clear();
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(
                query,
                query,
                args.k_candidates,
                args.k,
                &first_stage_params,
                &(),
                args.alpha,
                args.beta,
                false,
            );
            total_time_search += start_time.elapsed().as_micros();
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
        }
    }

    let avg = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg} \u{3bc}s");
    println!(
        "[######] QPS: {:.2}",
        1_000_000.0 * (num_queries * args.num_runs) as f64 / total_time_search as f64
    );

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

/// Load a first-stage index of one fully named type and hand it to [`run`].
macro_rules! dispatch {
    ($ds:ty, $graph:ty, $dist:ty, $params:expr, $args:expr) => {{
        let first_stage: HNSW<$ds, $graph> =
            <HNSW<$ds, $graph> as IndexSerializer>::load_index(&$args.index_file)
                .expect("failed to load the first-stage index");
        first_stage.print_space_usage_bytes();
        let rerank = read_rerank_dataset::<$dist>(&$args.rerank_file);
        let queries: PlainDenseDataset<f32, $dist> =
            read_npy_f32::<$dist>(&$args.query_file).expect("failed to read the query .npy");
        // The two stages address the same collection by vector id, so a length mismatch is a
        // mispaired index and rerank file -- silently wrong results, not an error later.
        assert_eq!(
            *first_stage.nodes_per_level().last().unwrap(),
            rerank.len(),
            "first-stage index and rerank dataset must hold the same collection"
        );
        run(first_stage, rerank, queries, $args, $params);
    }};
}

/// Expand the PQ subspace ladder for one (metric, graph) pair.
macro_rules! dispatch_pq {
    ($graph:ty, $dist:ty, $args:expr) => {{
        let params = HNSWSearchConfiguration::<()>::default()
            .with_ef_search($args.ef_search)
            .with_early_termination(early_termination($args));
        match $args.pq_subspaces {
            4 => dispatch!(DenseDataset<ProductQuantizer<4, $dist>>, $graph, $dist, params, $args),
            8 => dispatch!(DenseDataset<ProductQuantizer<8, $dist>>, $graph, $dist, params, $args),
            16 => dispatch!(DenseDataset<ProductQuantizer<16, $dist>>, $graph, $dist, params, $args),
            24 => dispatch!(DenseDataset<ProductQuantizer<24, $dist>>, $graph, $dist, params, $args),
            32 => dispatch!(DenseDataset<ProductQuantizer<32, $dist>>, $graph, $dist, params, $args),
            48 => dispatch!(DenseDataset<ProductQuantizer<48, $dist>>, $graph, $dist, params, $args),
            64 => dispatch!(DenseDataset<ProductQuantizer<64, $dist>>, $graph, $dist, params, $args),
            96 => dispatch!(DenseDataset<ProductQuantizer<96, $dist>>, $graph, $dist, params, $args),
            128 => dispatch!(DenseDataset<ProductQuantizer<128, $dist>>, $graph, $dist, params, $args),
            192 => dispatch!(DenseDataset<ProductQuantizer<192, $dist>>, $graph, $dist, params, $args),
            256 => dispatch!(DenseDataset<ProductQuantizer<256, $dist>>, $graph, $dist, params, $args),
            m => {
                eprintln!(
                    "Error: unsupported --pq-subspaces {m}. Supported: 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256."
                );
                process::exit(1);
            }
        }
    }};
}

macro_rules! dispatch_encoder {
    ($graph:ty, $dist:ty, $args:expr) => {{
        match $args.encoder {
            EncoderArg::Pq => dispatch_pq!($graph, $dist, $args),
            EncoderArg::Rabitq => {
                let params = HNSWSearchConfiguration::default()
                    .with_ef_search($args.ef_search)
                    .with_early_termination(early_termination($args))
                    .with_query_params(RabitqQueryParams::new($args.rabitq_query_bits));
                dispatch!(
                    DenseDataset<RabitqQuantizer<$dist>>,
                    $graph,
                    $dist,
                    params,
                    $args
                )
            }
            EncoderArg::RabitqExt => {
                let params = HNSWSearchConfiguration::<()>::default()
                    .with_ef_search($args.ef_search)
                    .with_early_termination(early_termination($args));
                dispatch!(
                    DenseDataset<RabitqExtQuantizer<$dist>>,
                    $graph,
                    $dist,
                    params,
                    $args
                )
            }
        }
    }};
}

fn main() {
    let args = Args::parse();

    if args.encoder == EncoderArg::Pq && args.pq_subspaces == 0 {
        eprintln!("Error: --encoder pq requires --pq-subspaces matching the index.");
        process::exit(1);
    }

    match (args.distance.as_str(), args.graph_type) {
        ("euclidean" | "l2", GraphTypeArg::Standard | GraphTypeArg::Permuted) => {
            dispatch_encoder!(
                GenericGraph<PlainNeighbors>,
                SquaredEuclideanDistance,
                &args
            )
        }
        ("euclidean" | "l2", GraphTypeArg::Streamvbyte) => {
            dispatch_encoder!(
                GenericGraph<StreamVByteNeighbors>,
                SquaredEuclideanDistance,
                &args
            )
        }
        ("dotproduct" | "ip", GraphTypeArg::Standard | GraphTypeArg::Permuted) => {
            dispatch_encoder!(GenericGraph<PlainNeighbors>, DotProduct, &args)
        }
        ("dotproduct" | "ip", GraphTypeArg::Streamvbyte) => {
            dispatch_encoder!(GenericGraph<StreamVByteNeighbors>, DotProduct, &args)
        }
        (d, _) => {
            eprintln!("Error: invalid --distance {d}. Choose 'euclidean' or 'dotproduct'.");
            process::exit(1);
        }
    }
}
