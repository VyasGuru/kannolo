//! `DenseRerankHNSW`: a compressed dense HNSW first stage reranked against plain `f16`.
//!
//! The Python face of `RerankIndex` for dense collections, mirroring
//! `hnsw_rerank_search_dense`'s argument set so the two harnesses cannot drift apart.
//!
//! The graph is always built over `f16` and only its *dataset* is then replaced by codes, so
//! the compressed representation never takes part in construction. The rerank dataset is the
//! same collection at plain `f16`, kept in the original vector order — `HNSW::search` already
//! translates permuted node ids back to original ids, so the two stages agree on ids even for
//! the `permuted` and `compressed` graph layouts.
//!
//! Unlike the sibling `Dense*HNSW` classes, which name every monomorphization in an enum, the
//! search path here is one generic impl behind a trait object. The (encoder x metric x graph x
//! PQ subspace) table has over fifty entries, and the dynamic call is made once per `search()`
//! — outside every inner loop, against a two-stage search that costs orders of magnitude more.

use half::f16;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::graph::GraphTrait;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{
    EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration,
};

use vectorium::SpaceUsage;
use vectorium::core::rerank_index::RerankIndex;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::vector::DenseVectorView;
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use vectorium::{
    Dataset, DenseDataset, IndexSerializer, RabitqConfig, RabitqExtConfig, RabitqExtQuantizer,
    RabitqQuantizer, RabitqQueryParams,
};

use super::super::hnsw::{GraphTypeKind, parse_build_graph_type, parse_graph_type};
use crate::pylib::common::{
    MetricKind, load_index_err, parse_metric, push_results, read_npy_dataset, save_index_err,
};

/// Search knobs shared by every instantiation. `beta` is carried through to `RerankIndex`;
/// rerank early exit is off unless it is set.
struct Knobs {
    k: usize,
    k_candidates: usize,
    ef_search: usize,
    alpha: Option<f32>,
    beta: Option<usize>,
    early_termination: EarlyTerminationStrategy,
}

trait DenseRerankSearcher: Send + Sync {
    fn search(&self, query: &[f32], knobs: &Knobs) -> (Vec<f32>, Vec<i64>);
    fn space_usage_bytes(&self) -> usize;
    fn dim(&self) -> usize;
    /// Write both stages to one file.
    fn save(&self, path: &str) -> PyResult<()>;
}

/// One first-stage encoder `E` over one graph backend `G`, plus the `f16` rerank dataset.
struct Stage<E, G>
where
    E: DenseVectorEncoder,
    <E as VectorEncoder>::Distance: ScalarDenseSupportedDistance,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    G: GraphTrait,
{
    index: RerankIndex<
        HNSW<DenseDataset<E>, G>,
        DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>,
    >,
    query_params: <E as VectorEncoder>::QueryParams,
    dim: usize,
    space_usage_bytes: usize,
}

impl<E, G> DenseRerankSearcher for Stage<E, G>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize,
    G: serde::Serialize,
    <E as DenseVectorEncoder>::OutputValueType: serde::Serialize,
    DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>: serde::Serialize,
{
    fn search(&self, query: &[f32], knobs: &Knobs) -> (Vec<f32>, Vec<i64>) {
        let first_stage_params = HNSWSearchConfiguration {
            ef_search: knobs.ef_search,
            early_termination: knobs.early_termination,
            query_params: self.query_params.clone(),
        };
        let view = DenseVectorView::new(query);
        let results = self.index.search(
            view,
            view,
            knobs.k_candidates,
            knobs.k,
            &first_stage_params,
            &(),
            knobs.alpha,
            knobs.beta,
            false,
        );

        let mut distances = Vec::with_capacity(knobs.k);
        let mut ids = Vec::with_capacity(knobs.k);
        push_results(results, knobs.k, &mut distances, &mut ids);
        (distances, ids)
    }

    fn space_usage_bytes(&self) -> usize {
        self.space_usage_bytes
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.index.save_index(path).map_err(save_index_err)
    }
}

/// A dense two-stage index: compressed HNSW over `f16`-derived codes, reranked on plain `f16`.
#[pyclass]
pub struct DenseRerankHNSW {
    inner: Box<dyn DenseRerankSearcher>,
}

/// Build the plain `f16` dataset once; the graph is built on a clone of it and the rerank stage
/// keeps the original-order copy.
fn plain_f16_dataset<D>(values: &[f32], dim: usize) -> DenseDataset<PlainDenseQuantizer<f16, D>>
where
    D: ScalarDenseSupportedDistance,
{
    let data: Vec<f16> = values.iter().map(|&x| f16::from_f32(x)).collect();
    let n_vecs = data.len() / dim;
    DenseDataset::from_raw(
        data.into_boxed_slice(),
        n_vecs,
        PlainDenseQuantizer::<f16, D>::new(dim),
    )
}

/// Build the `f16` graph, apply the requested adjacency layout, swap its dataset for `$target`
/// codes, and pair it with the original-order `f16` rerank dataset.
///
/// One expansion per (metric, encoder, PQ subspace); the graph layout is a runtime match inside.
macro_rules! build_stage {
    ($dist:ty, $target:ty, $cfg:expr, $qp:expr, $rerank:expr, $build_cfg:expr, $gt:expr, $dim:expr) => {{
        let plain: HNSW<DenseDataset<PlainDenseQuantizer<f16, $dist>>, Graph> =
            HNSW::build_index($rerank.clone(), $build_cfg);
        match $gt {
            GraphTypeKind::Standard => finish(
                plain.convert_dataset_into_ref::<$target, _>($cfg),
                $rerank,
                $qp,
                $dim,
            ),
            GraphTypeKind::Permuted => finish(
                plain
                    .permute_and_encode::<PlainNeighbors>()
                    .convert_dataset_into_ref::<$target, _>($cfg),
                $rerank,
                $qp,
                $dim,
            ),
            GraphTypeKind::Compressed => finish(
                plain
                    .permute_and_encode::<StreamVByteNeighbors>()
                    .convert_dataset_into_ref::<$target, _>($cfg),
                $rerank,
                $qp,
                $dim,
            ),
        }
    }};
}

fn finish<E, G>(
    index: HNSW<DenseDataset<E>, G>,
    rerank: DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>,
    query_params: <E as VectorEncoder>::QueryParams,
    dim: usize,
) -> Box<dyn DenseRerankSearcher>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize,
    G: serde::Serialize,
    <E as DenseVectorEncoder>::OutputValueType: serde::Serialize,
    DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>: serde::Serialize,
{
    // Both stages, since the rerank dataset is a second full copy of the collection and is what
    // makes RerankIndex a speed structure rather than a memory-saving one.
    let space_usage_bytes = index.space_usage_bytes() + rerank.space_usage_bytes();
    Box::new(Stage {
        index: RerankIndex::new(index, rerank),
        query_params,
        dim,
        space_usage_bytes,
    })
}

/// Wrap an index that came off disk.
///
/// `dim` and the size accounting are recovered from the loaded index rather than passed in, since
/// a caller reloading an index has no reason to know them.
#[allow(clippy::type_complexity)]
fn finish_loaded<E, G>(
    index: RerankIndex<
        HNSW<DenseDataset<E>, G>,
        DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>,
    >,
    query_params: <E as VectorEncoder>::QueryParams,
) -> Box<dyn DenseRerankSearcher>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize,
    G: serde::Serialize,
    <E as DenseVectorEncoder>::OutputValueType: serde::Serialize,
    DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>: serde::Serialize,
{
    let dim = index.rerank_dataset().output_dim();
    let space_usage_bytes =
        index.first_stage_index().space_usage_bytes() + index.rerank_dataset().space_usage_bytes();
    Box::new(Stage {
        index,
        query_params,
        dim,
        space_usage_bytes,
    })
}

/// Load one concrete (encoder, graph) instantiation from `path`.
fn load_stage<E, G>(
    path: &str,
    query_params: <E as VectorEncoder>::QueryParams,
) -> PyResult<Box<dyn DenseRerankSearcher>>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize + serde::de::DeserializeOwned,
    G: serde::Serialize + serde::de::DeserializeOwned,
    <E as DenseVectorEncoder>::OutputValueType:
        serde::Serialize + serde::de::DeserializeOwned + 'static,
    DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>:
        serde::Serialize + serde::de::DeserializeOwned,
{
    let index: RerankIndex<
        HNSW<DenseDataset<E>, G>,
        DenseDataset<PlainDenseQuantizer<f16, <E as VectorEncoder>::Distance>>,
    > = RerankIndex::load_index(path).map_err(load_index_err)?;
    Ok(finish_loaded::<E, G>(index, query_params))
}

/// The PQ arm of the load table: one `load_stage` per subspace count, from a literal list.
macro_rules! load_pq_by_m {
    ($dist:ty, $graph:ty, $path:expr, $m:expr, [$($subspaces:literal),+ $(,)?]) => {
        match $m {
            $($subspaces => load_stage::<ProductQuantizer<$subspaces, $dist>, $graph>($path, ())?,)+
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unsupported pq_subspaces {other}; supported: 4, 8, 16, 24, 32, 48, 64, 96, \
                     128, 192, 256"
                )));
            }
        }
    };
}

/// Expand one (metric, graph layout) pair into the encoder table.
///
/// Unlike the build table this needs no encoder *config*: the trained encoder comes off disk with
/// the index. Only the query-side parameters have to be supplied again, because they describe how
/// a query is processed rather than anything that was stored.
macro_rules! load_for_metric_graph {
    ($dist:ty, $graph:ty, $path:expr, $args:expr) => {{
        match $args.encoder.as_str() {
            "pq" => load_pq_by_m!(
                $dist,
                $graph,
                $path,
                $args.pq_subspaces,
                [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
            ),
            "rabitq" => load_stage::<RabitqQuantizer<$dist>, $graph>(
                $path,
                RabitqQueryParams::new($args.rabitq_query_bits),
            )?,
            "rabitq-ext" => load_stage::<RabitqExtQuantizer<$dist>, $graph>($path, ())?,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown encoder {other:?}; choose 'pq', 'rabitq' or 'rabitq-ext'"
                )));
            }
        }
    }};
}

/// Expand one metric into the encoder table.
macro_rules! build_for_metric {
    ($dist:ty, $values:expr, $dim:expr, $config:expr, $gt:expr, $args:expr) => {{
        let rerank = plain_f16_dataset::<$dist>($values, $dim);

        match $args.encoder.as_str() {
            "pq" => match $args.pq_subspaces {
                4 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<4, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                8 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<8, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                16 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<16, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                24 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<24, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                32 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<32, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                48 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<48, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                64 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<64, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                96 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<96, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                128 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<128, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                192 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<192, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                256 => build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<256, $dist>>,
                    (),
                    (),
                    rerank,
                    $config,
                    $gt,
                    $dim
                ),
                m => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "unsupported pq_subspaces {m}; supported: 4, 8, 16, 24, 32, 48, 64, 96, \
                         128, 192, 256"
                    )));
                }
            },
            "rabitq" => build_stage!(
                $dist,
                DenseDataset<RabitqQuantizer<$dist>>,
                RabitqConfig::default(),
                RabitqQueryParams::new($args.rabitq_query_bits),
                rerank,
                $config,
                $gt,
                $dim
            ),
            "rabitq-ext" => build_stage!(
                $dist,
                DenseDataset<RabitqExtQuantizer<$dist>>,
                RabitqExtConfig {
                    total_bits: $args.rabitq_total_bits,
                    ..Default::default()
                },
                (),
                rerank,
                $config,
                $gt,
                $dim
            ),
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown encoder {other:?}; choose 'pq', 'rabitq' or 'rabitq-ext'"
                )));
            }
        }
    }};
}

struct BuildArgs {
    encoder: String,
    pq_subspaces: usize,
    rabitq_total_bits: u32,
    rabitq_query_bits: u32,
}

/// What `load` needs to name the concrete type again.
///
/// Narrower than [`BuildArgs`]: `rabitq_total_bits` is a runtime field of the stored encoder, not
/// part of its type, so it comes back with the index and must not be asked for again.
struct LoadArgs {
    encoder: String,
    pq_subspaces: usize,
    rabitq_query_bits: u32,
}

/// Reject out-of-range encoder widths before they reach vectorium.
///
/// The encoders assert these ranges with `panic!`, and the release profile sets
/// `panic = "abort"` — so an out-of-range width would take the interpreter down instead of
/// raising. Checked here for the same reason `parse_build_graph_type` checks `m`.
///
/// Only the width belonging to `encoder` is checked: the others are unused, and rejecting them
/// would make the defaults unusable.
fn validate_encoder_widths(
    encoder: &str,
    total_bits: Option<u32>,
    query_bits: u32,
) -> PyResult<()> {
    match encoder {
        "rabitq" => {
            if !(1..=8).contains(&query_bits) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "rabitq_query_bits must be in 1..=8, got {query_bits}"
                )));
            }
        }
        "rabitq-ext" => {
            if let Some(bits) = total_bits
                && !matches!(bits, 2 | 4 | 8)
            {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "rabitq_total_bits must be 2, 4 or 8, got {bits}; for a 1-bit code use \
                     encoder='rabitq' instead"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

/// The build path shared by both constructors.
fn build_inner(
    values: &[f32],
    dim: usize,
    m: usize,
    ef_construction: usize,
    metric: &str,
    graph_type: &str,
    args: BuildArgs,
) -> PyResult<Box<dyn DenseRerankSearcher>> {
    if dim == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "dim must be greater than zero",
        ));
    }
    if !values.len().is_multiple_of(dim) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "data length {} is not a multiple of dim {}",
            values.len(),
            dim
        )));
    }
    validate_encoder_widths(
        &args.encoder,
        Some(args.rabitq_total_bits),
        args.rabitq_query_bits,
    )?;
    let config = HNSWBuildConfiguration::default()
        .with_num_neighbors(m)
        .with_ef_construction(ef_construction);
    let gt = parse_build_graph_type(graph_type, m)?;

    Ok(match parse_metric(metric)? {
        MetricKind::Euclidean => {
            build_for_metric!(SquaredEuclideanDistance, values, dim, &config, gt, args)
        }
        MetricKind::DotProduct => {
            build_for_metric!(DotProduct, values, dim, &config, gt, args)
        }
    })
}

#[pymethods]
impl DenseRerankHNSW {
    /// Build from a flat row-major `float32` array plus an explicit `dim`.
    ///
    /// Arguments mirror `hnsw_rerank_search_dense`: `encoder` is one of `pq`, `rabitq`,
    /// `rabitq-ext`; `pq_subspaces` applies to `pq`, `rabitq_total_bits` to `rabitq-ext`, and
    /// `rabitq_query_bits` to `rabitq`.
    #[staticmethod]
    #[pyo3(signature = (
        data_vec, dim, m=32, ef_construction=200, metric="dotproduct".to_string(),
        graph_type="standard".to_string(), encoder="pq".to_string(), pq_subspaces=32,
        rabitq_total_bits=4, rabitq_query_bits=1
    ))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
        encoder: String,
        pq_subspaces: usize,
        rabitq_total_bits: u32,
        rabitq_query_bits: u32,
    ) -> PyResult<Self> {
        let args = BuildArgs {
            encoder,
            pq_subspaces,
            rabitq_total_bits,
            rabitq_query_bits,
        };
        let inner = build_inner(
            data_vec.as_slice()?,
            dim,
            m,
            ef_construction,
            &metric,
            &graph_type,
            args,
        )?;

        Ok(DenseRerankHNSW { inner })
    }

    /// Build from a `.npy` collection, read as `f32`.
    ///
    /// `dim` comes from the file's shape. Otherwise identical to
    /// [`build_from_array`](Self::build_from_array).
    #[staticmethod]
    #[pyo3(signature = (
        data_path, m=32, ef_construction=200, metric="dotproduct".to_string(),
        graph_type="standard".to_string(), encoder="pq".to_string(), pq_subspaces=32,
        rabitq_total_bits=4, rabitq_query_bits=1
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_file(
        data_path: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
        encoder: String,
        pq_subspaces: usize,
        rabitq_total_bits: u32,
        rabitq_query_bits: u32,
    ) -> PyResult<Self> {
        // The metric only labels the dataset type here; the raw values are the same either way,
        // and the real one is applied by `build_inner`.
        let dataset = read_npy_dataset::<DotProduct>(data_path)?;
        let dim = dataset.output_dim();
        let args = BuildArgs {
            encoder,
            pq_subspaces,
            rabitq_total_bits,
            rabitq_query_bits,
        };
        let inner = build_inner(
            dataset.values(),
            dim,
            m,
            ef_construction,
            &metric,
            &graph_type,
            args,
        )?;

        Ok(DenseRerankHNSW { inner })
    }

    /// Writes both stages — first-stage graph and `f16` rerank dataset — to one file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
    }

    /// Loads an index written by [`Self::save`].
    ///
    /// Index files carry no header, so `metric`, `graph_type`, `encoder` and the encoder's width
    /// must match the values used at build time. `rabitq_total_bits` is deliberately absent: it is
    /// stored with the encoder. `rabitq_query_bits` is not stored — it describes how a query is
    /// processed, so it is free to differ from the build-time value.
    #[staticmethod]
    #[pyo3(signature = (
        path, metric="dotproduct".to_string(), graph_type="standard".to_string(),
        encoder="pq".to_string(), pq_subspaces=32, rabitq_query_bits=1
    ))]
    pub fn load(
        path: &str,
        metric: String,
        graph_type: String,
        encoder: String,
        pq_subspaces: usize,
        rabitq_query_bits: u32,
    ) -> PyResult<Self> {
        let args = LoadArgs {
            encoder,
            pq_subspaces,
            rabitq_query_bits,
        };
        validate_encoder_widths(&args.encoder, None, args.rabitq_query_bits)?;
        let inner = match (parse_metric(&metric)?, parse_graph_type(&graph_type)?) {
            (MetricKind::Euclidean, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                load_for_metric_graph!(SquaredEuclideanDistance, Graph, path, args)
            }
            (MetricKind::Euclidean, GraphTypeKind::Compressed) => {
                load_for_metric_graph!(
                    SquaredEuclideanDistance,
                    GenericGraph<StreamVByteNeighbors>,
                    path,
                    args
                )
            }
            (MetricKind::DotProduct, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                load_for_metric_graph!(DotProduct, Graph, path, args)
            }
            (MetricKind::DotProduct, GraphTypeKind::Compressed) => {
                load_for_metric_graph!(DotProduct, GenericGraph<StreamVByteNeighbors>, path, args)
            }
        };

        Ok(DenseRerankHNSW { inner })
    }

    /// Two-stage search for a single query.
    ///
    /// `k_candidates` is the frontier knob; `ef_search` sizes the first-stage candidate list;
    /// `alpha` is candidate pruning and `beta` rerank early exit; both default to `None`.
    #[pyo3(signature = (
        query, k, k_candidates=100, ef_search=100, alpha=None, beta=None,
        early_exit_threshold=None
    ))]
    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        k_candidates: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let query_slice = query.as_slice()?;
        if query_slice.len() != self.inner.dim() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "query dimension {} does not match index dimension {}",
                query_slice.len(),
                self.inner.dim()
            )));
        }

        let knobs = Knobs {
            k,
            k_candidates,
            ef_search,
            alpha,
            beta,
            early_termination: match early_exit_threshold {
                Some(lambda) => EarlyTerminationStrategy::DistanceAdaptive { lambda },
                None => EarlyTerminationStrategy::None,
            },
        };

        let (distances, ids) = self.inner.search(query_slice, &knobs);

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Two-stage search over many queries, flattened into one contiguous buffer.
    ///
    /// `queries` holds `n_queries * dim` floats. Results are query-major: `distances` and `ids`
    /// each hold `n_queries * k` entries. `num_threads` is 0 for every core, 1 for serial, or an
    /// explicit count.
    #[pyo3(signature = (
        queries, k, k_candidates=100, ef_search=100, alpha=None, beta=None,
        early_exit_threshold=None, num_threads=0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray1<f32>,
        k: usize,
        k_candidates: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let queries_slice = queries.as_slice()?;
        let dim = self.inner.dim();
        if !queries_slice.len().is_multiple_of(dim) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "queries array length {} is not a multiple of index dimension {}",
                queries_slice.len(),
                dim
            )));
        }
        let num_queries = queries_slice.len() / dim;

        let knobs = Knobs {
            k,
            k_candidates,
            ef_search,
            alpha,
            beta,
            early_termination: match early_exit_threshold {
                Some(lambda) => EarlyTerminationStrategy::DistanceAdaptive { lambda },
                None => EarlyTerminationStrategy::None,
            },
        };

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            self.inner
                .search(&queries_slice[i * dim..(i + 1) * dim], &knobs)
        };

        let results: Vec<(Vec<f32>, Vec<i64>)> = py.detach(|| match num_threads {
            1 => (0..num_queries).map(search_one).collect(),
            0 => (0..num_queries).into_par_iter().map(search_one).collect(),
            n => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("failed to build rayon thread pool")
                .install(|| (0..num_queries).into_par_iter().map(search_one).collect()),
        });

        let mut all_distances = Vec::with_capacity(num_queries * k);
        let mut all_ids = Vec::with_capacity(num_queries * k);
        for (distances, ids) in results {
            all_distances.extend(distances);
            all_ids.extend(ids);
        }

        let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
        let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }

    /// Bytes held by the first-stage index plus the `f16` rerank dataset.
    pub fn space_usage_bytes(&self) -> usize {
        self.inner.space_usage_bytes()
    }
}
