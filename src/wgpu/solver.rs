use super::operator::WgpuCsrOp;
use super::preconditioner::WgpuJacobi;
use super::runtime::{WgpuRuntime, encode_f32, encode_u32};
use super::vector::WgpuVector;
use crate::context::ksp_context::SolverType;
use crate::context::pc_context::PcType;
use crate::error::KError;
use crate::utils::convergence::{
    ConvergedReason, Convergence, ReductionModel, SolveStats, SolverCounters,
};
use std::sync::Arc;

const VECTOR_SHADER: &str = r"
struct VectorParams {
    alpha: f32,
    beta: f32,
    n: u32,
    padding: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(2) var<uniform> params: VectorParams;

@compute @workgroup_size(128)
fn axpby(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.n) {
        output_values[id.x] = fma(params.alpha, input_values[id.x],
                                    params.beta * output_values[id.x]);
    }
}
";

const REDUCTION_SHADER: &str = r"
struct ReductionParams {
    n: u32,
    partial_count: u32,
    padding0: u32,
    padding1: u32,
};

@group(0) @binding(0) var<storage, read> x0: array<f32>;
@group(0) @binding(1) var<storage, read> y0: array<f32>;
@group(0) @binding(2) var<storage, read> x1: array<f32>;
@group(0) @binding(3) var<storage, read> y1: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> result: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: ReductionParams;
@group(0) @binding(7) var<storage, read> x2: array<f32>;
@group(0) @binding(8) var<storage, read> y2: array<f32>;

var<workgroup> scratch: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn reduce_stage(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) group_id: vec3<u32>) {
    var value = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    if (global_id.x < params.n) {
        value = vec4(x0[global_id.x] * y0[global_id.x],
                     x1[global_id.x] * y1[global_id.x],
                     x2[global_id.x] * y2[global_id.x],
                     0.0f);
    }
    scratch[local_id.x] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (local_id.x < stride) {
            scratch[local_id.x] = scratch[local_id.x] + scratch[local_id.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }
    if (local_id.x == 0u) {
        partials[group_id.x] = scratch[0];
    }
}

@compute @workgroup_size(256)
fn reduce_final(@builtin(local_invocation_id) local_id: vec3<u32>) {
    var value = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    var index = local_id.x;
    loop {
        if (index >= params.partial_count) {
            break;
        }
        value = value + partials[index];
        index = index + 256u;
    }
    scratch[local_id.x] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (local_id.x < stride) {
            scratch[local_id.x] = scratch[local_id.x] + scratch[local_id.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }
    if (local_id.x == 0u) {
        result[0] = scratch[0];
    }
}
";

const RESIDENT_VECTOR_SHADER: &str = r"
struct SolverState {
    rho_previous: f32,
    rho: f32,
    alpha_hi: f32,
    alpha_lo: f32,
    omega_hi: f32,
    omega_lo: f32,
    beta: f32,
    bnorm_squared: f32,
    residual_squared: f32,
    rtol_squared: f32,
    atol_squared: f32,
    dtol_squared: f32,
    running: f32,
    iterations: f32,
    reason: f32,
    max_iterations: f32,
};

fn alpha_value() -> f32 {
    return state.alpha_hi + state.alpha_lo;
}

fn omega_value() -> f32 {
    return state.omega_hi + state.omega_lo;
}

@group(0) @binding(0) var<storage, read> state: SolverState;
@group(0) @binding(1) var<storage, read> input_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> input_c: array<f32>;

@compute @workgroup_size(128)
fn update_p_omega(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output) && state.running > 0.5f && state.iterations >= 1.5f) {
        output[id.x] = fma(-omega_value(), input_c[id.x], output[id.x]);
    }
}

@compute @workgroup_size(128)
fn prepare_p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output) && state.running > 0.5f) {
        if (state.iterations < 1.5f) {
            output[id.x] = input_a[id.x];
        } else {
            output[id.x] = fma(state.beta, output[id.x], input_a[id.x]);
        }
    }
}

@compute @workgroup_size(128)
fn make_s(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output) && state.running > 0.5f) {
        output[id.x] = fma(-alpha_value(), input_c[id.x], input_a[id.x]);
    }
}

@compute @workgroup_size(128)
fn update_x_alpha(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output) && state.running > 0.5f) {
        output[id.x] = fma(alpha_value(), input_a[id.x], output[id.x]);
    }
}

@compute @workgroup_size(128)
fn update_x_omega(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output) && state.running > 0.5f) {
        output[id.x] = fma(omega_value(), input_a[id.x], output[id.x]);
    }
}

@compute @workgroup_size(128)
fn make_r(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output) && state.running > 0.5f) {
        output[id.x] = fma(-omega_value(), input_c[id.x], input_a[id.x]);
    }
}
";

const RESIDENT_SCALAR_SHADER: &str = r"
struct SolverState {
    rho_previous: f32,
    rho: f32,
    alpha_hi: f32,
    alpha_lo: f32,
    omega_hi: f32,
    omega_lo: f32,
    beta: f32,
    bnorm_squared: f32,
    residual_squared: f32,
    rtol_squared: f32,
    atol_squared: f32,
    dtol_squared: f32,
    running: f32,
    iterations: f32,
    reason: f32,
    max_iterations: f32,
};

struct DoubleSingle {
    hi: f32,
    lo: f32,
};

const CONTINUED: f32 = 0.0f;
const CONVERGED_RTOL: f32 = 1.0f;
const CONVERGED_ATOL: f32 = 2.0f;
const DIVERGED_DTOL: f32 = 3.0f;
const DIVERGED_MAX_ITERS: f32 = 4.0f;
const DIVERGED_BREAKDOWN: f32 = 5.0f;
const DIVERGED_NAN: f32 = 6.0f;
const DIVERGED_INF: f32 = 7.0f;
const MIN_NORMAL: f32 = 1.17549435e-38f;

@group(0) @binding(0) var<storage, read_write> state: SolverState;
@group(0) @binding(1) var<storage, read> reduction: array<vec4<f32>>;

fn invalid_reason(value: f32) -> f32 {
    let bits = bitcast<u32>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if ((bits & 0x007fffffu) != 0u) {
            return DIVERGED_NAN;
        }
        return DIVERGED_INF;
    }
    return CONTINUED;
}

fn ds_from(value: f32) -> DoubleSingle {
    return DoubleSingle(value, 0.0f);
}

fn ds_value(value: DoubleSingle) -> f32 {
    return value.hi + value.lo;
}

fn ds_quick_sum(a: f32, b: f32) -> DoubleSingle {
    let sum = a + b;
    return DoubleSingle(sum, b - (sum - a));
}

fn ds_add(a: DoubleSingle, b: DoubleSingle) -> DoubleSingle {
    let sum = a.hi + b.hi;
    let virtual_b = sum - a.hi;
    let error = ((b.hi - virtual_b) + (a.hi - (sum - virtual_b))) + a.lo + b.lo;
    return ds_quick_sum(sum, error);
}

fn ds_negate(value: DoubleSingle) -> DoubleSingle {
    return DoubleSingle(-value.hi, -value.lo);
}

fn ds_multiply(a: DoubleSingle, b: DoubleSingle) -> DoubleSingle {
    let product = a.hi * b.hi;
    let error = fma(a.hi, b.hi, -product) + a.hi * b.lo + a.lo * b.hi;
    return ds_quick_sum(product, error);
}

fn ds_divide(a: DoubleSingle, b: DoubleSingle) -> DoubleSingle {
    let estimate = a.hi / b.hi;
    let remainder = ds_add(a, ds_negate(ds_multiply(b, ds_from(estimate))));
    let correction = ds_value(remainder) / b.hi;
    return ds_quick_sum(estimate, correction);
}

fn alpha_value() -> DoubleSingle {
    return DoubleSingle(state.alpha_hi, state.alpha_lo);
}

fn omega_value() -> DoubleSingle {
    return DoubleSingle(state.omega_hi, state.omega_lo);
}

fn norm_reason(norm_squared: f32) -> f32 {
    let invalid = invalid_reason(norm_squared);
    if (invalid != CONTINUED) {
        return invalid;
    }
    if (norm_squared < 0.0f) {
        return DIVERGED_BREAKDOWN;
    }
    if (norm_squared <= state.atol_squared) {
        return CONVERGED_ATOL;
    }
    if (norm_squared <= state.rtol_squared * state.bnorm_squared) {
        return CONVERGED_RTOL;
    }
    if (norm_squared >= state.dtol_squared * state.bnorm_squared) {
        return DIVERGED_DTOL;
    }
    return CONTINUED;
}

fn stop(reason: f32) {
    state.reason = reason;
    state.running = 0.0f;
}

@compute @workgroup_size(1)
fn begin_iteration() {
    if (state.running < 0.5f) {
        return;
    }
    let rho = reduction[0].x;
    let invalid = invalid_reason(rho);
    if (invalid != CONTINUED) {
        stop(invalid);
        return;
    }
    if (abs(rho) <= MIN_NORMAL) {
        stop(DIVERGED_BREAKDOWN);
        return;
    }
    state.iterations = state.iterations + 1.0f;
    state.rho = rho;
    if (state.iterations < 1.5f) {
        state.beta = 0.0f;
        return;
    }
    if (abs(state.rho_previous) <= MIN_NORMAL || abs(ds_value(omega_value())) <= MIN_NORMAL) {
        stop(DIVERGED_BREAKDOWN);
        return;
    }
    let rho_ratio = ds_divide(ds_from(rho), ds_from(state.rho_previous));
    let alpha_omega_ratio = ds_divide(alpha_value(), omega_value());
    let beta = ds_value(ds_multiply(rho_ratio, alpha_omega_ratio));
    let beta_invalid = invalid_reason(beta);
    if (beta_invalid != CONTINUED) {
        stop(beta_invalid);
        return;
    }
    state.beta = beta;
}

@compute @workgroup_size(1)
fn finish_alpha() {
    if (state.running < 0.5f) {
        return;
    }
    let denominator = reduction[0].x;
    let invalid = invalid_reason(denominator);
    if (invalid != CONTINUED) {
        stop(invalid);
        return;
    }
    if (abs(denominator) <= MIN_NORMAL) {
        stop(DIVERGED_BREAKDOWN);
        return;
    }
    let alpha = ds_divide(ds_from(state.rho), ds_from(denominator));
    let alpha_rounded = ds_value(alpha);
    let alpha_invalid = invalid_reason(alpha_rounded);
    if (alpha_invalid != CONTINUED || abs(alpha_rounded) <= MIN_NORMAL) {
        stop(select(DIVERGED_BREAKDOWN, alpha_invalid, alpha_invalid != CONTINUED));
        return;
    }
    state.alpha_hi = alpha.hi;
    state.alpha_lo = alpha.lo;
}

@compute @workgroup_size(1)
fn check_s() {
    if (state.running < 0.5f) {
        return;
    }
    state.residual_squared = reduction[0].x;
    let reason = norm_reason(state.residual_squared);
    if (reason != CONTINUED) {
        stop(reason);
    }
}

@compute @workgroup_size(1)
fn finish_omega() {
    if (state.running < 0.5f) {
        return;
    }
    let t_dot_t = reduction[0].y;
    let t_dot_s = reduction[0].z;
    let invalid_tt = invalid_reason(t_dot_t);
    let invalid_ts = invalid_reason(t_dot_s);
    if (invalid_tt != CONTINUED || invalid_ts != CONTINUED) {
        stop(select(invalid_tt, invalid_ts, invalid_ts != CONTINUED));
        return;
    }
    if (t_dot_t <= MIN_NORMAL) {
        stop(DIVERGED_BREAKDOWN);
        return;
    }
    let omega = ds_divide(ds_from(t_dot_s), ds_from(t_dot_t));
    let omega_rounded = ds_value(omega);
    let invalid = invalid_reason(omega_rounded);
    if (invalid != CONTINUED || abs(omega_rounded) <= MIN_NORMAL) {
        stop(select(DIVERGED_BREAKDOWN, invalid, invalid != CONTINUED));
        return;
    }
    state.omega_hi = omega.hi;
    state.omega_lo = omega.lo;
}

@compute @workgroup_size(1)
fn check_r() {
    if (state.running < 0.5f) {
        return;
    }
    state.residual_squared = reduction[0].y;
    state.rho_previous = state.rho;
    let reason = norm_reason(state.residual_squared);
    if (reason != CONTINUED) {
        stop(reason);
        return;
    }
    if (state.iterations >= state.max_iterations) {
        stop(DIVERGED_MAX_ITERS);
    }
}
";

const RESIDENT_BATCH_SIZE: usize = 16;
const RESIDENT_STATE_LEN: usize = 16;

struct ResidentPipelines {
    vector_layout: wgpu::BindGroupLayout,
    update_p_omega: wgpu::ComputePipeline,
    prepare_p: wgpu::ComputePipeline,
    make_s: wgpu::ComputePipeline,
    update_x_alpha: wgpu::ComputePipeline,
    update_x_omega: wgpu::ComputePipeline,
    make_r: wgpu::ComputePipeline,
    scalar_layout: wgpu::BindGroupLayout,
    begin_iteration: wgpu::ComputePipeline,
    finish_alpha: wgpu::ComputePipeline,
    check_s: wgpu::ComputePipeline,
    finish_omega: wgpu::ComputePipeline,
    check_r: wgpu::ComputePipeline,
}

impl ResidentPipelines {
    fn new(runtime: &WgpuRuntime) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let vector_layout =
            runtime
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("kryst resident BiCGSTAB vector layout"),
                    entries: &[
                        storage(0, true),
                        storage(1, true),
                        storage(2, false),
                        storage(3, true),
                    ],
                });
        let vector_pipeline_layout =
            runtime
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("kryst resident BiCGSTAB vector pipeline layout"),
                    bind_group_layouts: &[Some(&vector_layout)],
                    immediate_size: 0,
                });
        let vector_shader = runtime
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kryst resident BiCGSTAB vector kernels"),
                source: wgpu::ShaderSource::Wgsl(RESIDENT_VECTOR_SHADER.into()),
            });
        let vector_pipeline = |entry_point| {
            runtime
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&vector_pipeline_layout),
                    module: &vector_shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        };

        let scalar_layout =
            runtime
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("kryst resident BiCGSTAB scalar layout"),
                    entries: &[storage(0, false), storage(1, true)],
                });
        let scalar_pipeline_layout =
            runtime
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("kryst resident BiCGSTAB scalar pipeline layout"),
                    bind_group_layouts: &[Some(&scalar_layout)],
                    immediate_size: 0,
                });
        let scalar_shader = runtime
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kryst resident BiCGSTAB scalar kernels"),
                source: wgpu::ShaderSource::Wgsl(RESIDENT_SCALAR_SHADER.into()),
            });
        let scalar_pipeline = |entry_point| {
            runtime
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&scalar_pipeline_layout),
                    module: &scalar_shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        };

        Self {
            update_p_omega: vector_pipeline("update_p_omega"),
            prepare_p: vector_pipeline("prepare_p"),
            make_s: vector_pipeline("make_s"),
            update_x_alpha: vector_pipeline("update_x_alpha"),
            update_x_omega: vector_pipeline("update_x_omega"),
            make_r: vector_pipeline("make_r"),
            vector_layout,
            begin_iteration: scalar_pipeline("begin_iteration"),
            finish_alpha: scalar_pipeline("finish_alpha"),
            check_s: scalar_pipeline("check_s"),
            finish_omega: scalar_pipeline("finish_omega"),
            check_r: scalar_pipeline("check_r"),
            scalar_layout,
        }
    }
}

struct ReductionWorkspace {
    partials: wgpu::Buffer,
    result: wgpu::Buffer,
    partial_count: usize,
}

struct BiCgStabWorkspace {
    n: usize,
    r: WgpuVector,
    r_hat: WgpuVector,
    p: WgpuVector,
    v: WgpuVector,
    s: WgpuVector,
    t: WgpuVector,
    z_p: WgpuVector,
    z_s: WgpuVector,
    ax: WgpuVector,
    reduction: ReductionWorkspace,
    state: wgpu::Buffer,
}

impl BiCgStabWorkspace {
    fn new(runtime: Arc<WgpuRuntime>, n: usize) -> Result<Self, KError> {
        let partial_count = n.div_ceil(256).max(1);
        let partial_bytes =
            u64::try_from(partial_count.saturating_mul(4 * std::mem::size_of::<f32>()))
                .map_err(|_| KError::InvalidInput("WebGPU reduction size exceeds u64".into()))?;
        Ok(Self {
            n,
            r: WgpuVector::zeros(runtime.clone(), n)?,
            r_hat: WgpuVector::zeros(runtime.clone(), n)?,
            p: WgpuVector::zeros(runtime.clone(), n)?,
            v: WgpuVector::zeros(runtime.clone(), n)?,
            s: WgpuVector::zeros(runtime.clone(), n)?,
            t: WgpuVector::zeros(runtime.clone(), n)?,
            z_p: WgpuVector::zeros(runtime.clone(), n)?,
            z_s: WgpuVector::zeros(runtime.clone(), n)?,
            ax: WgpuVector::zeros(runtime.clone(), n)?,
            reduction: ReductionWorkspace {
                partials: runtime
                    .create_empty_storage_buffer("kryst WebGPU reduction partials", partial_bytes),
                result: runtime.create_empty_storage_buffer(
                    "kryst WebGPU reduction result",
                    4 * std::mem::size_of::<f32>() as u64,
                ),
                partial_count,
            },
            state: runtime.create_empty_storage_buffer(
                "kryst resident BiCGSTAB state",
                16 * std::mem::size_of::<f32>() as u64,
            ),
        })
    }
}

enum WgpuPreconditioner {
    None,
    Jacobi(WgpuJacobi),
}

/// Device-resident real BiCGSTAB context for portable WebGPU devices.
pub struct WgpuKspContext {
    runtime: Arc<WgpuRuntime>,
    operator: Option<Arc<WgpuCsrOp>>,
    pc_type: PcType,
    preconditioner: Option<WgpuPreconditioner>,
    custom_preconditioner: Option<Box<dyn super::preconditioner::WgpuPreconditioner>>,
    convergence: Convergence,
    workspace: Option<BiCgStabWorkspace>,
    vector_pipeline: wgpu::ComputePipeline,
    vector_params: wgpu::Buffer,
    reduction_stage_pipeline: wgpu::ComputePipeline,
    reduction_final_pipeline: wgpu::ComputePipeline,
    reduction_bind_group_layout: wgpu::BindGroupLayout,
    reduction_params: wgpu::Buffer,
    resident: ResidentPipelines,
}

impl std::fmt::Debug for WgpuKspContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuKspContext")
            .field("adapter", &self.runtime.adapter_name())
            .field("pc_type", &self.pc_type)
            .field("setup", &self.workspace.is_some())
            .finish_non_exhaustive()
    }
}

impl WgpuKspContext {
    /// Construct a portable context. BiCGSTAB is the only admitted solver in
    /// this first backend checkpoint.
    pub fn new(runtime: Arc<WgpuRuntime>) -> Self {
        let vector_shader = runtime
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kryst WebGPU vector kernels"),
                source: wgpu::ShaderSource::Wgsl(VECTOR_SHADER.into()),
            });
        let vector_pipeline =
            runtime
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("kryst WebGPU axpby"),
                    layout: None,
                    module: &vector_shader,
                    entry_point: Some("axpby"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let vector_params =
            runtime.create_uniform_buffer("kryst WebGPU vector parameters", &[0; 16]);
        let reduction_shader =
            runtime
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("kryst WebGPU reductions"),
                    source: wgpu::ShaderSource::Wgsl(REDUCTION_SHADER.into()),
                });
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let reduction_bind_group_layout =
            runtime
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("kryst WebGPU reduction layout"),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, true),
                        storage_entry(4, false),
                        storage_entry(5, false),
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        storage_entry(7, true),
                        storage_entry(8, true),
                    ],
                });
        let reduction_pipeline_layout =
            runtime
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("kryst WebGPU reduction pipeline layout"),
                    bind_group_layouts: &[Some(&reduction_bind_group_layout)],
                    immediate_size: 0,
                });
        let reduction_stage_pipeline =
            runtime
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("kryst WebGPU reduction stage"),
                    layout: Some(&reduction_pipeline_layout),
                    module: &reduction_shader,
                    entry_point: Some("reduce_stage"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let reduction_final_pipeline =
            runtime
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("kryst WebGPU reduction final"),
                    layout: Some(&reduction_pipeline_layout),
                    module: &reduction_shader,
                    entry_point: Some("reduce_final"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let reduction_params =
            runtime.create_uniform_buffer("kryst WebGPU reduction parameters", &[0; 16]);
        let resident = ResidentPipelines::new(&runtime);
        Self {
            runtime,
            operator: None,
            pc_type: PcType::None,
            preconditioner: None,
            custom_preconditioner: None,
            convergence: Convergence::new(1.0e-5, 1.0e-30, 1.0e5, 10_000),
            workspace: None,
            vector_pipeline,
            vector_params,
            reduction_stage_pipeline,
            reduction_final_pipeline,
            reduction_bind_group_layout,
            reduction_params,
            resident,
        }
    }

    /// Confirm that the requested solver is the currently supported
    /// device-resident BiCGSTAB path.
    pub fn set_type(&mut self, solver_type: SolverType) -> Result<&mut Self, KError> {
        if solver_type != SolverType::BiCgStab {
            return Err(KError::Unsupported(
                "WebGPU currently supports only BiCGStab",
            ));
        }
        Ok(self)
    }

    /// Select no preconditioner or right Jacobi scaling.
    pub fn set_pc_type(&mut self, pc_type: PcType) -> Result<&mut Self, KError> {
        if !matches!(pc_type, PcType::None | PcType::Jacobi) {
            return Err(KError::Unsupported(
                "WebGPU currently supports only None and Jacobi preconditioners",
            ));
        }
        self.pc_type = pc_type;
        self.custom_preconditioner = None;
        self.preconditioner = None;
        self.workspace = None;
        Ok(self)
    }

    /// Install a caller-owned resident WebGPU preconditioner.
    ///
    /// The object is retained across repeated solves and operator-value updates. Its dimensions
    /// are checked against the registered operator during setup.
    pub fn set_preconditioner(
        &mut self,
        preconditioner: Box<dyn super::preconditioner::WgpuPreconditioner>,
    ) -> &mut Self {
        self.custom_preconditioner = Some(preconditioner);
        self.preconditioner = None;
        self.workspace = None;
        self
    }

    /// Set the single square CSR operator used for both the equation and
    /// preconditioning matrix.
    pub fn set_operator(&mut self, operator: Arc<WgpuCsrOp>) -> Result<&mut Self, KError> {
        if !Arc::ptr_eq(&self.runtime, operator.runtime()) {
            return Err(KError::InvalidInput(
                "WebGPU context and operator belong to different runtimes".into(),
            ));
        }
        let (rows, cols) = operator.dims();
        if rows != cols {
            return Err(KError::InvalidInput(format!(
                "WebGPU BiCGStab requires a square operator, got {rows}x{cols}"
            )));
        }
        self.operator = Some(operator);
        self.preconditioner = None;
        self.workspace = None;
        Ok(self)
    }

    /// Configure finite convergence tolerances and an iteration bound.
    pub fn set_tolerances(
        &mut self,
        rtol: f64,
        atol: f64,
        dtol: f64,
        max_iters: usize,
    ) -> Result<&mut Self, KError> {
        if !rtol.is_finite()
            || !atol.is_finite()
            || !dtol.is_finite()
            || rtol < 0.0
            || atol < 0.0
            || dtol <= 0.0
            || max_iters == 0
        {
            return Err(KError::InvalidInput(
                "WebGPU solver tolerances must be finite/non-negative, dtol positive, and max_iters nonzero"
                    .into(),
            ));
        }
        self.convergence = Convergence::new(rtol, atol, dtol, max_iters);
        Ok(self)
    }

    /// Prepare the selected preconditioner and reusable device workspace.
    pub fn setup(&mut self) -> Result<&mut Self, KError> {
        let operator = self
            .operator
            .as_ref()
            .ok_or_else(|| KError::InvalidInput("WebGPU operator is not set".into()))?;
        let n = operator.dims().0;
        if let Some(preconditioner) = &self.custom_preconditioner {
            if preconditioner.dims() != (n, n) {
                return Err(KError::InvalidInput(format!(
                    "custom WebGPU preconditioner dimensions {:?} do not match operator dimensions ({n}, {n})",
                    preconditioner.dims()
                )));
            }
        } else if self.preconditioner.is_none() {
            self.preconditioner = Some(match self.pc_type {
                PcType::None => WgpuPreconditioner::None,
                PcType::Jacobi => WgpuPreconditioner::Jacobi(WgpuJacobi::from_csr(operator)?),
                _ => {
                    return Err(KError::Unsupported(
                        "selected preconditioner has no WebGPU implementation",
                    ));
                }
            });
        }
        if self.workspace.as_ref().map(|work| work.n) != Some(n) {
            self.workspace = Some(BiCgStabWorkspace::new(self.runtime.clone(), n)?);
        }
        Ok(self)
    }

    /// Solve one device-resident system. Matrix, Krylov vectors, scalar recurrence state, and
    /// convergence decisions stay on the selected GPU. The host reads the compact solver state
    /// once per bounded batch and performs one explicit true-residual check at the end.
    pub async fn solve(
        &mut self,
        b: &WgpuVector,
        x: &WgpuVector,
    ) -> Result<SolveStats<f64>, KError> {
        self.setup()?;
        let operator = self
            .operator
            .as_ref()
            .expect("setup checked operator")
            .clone();
        if b.len() != operator.dims().0 || x.len() != operator.dims().1 {
            return Err(KError::InvalidInput(format!(
                "WebGPU solve dimensions require b={}, x={}; got b={}, x={}",
                operator.dims().0,
                operator.dims().1,
                b.len(),
                x.len()
            )));
        }
        b.ensure_compatible(x)?;
        let mut workspace = self.workspace.take().ok_or_else(|| {
            KError::SolveError("WebGPU solver workspace was not initialized".into())
        })?;
        let result = self.solve_bicgstab(&operator, b, x, &mut workspace).await;
        self.workspace = Some(workspace);
        result
    }

    /// Atomic host convenience wrapper. The supplied `f64` solution changes
    /// only after a converged device solve has been downloaded successfully.
    pub async fn solve_host(
        &mut self,
        b: &[f64],
        x: &mut [f64],
    ) -> Result<SolveStats<f64>, KError> {
        let b_device = WgpuVector::from_host(self.runtime.clone(), b)?;
        let x_device = WgpuVector::from_host(self.runtime.clone(), x)?;
        let stats = self.solve(&b_device, &x_device).await?;
        if stats.reason.is_converged() {
            let candidate = x_device.to_host().await?;
            x.copy_from_slice(&candidate);
        }
        Ok(stats)
    }

    async fn solve_bicgstab(
        &self,
        operator: &WgpuCsrOp,
        b: &WgpuVector,
        x: &WgpuVector,
        workspace: &mut BiCgStabWorkspace,
    ) -> Result<SolveStats<f64>, KError> {
        let BiCgStabWorkspace {
            r,
            r_hat,
            p,
            v,
            s,
            t,
            z_p,
            z_s,
            ax,
            reduction,
            state,
            ..
        } = workspace;

        operator.apply(x, ax)?;
        self.copy(b, r)?;
        self.axpby(-1.0, ax, 1.0, r)?;
        self.copy(r, r_hat)?;
        self.copy(r, p)?;

        let initial = self.dot2(b, b, r, r, reduction).await?;
        let bnorm = checked_norm(initial[0], "WebGPU BiCGStab right-hand side norm")?;
        let mut rnorm = checked_norm(initial[1], "WebGPU BiCGStab residual norm")?;
        let mut reductions = 1usize;
        let (initial_reason, mut initial_stats) = self.convergence.check(rnorm, bnorm, 0);
        if initial_reason != ConvergedReason::Continued {
            initial_stats.final_true_residual = Some(rnorm);
            initial_stats.final_recurrence_residual = Some(rnorm);
            initial_stats.counters = SolverCounters {
                num_global_reductions: reductions,
                ..SolverCounters::default()
            };
            return Ok(initial_stats.finalize_reason_counters());
        }

        self.initialize_resident_state(state, initial[0], initial[1])?;
        self.prepare_reduction(reduction, r.len())?;

        let mut state_snapshot = vec![0.0_f32; RESIDENT_STATE_LEN];
        let mut batch_readbacks = 0usize;
        loop {
            let completed = state_snapshot[13].max(0.0).round() as usize;
            let batch_size = self
                .convergence
                .max_iters
                .saturating_sub(completed)
                .min(RESIDENT_BATCH_SIZE);
            if batch_size == 0 {
                break;
            }
            let mut encoder =
                self.runtime
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("kryst resident BiCGSTAB batch"),
                    });
            for _ in 0..batch_size {
                self.encode_resident_iteration(
                    &mut encoder,
                    operator,
                    state,
                    x,
                    r,
                    r_hat,
                    p,
                    v,
                    s,
                    t,
                    z_p,
                    z_s,
                    reduction,
                )?;
            }
            self.runtime.queue().submit([encoder.finish()]);
            state_snapshot = self
                .runtime
                .read_f32(
                    state,
                    RESIDENT_STATE_LEN,
                    "read resident BiCGSTAB batch state",
                )
                .await?;
            batch_readbacks += 1;
            if state_snapshot.iter().any(|value| !value.is_finite()) {
                return Err(KError::NonFiniteReduction {
                    kind: if state_snapshot.iter().any(|value| value.is_nan()) {
                        crate::error::NonFiniteKind::Nan
                    } else {
                        crate::error::NonFiniteKind::Inf
                    },
                    context: "resident WebGPU BiCGSTAB state",
                });
            }
            if state_snapshot[12] < 0.5 {
                break;
            }
        }

        let iterations = state_snapshot[13].max(0.0).round() as usize;
        let mut final_reason = resident_reason(state_snapshot[14]);
        if final_reason == ConvergedReason::Continued {
            final_reason = ConvergedReason::DivergedMaxIts;
        }
        rnorm = checked_norm(
            f64::from(state_snapshot[8]),
            "resident WebGPU BiCGStab residual norm",
        )?;
        reductions += 3usize.saturating_mul(iterations);

        operator.apply(x, ax)?;
        self.copy(b, r)?;
        self.axpby(-1.0, ax, 1.0, r)?;
        let true_residual = checked_norm(
            self.dot2(r, r, r, r, reduction).await?[0],
            "WebGPU BiCGStab true residual norm",
        )?;
        reductions += 1;
        if final_reason == ConvergedReason::DivergedMaxIts {
            let (reason, _) = self.convergence.check(true_residual, bnorm, iterations);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
            }
        }
        let mut stats = SolveStats::new(iterations, true_residual, final_reason);
        stats.final_recurrence_residual = Some(rnorm);
        stats.final_true_residual = Some(true_residual);
        stats.counters = SolverCounters {
            num_global_reductions: reductions,
            ..SolverCounters::default()
        };
        stats.reduction_model = Some(ReductionModel {
            variant: "wgpu-resident-batched-three-reduction-bicgstab",
            startup: 1,
            per_iteration: 3.0,
            tail: 1,
        });
        stats.effective_variant = Some(format!(
            "wgpu-resident-right-bicgstab-batch-{RESIDENT_BATCH_SIZE}-readbacks-{batch_readbacks}"
        ));
        Ok(stats.finalize_reason_counters())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_resident_iteration(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        operator: &WgpuCsrOp,
        state: &wgpu::Buffer,
        x: &WgpuVector,
        r: &WgpuVector,
        r_hat: &WgpuVector,
        p: &WgpuVector,
        v: &WgpuVector,
        s: &WgpuVector,
        t: &WgpuVector,
        z_p: &WgpuVector,
        z_s: &WgpuVector,
        reduction: &ReductionWorkspace,
    ) -> Result<(), KError> {
        self.encode_dot3(encoder, r_hat, r, r, r, r, r, reduction)?;
        self.encode_scalar(encoder, &self.resident.begin_iteration, state, reduction);
        self.encode_vector(encoder, &self.resident.update_p_omega, state, r, p, v)?;
        self.encode_vector(encoder, &self.resident.prepare_p, state, r, p, v)?;
        self.encode_preconditioner(encoder, p, z_p)?;
        operator.encode_apply(encoder, z_p, v)?;
        self.encode_dot3(encoder, r_hat, v, r_hat, v, r_hat, v, reduction)?;
        self.encode_scalar(encoder, &self.resident.finish_alpha, state, reduction);
        self.encode_vector(encoder, &self.resident.make_s, state, r, s, v)?;
        self.encode_vector(encoder, &self.resident.update_x_alpha, state, z_p, x, t)?;
        self.encode_preconditioner(encoder, s, z_s)?;
        operator.encode_apply(encoder, z_s, t)?;
        self.encode_dot3(encoder, s, s, t, t, t, s, reduction)?;
        self.encode_scalar(encoder, &self.resident.check_s, state, reduction);
        self.encode_scalar(encoder, &self.resident.finish_omega, state, reduction);
        self.encode_vector(encoder, &self.resident.update_x_omega, state, z_s, x, t)?;
        self.encode_vector(encoder, &self.resident.make_r, state, s, r, t)?;
        self.encode_dot3(encoder, r_hat, r, r, r, r, r, reduction)?;
        self.encode_scalar(encoder, &self.resident.check_r, state, reduction);
        Ok(())
    }

    fn initialize_resident_state(
        &self,
        state: &wgpu::Buffer,
        bnorm_squared: f64,
        residual_squared: f64,
    ) -> Result<(), KError> {
        const MAX_EXACT_F32_INTEGER: usize = 1 << 24;
        if self.convergence.max_iters > MAX_EXACT_F32_INTEGER {
            return Err(KError::InvalidInput(format!(
                "WebGPU resident iteration limit {} exceeds exact f32 control range",
                self.convergence.max_iters
            )));
        }
        let rtol_squared = (self.convergence.rtol * self.convergence.rtol) as f32;
        let atol_squared = (self.convergence.atol * self.convergence.atol) as f32;
        let dtol_squared = (self.convergence.dtol * self.convergence.dtol) as f32;
        if !rtol_squared.is_finite() || !atol_squared.is_finite() || !dtol_squared.is_finite() {
            return Err(KError::InvalidInput(
                "WebGPU squared solver tolerances exceed portable f32 range".into(),
            ));
        }
        let values = [
            residual_squared as f32,
            residual_squared as f32,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            bnorm_squared as f32,
            residual_squared as f32,
            rtol_squared,
            atol_squared,
            dtol_squared,
            1.0,
            0.0,
            0.0,
            self.convergence.max_iters as f32,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(KError::InvalidInput(
                "WebGPU resident solver state exceeds portable f32 range".into(),
            ));
        }
        self.runtime
            .queue()
            .write_buffer(state, 0, &encode_f32(values));
        Ok(())
    }

    fn prepare_reduction(&self, reduction: &ReductionWorkspace, n: usize) -> Result<(), KError> {
        let params = encode_u32([
            u32::try_from(n)
                .map_err(|_| KError::InvalidInput("WebGPU vector length exceeds u32".into()))?,
            u32::try_from(reduction.partial_count)
                .map_err(|_| KError::InvalidInput("WebGPU reduction count exceeds u32".into()))?,
            0,
            0,
        ]);
        self.runtime
            .queue()
            .write_buffer(&self.reduction_params, 0, &params);
        Ok(())
    }

    fn encode_preconditioner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: &WgpuVector,
        y: &WgpuVector,
    ) -> Result<(), KError> {
        if let Some(preconditioner) = &self.custom_preconditioner {
            return preconditioner.encode_apply(encoder, x, y);
        }
        match self
            .preconditioner
            .as_ref()
            .expect("setup checked preconditioner")
        {
            WgpuPreconditioner::None => {
                x.ensure_compatible(y)?;
                encoder.copy_buffer_to_buffer(x.buffer(), 0, y.buffer(), 0, x.byte_len());
                Ok(())
            }
            WgpuPreconditioner::Jacobi(jacobi) => jacobi.encode_apply(encoder, x, y),
        }
    }

    fn encode_vector(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        state: &wgpu::Buffer,
        input_a: &WgpuVector,
        output: &WgpuVector,
        input_c: &WgpuVector,
    ) -> Result<(), KError> {
        input_a.ensure_compatible(output)?;
        input_a.ensure_compatible(input_c)?;
        let bind_group = self
            .runtime
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kryst resident BiCGSTAB vector operation"),
                layout: &self.resident.vector_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_a.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input_c.buffer().as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kryst resident BiCGSTAB vector operation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            u32::try_from(output.len().div_ceil(128)).map_err(|_| {
                KError::InvalidInput("WebGPU vector size exceeds dispatch range".into())
            })?,
            1,
            1,
        );
        Ok(())
    }

    fn encode_scalar(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        state: &wgpu::Buffer,
        reduction: &ReductionWorkspace,
    ) {
        let bind_group = self
            .runtime
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kryst resident BiCGSTAB scalar operation"),
                layout: &self.resident.scalar_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: reduction.result.as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kryst resident BiCGSTAB scalar operation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dot3(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x0: &WgpuVector,
        y0: &WgpuVector,
        x1: &WgpuVector,
        y1: &WgpuVector,
        x2: &WgpuVector,
        y2: &WgpuVector,
        reduction: &ReductionWorkspace,
    ) -> Result<(), KError> {
        x0.ensure_compatible(y0)?;
        x0.ensure_compatible(x1)?;
        x0.ensure_compatible(y1)?;
        x0.ensure_compatible(x2)?;
        x0.ensure_compatible(y2)?;
        let bind_group = self
            .runtime
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kryst resident WebGPU dot2"),
                layout: &self.reduction_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x0.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y0.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: x1.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: y1.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: reduction.partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: reduction.result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.reduction_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: x2.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: y2.buffer().as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kryst resident WebGPU dot2"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.reduction_stage_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            u32::try_from(reduction.partial_count)
                .map_err(|_| KError::InvalidInput("WebGPU reduction count exceeds u32".into()))?,
            1,
            1,
        );
        pass.set_pipeline(&self.reduction_final_pipeline);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    fn copy(&self, source: &WgpuVector, destination: &WgpuVector) -> Result<(), KError> {
        source.ensure_compatible(destination)?;
        self.runtime.copy_buffer(
            source.buffer(),
            destination.buffer(),
            source.byte_len(),
            "copy WebGPU vector",
        );
        Ok(())
    }

    fn axpby(&self, alpha: f64, x: &WgpuVector, beta: f64, y: &WgpuVector) -> Result<(), KError> {
        x.ensure_compatible(y)?;
        let alpha = alpha as f32;
        let beta = beta as f32;
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(KError::InvalidInput(
                "WebGPU axpby coefficient exceeds portable f32 range".into(),
            ));
        }
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&alpha.to_ne_bytes());
        params.extend_from_slice(&beta.to_ne_bytes());
        params.extend_from_slice(
            &u32::try_from(x.len())
                .map_err(|_| KError::InvalidInput("WebGPU vector length exceeds u32".into()))?
                .to_ne_bytes(),
        );
        params.extend_from_slice(&0_u32.to_ne_bytes());
        self.runtime
            .queue()
            .write_buffer(&self.vector_params, 0, &params);
        let bind_group = self
            .runtime
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kryst WebGPU axpby"),
                layout: &self.vector_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.vector_params.as_entire_binding(),
                    },
                ],
            });
        let mut encoder =
            self.runtime
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kryst WebGPU axpby"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kryst WebGPU axpby"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vector_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                u32::try_from(x.len().div_ceil(128)).map_err(|_| {
                    KError::InvalidInput("WebGPU vector size exceeds dispatch range".into())
                })?,
                1,
                1,
            );
        }
        self.runtime.queue().submit([encoder.finish()]);
        Ok(())
    }

    async fn dot2(
        &self,
        x0: &WgpuVector,
        y0: &WgpuVector,
        x1: &WgpuVector,
        y1: &WgpuVector,
        reduction: &ReductionWorkspace,
    ) -> Result<[f64; 2], KError> {
        let result = self.dot3(x0, y0, x1, y1, x1, y1, reduction).await?;
        Ok([result[0], result[1]])
    }

    async fn dot3(
        &self,
        x0: &WgpuVector,
        y0: &WgpuVector,
        x1: &WgpuVector,
        y1: &WgpuVector,
        x2: &WgpuVector,
        y2: &WgpuVector,
        reduction: &ReductionWorkspace,
    ) -> Result<[f64; 3], KError> {
        x0.ensure_compatible(y0)?;
        x0.ensure_compatible(x1)?;
        x0.ensure_compatible(y1)?;
        x0.ensure_compatible(x2)?;
        x0.ensure_compatible(y2)?;
        let params = encode_u32([
            u32::try_from(x0.len())
                .map_err(|_| KError::InvalidInput("WebGPU vector length exceeds u32".into()))?,
            u32::try_from(reduction.partial_count)
                .map_err(|_| KError::InvalidInput("WebGPU reduction count exceeds u32".into()))?,
            0,
            0,
        ]);
        self.runtime
            .queue()
            .write_buffer(&self.reduction_params, 0, &params);
        let bind_group = self
            .runtime
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kryst WebGPU dot3"),
                layout: &self.reduction_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x0.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y0.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: x1.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: y1.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: reduction.partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: reduction.result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.reduction_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: x2.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: y2.buffer().as_entire_binding(),
                    },
                ],
            });
        let mut encoder =
            self.runtime
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kryst WebGPU dot3"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kryst WebGPU dot3"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduction_stage_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                u32::try_from(reduction.partial_count).map_err(|_| {
                    KError::InvalidInput("WebGPU reduction count exceeds u32".into())
                })?,
                1,
                1,
            );
            pass.set_pipeline(&self.reduction_final_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.runtime.queue().submit([encoder.finish()]);
        let result = self
            .runtime
            .read_f32(&reduction.result, 4, "read WebGPU dot3")
            .await?;
        let result = [
            f64::from(result[0]),
            f64::from(result[1]),
            f64::from(result[2]),
        ];
        if result.iter().any(|value| !value.is_finite()) {
            return Err(KError::NonFiniteReduction {
                kind: if result.iter().any(|value| value.is_nan()) {
                    crate::error::NonFiniteKind::Nan
                } else {
                    crate::error::NonFiniteKind::Inf
                },
                context: "WebGPU dot3",
            });
        }
        Ok(result)
    }
}

fn checked_norm(value: f64, context: &'static str) -> Result<f64, KError> {
    if !value.is_finite() {
        return Err(KError::NonFiniteReduction {
            kind: if value.is_nan() {
                crate::error::NonFiniteKind::Nan
            } else {
                crate::error::NonFiniteKind::Inf
            },
            context,
        });
    }
    if value < 0.0 {
        return Err(KError::SolveError(format!(
            "{context} squared norm is negative ({value:e})"
        )));
    }
    Ok(value.sqrt())
}

fn resident_reason(code: f32) -> ConvergedReason {
    match code.round() as i32 {
        1 => ConvergedReason::ConvergedRtol,
        2 => ConvergedReason::ConvergedAtol,
        3 => ConvergedReason::DivergedDtol,
        4 => ConvergedReason::DivergedMaxIts,
        5 => ConvergedReason::DivergedBreakdownBiCG,
        6 => ConvergedReason::DivergedNan,
        7 => ConvergedReason::DivergedInf,
        _ => ConvergedReason::Continued,
    }
}
