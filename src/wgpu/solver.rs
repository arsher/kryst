use super::operator::WgpuCsrOp;
use super::preconditioner::WgpuJacobi;
use super::runtime::{WgpuRuntime, encode_u32};
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
@group(0) @binding(4) var<storage, read_write> partials: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> result: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> params: ReductionParams;

var<workgroup> scratch: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn reduce_stage(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) group_id: vec3<u32>) {
    var value = vec2(0.0f, 0.0f);
    if (global_id.x < params.n) {
        value = vec2(x0[global_id.x] * y0[global_id.x],
                     x1[global_id.x] * y1[global_id.x]);
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
    var value = vec2(0.0f, 0.0f);
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
}

impl BiCgStabWorkspace {
    fn new(runtime: Arc<WgpuRuntime>, n: usize) -> Result<Self, KError> {
        let partial_count = n.div_ceil(256).max(1);
        let partial_bytes =
            u64::try_from(partial_count.saturating_mul(2 * std::mem::size_of::<f32>()))
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
                    2 * std::mem::size_of::<f32>() as u64,
                ),
                partial_count,
            },
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
    convergence: Convergence,
    workspace: Option<BiCgStabWorkspace>,
    vector_pipeline: wgpu::ComputePipeline,
    vector_params: wgpu::Buffer,
    reduction_stage_pipeline: wgpu::ComputePipeline,
    reduction_final_pipeline: wgpu::ComputePipeline,
    reduction_bind_group_layout: wgpu::BindGroupLayout,
    reduction_params: wgpu::Buffer,
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
        Self {
            runtime,
            operator: None,
            pc_type: PcType::None,
            preconditioner: None,
            convergence: Convergence::new(1.0e-5, 1.0e-30, 1.0e5, 10_000),
            workspace: None,
            vector_pipeline,
            vector_params,
            reduction_stage_pipeline,
            reduction_final_pipeline,
            reduction_bind_group_layout,
            reduction_params,
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
        self.preconditioner = None;
        self.workspace = None;
        Ok(self)
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
        if self.preconditioner.is_none() {
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

    /// Solve one device-resident system. Matrix and Krylov vectors stay on the
    /// selected GPU; each convergence reduction returns only two `f32`
    /// scalars to the host controller.
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

        let mut rho_previous = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;
        let mut iterations = 0usize;
        let mut final_reason = ConvergedReason::DivergedMaxIts;

        for iteration in 1..=self.convergence.max_iters {
            let rho = self.dot2(r_hat, r, r_hat, r, reduction).await?[0];
            reductions += 1;
            ensure_nonzero_finite(rho, "WebGPU BiCGStab rho")?;
            if iteration > 1 {
                let beta = (rho / rho_previous) * (alpha / omega);
                ensure_finite(beta, "WebGPU BiCGStab beta")?;
                self.axpby(-omega, v, 1.0, p)?;
                self.axpby(1.0, r, beta, p)?;
            }

            self.apply_preconditioner(p, z_p)?;
            operator.apply(z_p, v)?;
            let alpha_denominator = self.dot2(r_hat, v, r_hat, v, reduction).await?[0];
            reductions += 1;
            ensure_nonzero_finite(alpha_denominator, "WebGPU BiCGStab alpha denominator")?;
            alpha = rho / alpha_denominator;
            ensure_nonzero_finite(alpha, "WebGPU BiCGStab alpha")?;

            self.copy(r, s)?;
            self.axpby(-alpha, v, 1.0, s)?;
            let s_norm = checked_norm(
                self.dot2(s, s, s, s, reduction).await?[0],
                "WebGPU BiCGStab intermediate residual norm",
            )?;
            reductions += 1;
            let (s_reason, _) = self.convergence.check(s_norm, bnorm, iteration);
            if s_reason != ConvergedReason::Continued {
                self.axpby(alpha, z_p, 1.0, x)?;
                iterations = iteration;
                rnorm = s_norm;
                final_reason = s_reason;
                break;
            }

            self.apply_preconditioner(s, z_s)?;
            operator.apply(z_s, t)?;
            let omega_dots = self.dot2(t, t, t, s, reduction).await?;
            reductions += 1;
            if !omega_dots[0].is_finite() || omega_dots[0] <= 0.0 {
                return Err(KError::BreakdownOrIndefinite);
            }
            omega = omega_dots[1] / omega_dots[0];
            ensure_nonzero_finite(omega, "WebGPU BiCGStab omega")?;

            self.axpby(alpha, z_p, 1.0, x)?;
            self.axpby(omega, z_s, 1.0, x)?;
            self.copy(s, r)?;
            self.axpby(-omega, t, 1.0, r)?;
            rnorm = checked_norm(
                self.dot2(r, r, r, r, reduction).await?[0],
                "WebGPU BiCGStab residual norm",
            )?;
            reductions += 1;
            iterations = iteration;
            let (reason, _) = self.convergence.check(rnorm, bnorm, iteration);
            if reason != ConvergedReason::Continued {
                final_reason = reason;
                break;
            }
            rho_previous = rho;
        }

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
            variant: "wgpu-classical-bicgstab",
            startup: 1,
            per_iteration: 5.0,
            tail: 1,
        });
        stats.effective_variant = Some("wgpu-right-preconditioned-bicgstab".into());
        Ok(stats.finalize_reason_counters())
    }

    fn apply_preconditioner(&self, x: &WgpuVector, y: &WgpuVector) -> Result<(), KError> {
        match self
            .preconditioner
            .as_ref()
            .expect("setup checked preconditioner")
        {
            WgpuPreconditioner::None => self.copy(x, y),
            WgpuPreconditioner::Jacobi(jacobi) => jacobi.apply(x, y),
        }
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
        x0.ensure_compatible(y0)?;
        x0.ensure_compatible(x1)?;
        x0.ensure_compatible(y1)?;
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
                label: Some("kryst WebGPU dot2"),
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
                ],
            });
        let mut encoder =
            self.runtime
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kryst WebGPU dot2"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kryst WebGPU dot2"),
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
            .read_f32(&reduction.result, 2, "read WebGPU dot2")
            .await?;
        let result = [f64::from(result[0]), f64::from(result[1])];
        if !result[0].is_finite() || !result[1].is_finite() {
            return Err(KError::NonFiniteReduction {
                kind: if result.iter().any(|value| value.is_nan()) {
                    crate::error::NonFiniteKind::Nan
                } else {
                    crate::error::NonFiniteKind::Inf
                },
                context: "WebGPU dot2",
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
        return Err(KError::BreakdownOrIndefinite);
    }
    Ok(value.sqrt())
}

fn ensure_nonzero_finite(value: f64, context: &'static str) -> Result<(), KError> {
    ensure_finite(value, context)?;
    if value.abs() <= f64::from(f32::MIN_POSITIVE) {
        return Err(KError::SolveError(format!("{context} is zero")));
    }
    Ok(())
}

fn ensure_finite(value: f64, context: &'static str) -> Result<(), KError> {
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
    Ok(())
}
