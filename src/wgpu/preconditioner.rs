use super::operator::WgpuCsrOp;
use super::runtime::{WgpuRuntime, encode_f32};
use super::vector::WgpuVector;
use crate::error::{KError, WgpuErrorKind};
use std::sync::Arc;

const JACOBI_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> inverse_diagonal: array<f32>;
@group(0) @binding(1) var<storage, read> input_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;

@compute @workgroup_size(128)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&input_values)) {
        output_values[id.x] = inverse_diagonal[id.x] * input_values[id.x];
    }
}
";

/// Caller-owned preconditioner operating entirely on resident WebGPU vectors.
///
/// Implementations may submit one or more command buffers from [`Self::apply`], but must not map
/// or read back the vectors. Queue ordering makes the result visible to the following resident
/// Krylov kernel.
pub trait WgpuPreconditioner: Send + Sync {
    /// Square dimensions of the approximate inverse.
    fn dims(&self) -> (usize, usize);

    /// Apply the approximate inverse, writing all output entries.
    fn apply(&self, input: &WgpuVector, output: &WgpuVector) -> Result<(), KError>;
}

pub(crate) struct WgpuJacobi {
    runtime: Arc<WgpuRuntime>,
    inverse_diagonal: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    n: usize,
}

impl WgpuJacobi {
    pub(crate) fn from_csr(matrix: &WgpuCsrOp) -> Result<Self, KError> {
        let (n, cols) = matrix.dims();
        if n != cols {
            return Err(KError::InvalidInput(format!(
                "WebGPU Jacobi requires a square operator, got {n}x{cols}"
            )));
        }
        let inverse = matrix
            .diagonal_host()
            .iter()
            .enumerate()
            .map(|(row, &value)| {
                if !value.is_finite() || value.abs() <= 1.0e-30 {
                    return Err(KError::ZeroPivot(row));
                }
                let inverse = (1.0 / value) as f32;
                if !inverse.is_finite() {
                    return Err(KError::InvalidInput(format!(
                        "WebGPU Jacobi inverse diagonal at row {row} exceeds f32 range"
                    )));
                }
                Ok(inverse)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let runtime = matrix.runtime().clone();
        let inverse_diagonal =
            runtime.create_storage_buffer("kryst WebGPU inverse diagonal", &encode_f32(inverse));
        let shader = runtime
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kryst WebGPU Jacobi"),
                source: wgpu::ShaderSource::Wgsl(JACOBI_SHADER.into()),
            });
        let pipeline = runtime
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("kryst WebGPU Jacobi"),
                layout: None,
                module: &shader,
                entry_point: Some("apply"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Ok(Self {
            runtime,
            inverse_diagonal,
            pipeline,
            n,
        })
    }

    pub(crate) fn encode_apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: &WgpuVector,
        y: &WgpuVector,
    ) -> Result<(), KError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(KError::InvalidInput(format!(
                "WebGPU Jacobi requires vectors of length {}; got {} and {}",
                self.n,
                x.len(),
                y.len()
            )));
        }
        if !Arc::ptr_eq(&self.runtime, x.runtime()) || !Arc::ptr_eq(&self.runtime, y.runtime()) {
            return Err(super::runtime::wgpu_error_message(
                WgpuErrorKind::DeviceMismatch,
                "apply WebGPU Jacobi",
                "preconditioner and vectors belong to different WebGPU runtimes",
            ));
        }
        let bind_group = self
            .runtime
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kryst WebGPU Jacobi"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.inverse_diagonal.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: y.buffer().as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kryst WebGPU Jacobi"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                u32::try_from(self.n.div_ceil(128)).map_err(|_| {
                    KError::InvalidInput("WebGPU Jacobi size exceeds dispatch range".into())
                })?,
                1,
                1,
            );
        }
        Ok(())
    }

    pub(crate) fn apply(&self, x: &WgpuVector, y: &WgpuVector) -> Result<(), KError> {
        let mut encoder =
            self.runtime
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kryst WebGPU Jacobi"),
                });
        self.encode_apply(&mut encoder, x, y)?;
        self.runtime.queue().submit([encoder.finish()]);
        Ok(())
    }
}
