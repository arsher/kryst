use super::runtime::{WgpuRuntime, encode_f32, encode_u32};
use super::vector::WgpuVector;
use crate::error::{KError, WgpuErrorKind};
use crate::matrix::sparse::CsrMatrix;
use std::sync::Arc;

const CSR_SPMV_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> column_indices: array<u32>;
@group(0) @binding(2) var<storage, read> matrix_values: array<f32>;
@group(0) @binding(3) var<storage, read> input_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_values: array<f32>;

@compute @workgroup_size(128)
fn spmv(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row + 1u >= arrayLength(&row_offsets)) {
        return;
    }
    var sum = 0.0f;
    let start = row_offsets[row];
    let end = row_offsets[row + 1u];
    for (var entry = start; entry < end; entry = entry + 1u) {
        sum = fma(matrix_values[entry], input_values[column_indices[entry]], sum);
    }
    output_values[row] = sum;
}
";

/// A validated real CSR operator retained on one portable WebGPU device.
pub struct WgpuCsrOp {
    runtime: Arc<WgpuRuntime>,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    row_offsets: wgpu::Buffer,
    column_indices: wgpu::Buffer,
    values: wgpu::Buffer,
    diagonal_host: Vec<f64>,
    pipeline: wgpu::ComputePipeline,
}

impl std::fmt::Debug for WgpuCsrOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuCsrOp")
            .field("dims", &(self.nrows, self.ncols))
            .field("nnz", &self.nnz)
            .field("adapter", &self.runtime.adapter_name())
            .finish()
    }
}

impl WgpuCsrOp {
    /// Upload one host CSR matrix into portable `f32` device storage.
    pub fn from_host(runtime: Arc<WgpuRuntime>, matrix: &CsrMatrix<f64>) -> Result<Self, KError> {
        Self::from_csr_parts(
            runtime,
            matrix.nrows(),
            matrix.ncols(),
            matrix.row_ptr(),
            matrix.col_idx(),
            matrix.values(),
        )
    }

    /// Validate and upload explicit CSR parts.
    pub fn from_csr_parts(
        runtime: Arc<WgpuRuntime>,
        nrows: usize,
        ncols: usize,
        row_offsets: &[usize],
        column_indices: &[usize],
        values: &[f64],
    ) -> Result<Self, KError> {
        validate_csr(nrows, ncols, row_offsets, column_indices, values)?;
        let encoded_rows = encode_u32(
            row_offsets
                .iter()
                .map(|&value| u32::try_from(value))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| KError::InvalidInput("WebGPU CSR row offset exceeds u32".into()))?,
        );
        let encoded_columns = encode_u32(
            column_indices
                .iter()
                .map(|&value| u32::try_from(value))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| KError::InvalidInput("WebGPU CSR column index exceeds u32".into()))?,
        );
        let encoded_values = encode_f32(values.iter().map(|&value| value as f32));
        let row_buffer = runtime.create_storage_buffer("WebGPU CSR row offsets", &encoded_rows);
        let column_buffer =
            runtime.create_storage_buffer("WebGPU CSR column indices", &encoded_columns);
        let value_buffer = runtime.create_storage_buffer("WebGPU CSR values", &encoded_values);
        let shader = runtime
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kryst WebGPU CSR SpMV"),
                source: wgpu::ShaderSource::Wgsl(CSR_SPMV_SHADER.into()),
            });
        let pipeline = runtime
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("kryst WebGPU CSR SpMV"),
                layout: None,
                module: &shader,
                entry_point: Some("spmv"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Ok(Self {
            runtime,
            nrows,
            ncols,
            nnz: values.len(),
            row_offsets: row_buffer,
            column_indices: column_buffer,
            values: value_buffer,
            diagonal_host: extract_diagonal(nrows, row_offsets, column_indices, values),
            pipeline,
        })
    }

    /// Matrix dimensions `(rows, columns)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of stored nonzero entries.
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub(crate) fn runtime(&self) -> &Arc<WgpuRuntime> {
        &self.runtime
    }

    pub(crate) fn diagonal_host(&self) -> &[f64] {
        &self.diagonal_host
    }

    pub(crate) fn apply(&self, x: &WgpuVector, y: &WgpuVector) -> Result<(), KError> {
        if x.len() != self.ncols || y.len() != self.nrows {
            return Err(KError::InvalidInput(format!(
                "WebGPU CSR product requires x={}, y={}; got x={}, y={}",
                self.ncols,
                self.nrows,
                x.len(),
                y.len()
            )));
        }
        if !Arc::ptr_eq(&self.runtime, x.runtime()) || !Arc::ptr_eq(&self.runtime, y.runtime()) {
            return Err(super::runtime::wgpu_error_message(
                WgpuErrorKind::DeviceMismatch,
                "apply WebGPU CSR operator",
                "operator and vectors belong to different WebGPU runtimes",
            ));
        }
        let bind_group = self
            .runtime
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kryst WebGPU CSR SpMV"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.row_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.column_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: x.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: y.buffer().as_entire_binding(),
                    },
                ],
            });
        let mut encoder =
            self.runtime
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kryst WebGPU CSR SpMV"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kryst WebGPU CSR SpMV"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                u32::try_from(self.nrows.div_ceil(128)).map_err(|_| {
                    KError::InvalidInput("WebGPU CSR row count exceeds dispatch range".into())
                })?,
                1,
                1,
            );
        }
        self.runtime.queue().submit([encoder.finish()]);
        Ok(())
    }
}

fn validate_csr(
    nrows: usize,
    ncols: usize,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[f64],
) -> Result<(), KError> {
    if row_offsets.len() != nrows + 1
        || row_offsets.first().copied() != Some(0)
        || row_offsets.last().copied() != Some(values.len())
        || column_indices.len() != values.len()
    {
        return Err(KError::InvalidInput(
            "invalid WebGPU CSR offsets or value/index lengths".into(),
        ));
    }
    if row_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(KError::InvalidInput(
            "WebGPU CSR row offsets are not monotone".into(),
        ));
    }
    if column_indices.iter().any(|&column| column >= ncols) {
        return Err(KError::InvalidInput(
            "WebGPU CSR column index is out of bounds".into(),
        ));
    }
    if values
        .iter()
        .any(|&value| !value.is_finite() || (value as f32).is_infinite())
    {
        return Err(KError::InvalidInput(
            "WebGPU CSR value is non-finite or exceeds portable f32 range".into(),
        ));
    }
    Ok(())
}

fn extract_diagonal(
    nrows: usize,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[f64],
) -> Vec<f64> {
    (0..nrows)
        .map(|row| {
            (row_offsets[row]..row_offsets[row + 1])
                .find(|&entry| column_indices[entry] == row)
                .map_or(0.0, |entry| values[entry])
        })
        .collect()
}
