use super::runtime::{WgpuRuntime, encode_f32};
use crate::error::{KError, WgpuErrorKind};
use std::sync::Arc;

/// A real vector retained in portable WebGPU storage as `f32`.
pub struct WgpuVector {
    runtime: Arc<WgpuRuntime>,
    buffer: wgpu::Buffer,
    len: usize,
}

impl std::fmt::Debug for WgpuVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuVector")
            .field("len", &self.len)
            .field("adapter", &self.runtime.adapter_name())
            .finish()
    }
}

impl WgpuVector {
    /// Allocate one zero-filled device vector.
    pub fn zeros(runtime: Arc<WgpuRuntime>, len: usize) -> Result<Self, KError> {
        let bytes = u64::try_from(len.saturating_mul(std::mem::size_of::<f32>()))
            .map_err(|_| KError::InvalidInput("WebGPU vector size exceeds u64".into()))?;
        let buffer = runtime.create_empty_storage_buffer("kryst WebGPU vector", bytes);
        let mut encoder =
            runtime
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("zero WebGPU vector"),
                });
        encoder.clear_buffer(&buffer, 0, None);
        runtime.queue().submit([encoder.finish()]);
        Ok(Self {
            runtime,
            buffer,
            len,
        })
    }

    /// Upload an `f64` host vector into portable `f32` device storage.
    pub fn from_host(runtime: Arc<WgpuRuntime>, host: &[f64]) -> Result<Self, KError> {
        validate_host_vector(host)?;
        let encoded = encode_f32(host.iter().map(|&value| value as f32));
        let buffer = runtime.create_storage_buffer("kryst WebGPU vector", &encoded);
        Ok(Self {
            runtime,
            buffer,
            len: host.len(),
        })
    }

    /// Number of retained entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Replace the complete vector from finite host values.
    pub fn copy_from_host(&mut self, host: &[f64]) -> Result<(), KError> {
        if host.len() != self.len {
            return Err(KError::InvalidInput(format!(
                "WebGPU vector upload length mismatch: {} vs {}",
                host.len(),
                self.len
            )));
        }
        validate_host_vector(host)?;
        let encoded = encode_f32(host.iter().map(|&value| value as f32));
        self.runtime.queue().write_buffer(&self.buffer, 0, &encoded);
        Ok(())
    }

    /// Download and widen one complete device vector to host `f64`.
    pub async fn to_host(&self) -> Result<Vec<f64>, KError> {
        Ok(self
            .runtime
            .read_f32(&self.buffer, self.len, "download WebGPU vector")
            .await?
            .into_iter()
            .map(f64::from)
            .collect())
    }

    pub(crate) fn runtime(&self) -> &Arc<WgpuRuntime> {
        &self.runtime
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(crate) fn byte_len(&self) -> u64 {
        u64::try_from(self.len.saturating_mul(std::mem::size_of::<f32>()))
            .expect("validated WebGPU vector byte length")
    }

    pub(crate) fn ensure_compatible(&self, other: &Self) -> Result<(), KError> {
        if !Arc::ptr_eq(&self.runtime, &other.runtime) {
            return Err(super::runtime::wgpu_error_message(
                WgpuErrorKind::DeviceMismatch,
                "validate WebGPU vectors",
                "vectors belong to different WebGPU runtimes",
            ));
        }
        if self.len != other.len {
            return Err(KError::InvalidInput(format!(
                "WebGPU vector length mismatch: {} vs {}",
                self.len, other.len
            )));
        }
        Ok(())
    }
}

fn validate_host_vector(host: &[f64]) -> Result<(), KError> {
    if let Some(index) = host.iter().position(|value| !value.is_finite()) {
        return Err(KError::InvalidInput(format!(
            "WebGPU vector host value {index} is non-finite"
        )));
    }
    if let Some(index) = host.iter().position(|&value| (value as f32).is_infinite()) {
        return Err(KError::InvalidInput(format!(
            "WebGPU vector host value {index} exceeds portable f32 range"
        )));
    }
    Ok(())
}
