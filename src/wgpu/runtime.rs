use crate::error::{KError, WgpuErrorKind};
use futures_channel::oneshot;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// One selected portable WebGPU device and queue.
pub struct WgpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
}

impl std::fmt::Debug for WgpuRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuRuntime")
            .field("adapter_name", &self.adapter_name)
            .finish_non_exhaustive()
    }
}

impl WgpuRuntime {
    /// Select a high-performance portable WebGPU adapter and request its
    /// default limits.
    pub async fn request() -> Result<Arc<Self>, KError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|error| {
                wgpu_error(WgpuErrorKind::Unavailable, "request WebGPU adapter", error)
            })?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kryst WebGPU device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| {
                wgpu_error(WgpuErrorKind::Unavailable, "request WebGPU device", error)
            })?;
        Ok(Arc::new(Self {
            device,
            queue,
            adapter_name: format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type),
        }))
    }

    /// Human-readable selected adapter identity.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Underlying device for composing feature-specific kernels with Kryst's resident vectors.
    ///
    /// Buffers passed back into Kryst must have been created from this exact runtime.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Underlying queue for composing feature-specific kernels with a Kryst solve.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn create_storage_buffer(
        &self,
        label: &'static str,
        contents: &[u8],
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub(crate) fn create_empty_storage_buffer(
        &self,
        label: &'static str,
        bytes: u64,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn create_uniform_buffer(
        &self,
        label: &'static str,
        contents: &[u8],
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub(crate) async fn read_f32(
        &self,
        source: &wgpu::Buffer,
        count: usize,
        operation: &'static str,
    ) -> Result<Vec<f32>, KError> {
        let byte_count = u64::try_from(count.saturating_mul(std::mem::size_of::<f32>()))
            .map_err(|_| KError::InvalidInput("WebGPU readback size exceeds u64".into()))?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kryst WebGPU readback"),
            size: byte_count.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(operation),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_count);
        let submission = self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..byte_count);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| wgpu_error(WgpuErrorKind::Synchronization, operation, error))?;
        #[cfg(target_arch = "wasm32")]
        let _submission = submission;
        receiver
            .await
            .map_err(|_| {
                wgpu_error_message(
                    WgpuErrorKind::Synchronization,
                    operation,
                    "readback callback was dropped",
                )
            })?
            .map_err(|error| wgpu_error(WgpuErrorKind::Synchronization, operation, error))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| wgpu_error(WgpuErrorKind::Synchronization, operation, error))?;
        let result = mapped
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("one f32")))
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(result)
    }

    pub(crate) fn copy_buffer(
        &self,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        bytes: u64,
        label: &'static str,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, bytes);
        self.queue.submit([encoder.finish()]);
    }
}

pub(crate) fn encode_u32(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    let values = values.into_iter();
    let mut encoded = Vec::with_capacity(values.size_hint().0 * std::mem::size_of::<u32>());
    for value in values {
        encoded.extend_from_slice(&value.to_ne_bytes());
    }
    encoded
}

pub(crate) fn encode_f32(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    let values = values.into_iter();
    let mut encoded = Vec::with_capacity(values.size_hint().0 * std::mem::size_of::<f32>());
    for value in values {
        encoded.extend_from_slice(&value.to_ne_bytes());
    }
    encoded
}

pub(crate) fn wgpu_error(
    kind: WgpuErrorKind,
    operation: &'static str,
    error: impl std::fmt::Display,
) -> KError {
    wgpu_error_message(kind, operation, error.to_string())
}

pub(crate) fn wgpu_error_message(
    kind: WgpuErrorKind,
    operation: &'static str,
    message: impl Into<String>,
) -> KError {
    KError::Wgpu {
        kind,
        operation,
        message: message.into(),
    }
}
