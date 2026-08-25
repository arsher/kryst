//! Optional portable WebGPU execution support.
//!
//! The first checkpoint deliberately supports real-valued CSR systems,
//! device-resident BiCGSTAB, and right Jacobi preconditioning. Matrix and
//! Krylov vectors remain on the selected Vulkan, Metal, D3D12, or browser
//! WebGPU device during a solve; only scalar reductions cross the host/device
//! boundary. The ordinary host-slice [`crate::context::KspContext`] API is
//! unchanged.

mod operator;
mod preconditioner;
mod runtime;
mod solver;
mod vector;

pub use operator::WgpuCsrOp;
pub use preconditioner::WgpuPreconditioner;
pub use runtime::WgpuRuntime;
pub use solver::WgpuKspContext;
pub use vector::WgpuVector;

#[cfg(test)]
mod tests;
