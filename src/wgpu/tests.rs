use super::*;
use crate::context::{ksp_context::SolverType, pc_context::PcType};
use crate::matrix::sparse::CsrMatrix;
use std::sync::Arc;

struct CopyPreconditioner {
    runtime: Arc<WgpuRuntime>,
    dimension: usize,
}

impl WgpuPreconditioner for CopyPreconditioner {
    fn dims(&self) -> (usize, usize) {
        (self.dimension, self.dimension)
    }

    fn apply(&self, input: &WgpuVector, output: &WgpuVector) -> Result<(), crate::KError> {
        if !Arc::ptr_eq(&self.runtime, input.runtime())
            || !Arc::ptr_eq(&self.runtime, output.runtime())
            || input.len() != self.dimension
            || output.len() != self.dimension
        {
            return Err(crate::KError::InvalidInput(
                "copy preconditioner received an incompatible vector".into(),
            ));
        }
        let bytes = u64::try_from(self.dimension * std::mem::size_of::<f32>())
            .expect("test dimension fits u64");
        let mut encoder =
            self.runtime
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test caller-owned WebGPU preconditioner"),
                });
        encoder.copy_buffer_to_buffer(input.buffer(), 0, output.buffer(), 0, bytes);
        self.runtime.queue().submit([encoder.finish()]);
        Ok(())
    }
}

fn tridiagonal() -> CsrMatrix<f64> {
    CsrMatrix::from_csr(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
    )
}

#[test]
fn resident_bicgstab_jacobi_matches_a_known_solution() {
    pollster::block_on(async {
        let runtime = WgpuRuntime::request().await.expect("request WebGPU");
        let operator =
            Arc::new(WgpuCsrOp::from_host(runtime.clone(), &tridiagonal()).expect("upload CSR"));
        let mut context = WgpuKspContext::new(runtime);
        context.set_type(SolverType::BiCgStab).unwrap();
        context.set_pc_type(PcType::Jacobi).unwrap();
        context.set_tolerances(1.0e-5, 1.0e-7, 1.0e5, 100).unwrap();
        context.set_operator(operator).unwrap();
        let mut solution = vec![0.0; 3];
        let stats = context
            .solve_host(&[2.0, 4.0, 7.0], &mut solution)
            .await
            .expect("solve known system");
        assert!(stats.reason.is_converged(), "{stats:?}");
        for (actual, expected) in solution.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 5.0e-4);
        }
        assert!(stats.final_true_residual.unwrap() < 1.0e-4);
    });
}

#[test]
fn nonconverged_host_solve_is_atomic() {
    pollster::block_on(async {
        let runtime = WgpuRuntime::request().await.expect("request WebGPU");
        let operator =
            Arc::new(WgpuCsrOp::from_host(runtime.clone(), &tridiagonal()).expect("upload CSR"));
        let mut context = WgpuKspContext::new(runtime);
        context.set_pc_type(PcType::None).unwrap();
        context.set_tolerances(0.0, 0.0, 1.0e5, 1).unwrap();
        context.set_operator(operator).unwrap();
        let mut solution = vec![0.25, 0.5, 0.75];
        let before = solution.clone();
        let stats = context
            .solve_host(&[2.0, 4.0, 7.0], &mut solution)
            .await
            .expect("nonconverged solve returns diagnostics");
        assert!(!stats.reason.is_converged());
        assert_eq!(solution, before);
    });
}

#[test]
fn caller_owned_resident_preconditioner_participates_in_solve() {
    pollster::block_on(async {
        let runtime = WgpuRuntime::request().await.expect("request WebGPU");
        let operator =
            Arc::new(WgpuCsrOp::from_host(runtime.clone(), &tridiagonal()).expect("upload CSR"));
        let preconditioner = CopyPreconditioner {
            runtime: runtime.clone(),
            dimension: 3,
        };
        let mut context = WgpuKspContext::new(runtime);
        context.set_operator(operator).unwrap();
        context.set_preconditioner(Box::new(preconditioner));
        context.set_tolerances(1.0e-5, 1.0e-7, 1.0e5, 100).unwrap();

        let mut solution = vec![0.0; 3];
        let stats = context
            .solve_host(&[2.0, 4.0, 7.0], &mut solution)
            .await
            .expect("solve with caller-owned resident preconditioner");

        assert!(stats.reason.is_converged(), "{stats:?}");
        for (actual, expected) in solution.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 5.0e-4);
        }
    });
}
