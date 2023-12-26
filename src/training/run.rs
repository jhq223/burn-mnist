mod wgpu {
    use crate::training::training;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device);
    }
}

pub fn run() {
    wgpu::run();
}
