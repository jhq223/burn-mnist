use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,       // 图像张量
    pub targets: Tensor<B, 1, Int>, // 目标标签张量
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image)) // 从MNISTItem中提取图像
            .map(|data| Tensor::<B, 2>::from_data(data.convert())) // 转换为Tensor
            .map(|tensor| tensor.reshape([1, 28, 28])) // 改变形状为[1, 28, 28]
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081) // 归一化
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()]))) // 从MNISTItem中提取标签
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device); // 拼接图像张量
        let targets = Tensor::cat(targets, 0).to_device(&self.device); // 拼接目标张量

        MNISTBatch { images, targets }
    }
}
