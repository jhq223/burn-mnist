use crate::inference::state::{build_and_load_model, Backend};
use crate::training::model::Model;
use burn::tensor::Tensor;

pub struct Mnist {
    model: Option<Model<Backend>>,
}

impl Mnist {
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Returns the inference results.
    ///
    /// # Arguments
    ///
    /// * `input` - A f32 slice of input 28x28 image
    ///
    pub fn inference(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if self.model.is_none() {
            self.model = Some(build_and_load_model());
        }

        let model = self.model.as_ref().unwrap();

        // Reshape from the 1D array to 3d tensor [batch, height, width]
        let input: Tensor<Backend, 3> = Tensor::from_floats(input).reshape([1, 28, 28]);

        // Normalize input: make between [0,1] and make the mean=0 and std=1
        // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
        // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122

        let input = ((input / 255) - 0.1307) / 0.3081;

        // Run the tensor input through the model
        let output: Tensor<Backend, 2> = model.forward(input);

        // Convert the model output into probability distribution using softmax formula
        let output = burn::tensor::activation::softmax(output, 1);

        // Flatten output tensor with [1, 10] shape into boxed slice of [f32]
        let output = output.into_data().convert::<f32>().value;

        Ok(output)
    }
}
