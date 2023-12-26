use std::fs::File;
use std::io::Read;

use crate::training::model::Model;
use burn::backend::wgpu::{AutoGraphicsApi, Wgpu};
use burn::module::Module;
use burn::record::BinBytesRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
pub type Backend = Wgpu<AutoGraphicsApi, f32, i32>;

static MODEL_FILE_PATH: &str = "./model.bin";

/// Builds and loads trained parameters into the model.
pub fn build_and_load_model() -> Model<Backend> {
    let model: Model<Backend> = Model::new();
    // Open the file
    let mut file = File::open(MODEL_FILE_PATH).expect("Failed to open model file");

    // Read the file content into a Vec<u8>
    let mut model_data = Vec::new();
    file.read_to_end(&mut model_data)
        .expect("Failed to read model file");
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(model_data)
        .expect("Failed to decode state");

    model.load_record(record)
}
