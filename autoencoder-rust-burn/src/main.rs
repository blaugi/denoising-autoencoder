mod autoencoder;
mod encoder;
mod decoder;
mod data;
mod training;
mod evaluation;

use burn::{
    backend::Autodiff,
    backend::wgpu::{Wgpu, WgpuDevice},
    optim::AdamConfig,
};
use autoencoder::ModelConfig;
use training::TrainingConfig;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default(); 
    let artifact_dir = "/tmp/autoencoder";

    let optimizer_config = AdamConfig::new();
    
    let model_config = ModelConfig::new(64, 3, 256, 32, 32); 

    let config = TrainingConfig::new(model_config, optimizer_config)
        .with_num_epochs(10)
        .with_batch_size(64)
        .with_learning_rate(1.0e-3);

    training::train::<MyAutodiffBackend>(artifact_dir, config, device.clone());

    println!("Testing and saving evaluation images...");
    evaluation::evaluate_model::<MyBackend>(artifact_dir, device.clone());
}