mod autoencoder;
mod encoder;
mod decoder;
mod data;
mod training;

use burn::{
    backend::Autodiff,
    optim::AdamConfig,
};
use burn_tch::{LibTorch, LibTorchDevice}; // Import the LibTorch integration
use autoencoder::ModelConfig;
use training::TrainingConfig;

fn main() {
    type MyBackend = LibTorch<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = LibTorchDevice::Cuda(0); 
    let artifact_dir = "/tmp/autoencoder";

    let optimizer_config = AdamConfig::new();
    
    let model_config = ModelConfig::new(8, 3, 128, 32, 32); 

    let config = TrainingConfig::new(model_config, optimizer_config)
        .with_num_epochs(10)
        .with_batch_size(64)
        .with_learning_rate(1.0e-4);

    training::train::<MyAutodiffBackend>(artifact_dir, config, device.clone());
}