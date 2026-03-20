use crate::encoder::{EncoderConfig, Encoder};
use crate::decoder::{DecoderConfig, Decoder};

use burn::{
    nn::{
        GaussianNoise, GaussianNoiseConfig,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
    noise_fn : GaussianNoise,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    base_channel_size: usize,
    num_input_channels: usize,
    latent_dim: usize,
    width:usize,
    height:usize,
    noise_fn: GaussianNoiseConfig,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Autoencoder<B> {
        Autoencoder {
            encoder : EncoderConfig{
                num_input_channels: self.num_input_channels,
                latent_dim: self.latent_dim,
                base_channel_size: self.base_channel_size,
            }.init(device),
            decoder : DecoderConfig{
                num_input_channels: self.num_input_channels,
                latent_dim: self.latent_dim,
                base_channel_size: self.base_channel_size,
            }.init(device),
            noise_fn: GaussianNoiseConfig{std:1.}.init(),
          
        }
    }
}

impl<B: Backend> Autoencoder<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, class_prob]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = images.dims();

        let x = images;
        
        let x = self.encoder.forward(x);
        self.decoder.forward(x)
    }
}
