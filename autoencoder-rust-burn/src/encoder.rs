use burn::{
    nn::{
        Gelu, Linear, LinearConfig,PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    activation: Gelu,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    pub base_channel_size: usize,
    pub num_input_channels: usize,
    pub latent_dim: usize,
}

impl EncoderConfig{
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let c_hid = self.base_channel_size;
        Encoder {
            conv1: Conv2dConfig::new([self.num_input_channels, c_hid], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_stride([2,2])
                .init(device),
            conv2: Conv2dConfig::new([c_hid, c_hid], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv3: Conv2dConfig::new([c_hid, 2 * c_hid], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_stride([2,2])
                .init(device),
            conv4: Conv2dConfig::new([2 * c_hid, 2 * c_hid], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv5: Conv2dConfig::new([2 * c_hid, 2 * c_hid], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_stride([2,2])
                .init(device),
            activation: Gelu::new(),
            linear: LinearConfig::new(2 * 16 * c_hid, self.latent_dim).init(device),
        }
    }
}

impl<B: Backend> Encoder<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, class_prob]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = images;

        let x = self.conv1.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv3.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv4.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv5.forward(x); 
        let x = self.activation.forward(x);

        let [batch_size, current_channels, current_height, current_width] = x.dims();
        let x = x.reshape([batch_size, current_channels * current_height * current_width]);

        self.linear.forward(x)

    }
}
