use burn::{
    nn::{
        Tanh, Gelu, Linear, LinearConfig,
        conv::{ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv1: ConvTranspose2d<B>,
    conv2: ConvTranspose2d<B>,
    conv3: ConvTranspose2d<B>,
    conv4: ConvTranspose2d<B>,
    conv5: ConvTranspose2d<B>,
    tanh: Tanh,
    activation: Gelu,
    linear:Linear<B>,
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    pub base_channel_size: usize,
    pub num_input_channels: usize, 
    pub latent_dim: usize,
}

impl DecoderConfig{
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let c_hid = self.base_channel_size;
        Decoder {
            linear: LinearConfig::new(self.latent_dim, 2 * 16 * c_hid).init(device),
            conv1: ConvTranspose2dConfig::new([2 * c_hid, 2 * c_hid], [3, 3])
                .with_padding([1,1])
                .with_padding_out([1,1])
                .with_stride([2,2])
                .init(device),
            conv2: ConvTranspose2dConfig::new([2 * c_hid, 2 * c_hid], [3, 3])
                .with_padding([1,1])
                .init(device),
            conv3: ConvTranspose2dConfig::new([2 * c_hid, c_hid], [3, 3])
                .with_padding([1,1])
                .with_padding_out([1,1])
                .with_stride([2,2])
                .init(device),
            conv4: ConvTranspose2dConfig::new([c_hid, c_hid], [3, 3])
                .with_padding([1,1])
                .init(device),
            conv5: ConvTranspose2dConfig::new([c_hid, self.num_input_channels], [3, 3])
                .with_padding([1,1])
                .with_padding_out([1,1])
                .with_stride([2,2])
                .init(device),
            tanh: Tanh::new(),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> Decoder<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, class_prob]
    pub fn forward(&self, images: Tensor<B, 2>) -> Tensor<B, 4> {
        let [_batch_size, _flattened] = images.dims();

        let x = images;

        let x = self.linear.forward(x);
        let x = self.activation.forward(x);
        let batch_size = x.dims()[0];
        let flattened = x.dims()[1];
        let x = x.reshape([batch_size, flattened / 16, 4, 4]);

        let x = self.conv1.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv3.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv4.forward(x); 
        let x = self.activation.forward(x);

        let x = self.conv5.forward(x); 
        self.tanh.forward(x)


    }
}
