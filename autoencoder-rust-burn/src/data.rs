use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
};
use burn_dataset::{source::huggingface::HuggingfaceDatasetLoader, SqliteDataset};
use image::load_from_memory;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Cifar10ItemRaw {
    pub img: Vec<u8>,
    pub label: usize,
}

pub fn get_train_dataset() -> SqliteDataset<Cifar10ItemRaw> {
    HuggingfaceDatasetLoader::new("cifar10")
        .dataset("train")
        .unwrap()
}

pub fn get_test_dataset() -> SqliteDataset<Cifar10ItemRaw> {
    HuggingfaceDatasetLoader::new("cifar10")
        .dataset("test")
        .unwrap()
}

#[derive(Clone, Debug)]
pub struct Cifar10Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct Cifar10Batcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Cifar10Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> burn::data::dataloader::batcher::Batcher<B, Cifar10ItemRaw, Cifar10Batch<B>> for Cifar10Batcher<B> {
    fn batch(&self, items: Vec<Cifar10ItemRaw>, device: &B::Device) -> Cifar10Batch<B> {
        let batch_size = items.len();
        let mut images_vec = Vec::with_capacity(batch_size * 3 * 32 * 32);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for item in items {
            let img = load_from_memory(&item.img).expect("Failed to load image from bytes");
            let img = img.to_rgb8();
            
            for c in 0..3 {
                for y in 0..32 {
                    for x in 0..32 {
                        let pixel = img.get_pixel(x as u32, y as u32);
                        images_vec.push(pixel[c as usize] as f32 / 255.0);
                    }
                }
            }

            targets_vec.push(item.label as i32);
        }

        let images = Tensor::<B, 1>::from_floats(images_vec.as_slice(), device)
            .reshape([batch_size, 3, 32, 32]);

        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), device);

        Cifar10Batch { images, targets }
    }
}

