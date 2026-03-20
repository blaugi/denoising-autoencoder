 use burn_dataset::HuggingfaceDatasetLoader;
 use burn_dataset::SqliteDataset;
 use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug, Clone)]
struct cifar10ItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: usize,
}

pub fn get_train_dataset() -> SqliteDataset<cifar10ItemRaw> {
    HuggingfaceDatasetLoader::new("cifar10")
        .dataset("train")
        .unwrap()
}