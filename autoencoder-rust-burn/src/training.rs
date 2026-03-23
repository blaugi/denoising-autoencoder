use crate::{
    autoencoder::{Autoencoder, ModelConfig},
    data::{Cifar10Batch, Cifar10Batcher},
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::{MseLoss, Reduction},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep, RegressionOutput,
        metric::{LossMetric},
    },
};

impl<B: Backend> Autoencoder<B> {
    pub fn forward_reconstruction(
        &self,
        images: Tensor<B, 4>,
    ) -> RegressionOutput<B> {
        let targets = images.clone();
        let output = self.forward(images);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Auto);

        let output_flat = output.clone().flatten::<2>(1, 3);
        let targets_flat = targets.clone().flatten::<2>(1, 3);
        RegressionOutput::new(loss, output_flat, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep for Autoencoder<B> {
    type Input = Cifar10Batch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Cifar10Batch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_reconstruction(batch.images);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Autoencoder<B> {
    type Input = Cifar10Batch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Cifar10Batch<B>) -> RegressionOutput<B> {
        self.forward_reconstruction(batch.images)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher_train = Cifar10Batcher::<B>::new(device.clone());
    let batcher_valid = Cifar10Batcher::<B::InnerBackend>::new(device.clone().into());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(crate::data::get_train_dataset());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(crate::data::get_test_dataset());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((LossMetric::new(),))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

