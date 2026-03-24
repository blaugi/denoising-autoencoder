use burn::prelude::*;
use image::{ImageBuffer, Rgb};

use crate::autoencoder::{Autoencoder, ModelConfig};
use crate::data::{get_test_dataset, Cifar10Batcher};
use burn::data::dataloader::batcher::Batcher;
use burn::record::{CompactRecorder, Recorder};

fn tensor_to_image<B: Backend>(tensor: Tensor<B, 3>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let [channels, height, width] = tensor.dims();
    
    let data_vec = tensor.into_data().to_vec::<f32>().unwrap();
    
    let mut img = ImageBuffer::new(width as u32, height as u32);
    
    for y in 0..height {
        for x in 0..width {
            let get_val = |c: usize| -> u8 {
                let idx = c * (height * width) + y * width + x;
                let val = (data_vec[idx] * 255.0).clamp(0.0, 255.0) as u8;
                val
            };
            img.put_pixel(x as u32, y as u32, Rgb([get_val(0), get_val(1), get_val(2)]));
        }
    }
    img
}

pub fn save_comparisons<B: Backend>(
    images: Tensor<B, 4>, 
    noisy_images: Tensor<B, 4>, 
    outputs: Tensor<B, 4>, 
    num_samples: usize,
    output_dir: &str,
) {
    let [_batch_size, _c, h, w] = images.dims();
    let padding = 2; // Fixed spacing between images
    let grid_w = (w * 3 + padding * 2) as u32;
    let grid_h = (h * num_samples + padding * (num_samples - 1)) as u32;

    let mut grid = ImageBuffer::new(grid_w, grid_h);
    // Fill background with white
    for p in grid.pixels_mut() {
        *p = Rgb([255, 255, 255]);
    }

    for i in 0..num_samples {
        let orig_tensor = images.clone().slice([i..i+1]).squeeze::<3>();
        let noisy_tensor = noisy_images.clone().slice([i..i+1]).squeeze::<3>();
        let out_tensor = outputs.clone().slice([i..i+1]).squeeze::<3>();

        let orig_img = tensor_to_image(orig_tensor);
        let noisy_img = tensor_to_image(noisy_tensor);
        let out_img = tensor_to_image(out_tensor);

        let y_offset = (i * (h + padding)) as u32;
        let w_u32 = w as u32;
        let p_u32 = padding as u32;

        for y in 0..h as u32 {
            for x in 0..w as u32 {
                // Column 1: Original
                grid.put_pixel(x, y_offset + y, *orig_img.get_pixel(x, y));
                // Column 2: Noisy
                grid.put_pixel(w_u32 + p_u32 + x, y_offset + y, *noisy_img.get_pixel(x, y));
                // Column 3: Reconstructed
                grid.put_pixel(2 * w_u32 + 2 * p_u32 + x, y_offset + y, *out_img.get_pixel(x, y));
            }
        }
    }
    
    let path = format!("{}/evaluation_grid.png", output_dir);
    grid.save(&path).unwrap();
    println!("Evaluation grid saved to {}", path);
}

pub fn evaluate_model<B: Backend>(artifact_dir: &str, device: B::Device) {
    let config = ModelConfig::new(64, 3, 256, 32, 32); 
    
    let record = CompactRecorder::new()
        .load(format!("{}/model", artifact_dir).into(), &device)
        .expect("Trained model weights should be present");
        
    let model = config.init::<B>(&device).load_record(record);

    let dataset = get_test_dataset();
    let batcher = Cifar10Batcher::<B>::new(device.clone());
    
    // Convert first n items properly using traits if necessary, here assume simple get 
    let mut items = Vec::new();
    for i in 0..5 {
         items.push(burn_dataset::Dataset::get(&dataset, i).unwrap());
    }
    
    let batch = batcher.batch(items, &device);

    let original = batch.images.clone();
    
    // Manually apply noise during evaluation because GaussianNoise is disabled in inference mode
    let noise = Tensor::<B, 4>::random(
        original.dims(), 
        burn::tensor::Distribution::Normal(0.0, 0.05), 
        &device
    );
    let noisy = original.clone() + noise;
    let noisy = noisy.clamp(0.0, 1.0); // Clamp to valid image bounds for preview
    
    let output = model.forward(noisy.clone());

    save_comparisons(original, noisy, output, 5, artifact_dir);
}