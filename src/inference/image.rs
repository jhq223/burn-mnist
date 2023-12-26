use image::{Luma, Rgba};

// Convert RGB to gray function
pub fn rgb2gray(rgba: Rgba<u8>) -> Luma<u8> {
    let gray = (0.299 * rgba[0] as f32 + 0.587 * rgba[1] as f32 + 0.114 * rgba[2] as f32) as u8;
    Luma([gray])
}
