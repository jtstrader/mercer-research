pub mod rcn;
pub mod utils;

mod errors;

use errors::InvalidGrayscaleImageError;
use image::{DynamicImage, GenericImageView};
use nalgebra::DMatrix;

/// Log information of the current image.
pub fn __log_image_info(path: &str, image: &DynamicImage) {
    println!("Image Path: {path}");
    println!("Dimensions: {:?}", image.dimensions());
    println!("Pixel Count: {}", image.pixels().count());
    println!("Byte Count: {}", image.as_bytes().len());
}

/// Generate a matrix of pixels from a provided image. Every pixel in the pixel iterator is a tuple struct of type
/// Luma that contains a single element array. The first value can be obtained immediately using .0 and then index
/// into the single element.
///
/// Can accept either Luma8 or LumaA8. Alpha channels are ignored.
///
/// # Arguments
/// * `image` - A dynamic image reference, expected to be in grayscale.
///
pub fn get_pixel_matrix(image: &DynamicImage) -> Result<DMatrix<f64>, InvalidGrayscaleImageError> {
    match image {
        DynamicImage::ImageLuma8(gray_image) => Ok(DMatrix::from_row_iterator(
            gray_image.dimensions().1 as usize,
            gray_image.dimensions().0 as usize,
            gray_image.pixels().map(|p| p.0[0] as f64),
        )),
        DynamicImage::ImageLumaA8(gray_image) => Ok(DMatrix::from_row_iterator(
            gray_image.dimensions().1 as usize,
            gray_image.dimensions().0 as usize,
            gray_image.pixels().map(|p| p.0[0] as f64),
        )),
        _ => Err(InvalidGrayscaleImageError),
    }
}
