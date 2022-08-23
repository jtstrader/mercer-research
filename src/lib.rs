mod errors;

use errors::InvalidGrayscaleImageError;
use image::{DynamicImage, GenericImageView};
use nalgebra::DMatrix;

/// Log information of the current image.
pub fn log_image_info(path: &str, image: &DynamicImage) {
    println!("Image Path: {}", path);
    println!("Dimensions: {:?}", image.dimensions());
    println!("Pixel Count: {}", image.pixels().count());
    println!("Byte Count: {}", image.as_bytes().len());
}

/// Generate a matrix of pixels from a provided image.
///
/// # Arguments
/// * `image` - A dynamic image reference, expected to be in grayscale.
pub fn get_pixel_matrix(image: &DynamicImage) -> Result<DMatrix<u8>, InvalidGrayscaleImageError> {
    match image {
        DynamicImage::ImageLuma8(gray_image) => {
            // Every pixel in the pixel iterator is a tuple struct of type Luma that contains a single element array.
            // The first value can be obtained immediately using .0 and then index into the single element.
            Ok(DMatrix::from_iterator(
                gray_image.dimensions().1 as usize,
                gray_image.dimensions().0 as usize,
                gray_image.pixels().map(|p| p.0[0]),
            ))
        }
        _ => Err(InvalidGrayscaleImageError),
    }
}
