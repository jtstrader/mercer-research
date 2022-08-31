pub mod utils;

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
///
pub fn get_pixel_matrix(image: &DynamicImage) -> Result<DMatrix<i16>, InvalidGrayscaleImageError> {
    // Every pixel in the pixel iterator is a tuple struct of type Luma that contains a single element array.
    // The first value can be obtained immediately using .0 and then index into the single element.
    //
    // Data is converted into i16 form so it can later be convolved against negative values. Note that converting
    // between u8 and i8 is not safe and may result in overflow. Therefore, i16 is used to guarantee that no data
    // is lost when convolving.
    //
    // Can accept either Luma8 or LumaA8. Alpha channels are ignored.
    match image {
        DynamicImage::ImageLuma8(gray_image) => Ok(DMatrix::from_row_iterator(
            gray_image.dimensions().1 as usize,
            gray_image.dimensions().0 as usize,
            gray_image.pixels().map(|p| p.0[0] as i16),
        )),
        DynamicImage::ImageLumaA8(gray_image) => Ok(DMatrix::from_row_iterator(
            gray_image.dimensions().1 as usize,
            gray_image.dimensions().0 as usize,
            gray_image.pixels().map(|p| p.0[0] as i16),
        )),
        _ => Err(InvalidGrayscaleImageError),
    }
}
