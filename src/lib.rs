pub mod utils;

mod errors;

use errors::InvalidGrayscaleImageError;
use image::{DynamicImage, GenericImageView};
use nalgebra::DMatrix;
use rand::{distributions::Uniform, Rng};
use rand_distr::StandardNormal;

/// Log information of the current image.
pub fn __log_image_info(path: &str, image: &DynamicImage) {
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

/// Generate set of weights using Xavier initializing.
///
/// # Arguments
/// *input_size* - The size of the input layer
/// *layer_1_size* - The number of
///
fn get_xavier_weight_matrix(input_size: usize, output_size: usize) -> DMatrix<f64> {
    // rows, columns
    let dims = (output_size, input_size + 1);

    let (upper, lower) = (
        1_f64 / (input_size as f64).sqrt(),
        -1_f64 / (input_size as f64).sqrt(),
    );

    DMatrix::from_iterator(
        dims.0,
        dims.1,
        rand::thread_rng()
            .sample_iter(Uniform::new_inclusive(lower, upper))
            .take(dims.0 * dims.1),
    )
}

/// Generate set of weights using He initialization
///
/// # Arguments
/// *input_size* - The size of the input layer
///
fn get_he_weight_matrix(input_size: usize, output_size: usize) -> DMatrix<f64> {
    // rows, columns
    let dims = (output_size, input_size + 1);

    let std: f64 = (2_f64 / input_size as f64).sqrt();

    DMatrix::from_iterator(
        dims.0,
        dims.1,
        rand::thread_rng()
            .sample_iter(StandardNormal)
            .take(dims.0 * dims.1)
            .map(|n: f64| n * std),
    )
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    /// Test obtaining weight values
    fn xavier_init() {
        let (input_size, output_size) = (100, 32);
        assert_eq!(
            get_xavier_weight_matrix(input_size, output_size).len(),
            output_size * (input_size + 1)
        );
    }

    #[test]
    /// Print sample matrix from init_weight
    fn print_xavier_init() {
        let (input_size, output_size) = (5, 10);
        println!("{}", get_xavier_weight_matrix(input_size, output_size));
    }

    #[test]
    /// Verify no value in the weight matrix is outside the distribution range
    fn check_weight_in_range() {
        let (input_size, output_size) = (30, 300);
        let m = get_xavier_weight_matrix(input_size, output_size);

        let (upper, lower) = (
            1_f64 / (input_size as f64).sqrt(),
            -1_f64 / (input_size as f64).sqrt(),
        );

        for x in m.into_iter() {
            assert!(*x >= lower && *x <= upper);
        }
    }

    #[test]
    /// Test obtaining weight values
    fn he_init() {
        let (input_size, output_size) = (100, 32);
        assert_eq!(
            get_he_weight_matrix(input_size, output_size).len(),
            output_size * (input_size + 1)
        );
    }

    #[test]
    /// Print sample matrix from init_weight
    fn print_he_init() {
        let (input_size, output_size) = (5, 10);
        println!("{}", get_he_weight_matrix(input_size, output_size));
    }
}
