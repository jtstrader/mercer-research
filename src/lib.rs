pub mod utils;

mod errors;

use errors::InvalidGrayscaleImageError;
use image::{io::Reader as ImageReader, ImageError};
use image::{DynamicImage, GenericImageView};
use nalgebra::DMatrix;
use rand::{distributions::Uniform, Rng};
use rand_distr::StandardNormal;
use std::ffi::OsString;
use std::{fs, path::PathBuf};
use utils::kernel::{Padding, Pooling, SeparableOperator};

use crate::utils::kernel::{Convolve2D, Pool2D};

/// Rust Convolutional Neural Network (RCN)
pub struct RCN<'a> {
    classes: usize,
    layer_cfg: &'a [RCNLayer<'a>],
    layer_weights: Vec<DMatrix<f64>>,

    training_path: &'a str,
    testing_path: &'a str,
}

/// Characteristics of a layer
pub enum RCNLayer<'a> {
    Convolve2D(Padding),
    Pool2D(Pooling),
    Feedforward(&'a [usize]),
}

/// List of operators that can be iterated through for convolutions
const SEP_OPS: [SeparableOperator; 4] = [
    SeparableOperator::Top,
    SeparableOperator::Left,
    SeparableOperator::Right,
    SeparableOperator::Bottom,
];

impl<'a> RCN<'a> {
    /// Create new RCN instance
    ///
    /// # Arguments
    /// * `classes` - The number of final classifications to be made
    /// * `layer_cfg` - A collection of layer configurations
    ///
    fn new(
        classes: usize,
        layer_cfg: &'a [RCNLayer],
        training_path: &'a str,
        testing_path: &'a str,
    ) -> Self {
        RCN {
            classes,
            layer_cfg,
            layer_weights: Vec::new(),
            training_path,
            testing_path,
        }
    }

    /// Train the model
    fn train(
        &mut self,
        batch_size: usize,
        epochs: usize,
        class_size_limit: usize,
    ) -> Result<(), ImageError> {
        let training_set = load_data(self.training_path, class_size_limit);

        println!("Training Set Size: {}", training_set.len());

        // FIXME: verify all classes are touched, remove later
        let mut count = 0;

        // Highest level loop is the epoch loop, since all inner code will be working through the entire dataset
        for _ in 0..epochs {
            // go through a single image each time from each class
            // iterator to get data should be able to shift class_size_limit in training_set
            for shift in 0..training_set.len() / self.classes {
                for i in 0..self.classes {
                    count += 1;
                    // each iteration represents one sample training
                    let m = &training_set[shift * self.classes + i].0;
                    let mut feature_set: Vec<DMatrix<f64>> = Vec::new();

                    for layer in self.layer_cfg {
                        match layer {
                            RCNLayer::Convolve2D(p) => {
                                // build feature set from current matrix if feature_set is not already populated
                                if feature_set.len() != 0 {
                                    // preserve current length and iterate over those only
                                    let curr_len = feature_set.len();
                                    for i in 0..curr_len {
                                        let mut it = SEP_OPS.iter().peekable();

                                        // if feature set is populated, change current matrix and extend any other matrices
                                        while let Some(op) = it.next() {
                                            if it.peek().is_none() {
                                                feature_set[i] =
                                                    feature_set[i].convolve_2d_separated(*op, p)
                                            } else {
                                                feature_set.push(
                                                    feature_set[i].convolve_2d_separated(*op, p),
                                                );
                                            }
                                        }
                                    }
                                } else {
                                    feature_set
                                        .extend(SEP_OPS.map(|op| m.convolve_2d_separated(op, p)));
                                }
                            }
                            RCNLayer::Pool2D(p) => {
                                for feature in &mut feature_set {
                                    *feature = feature.pool_2d(&Padding::Same, p);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// *Internal Function*
///
/// Take a path for data (training or testing/validating) and return a vector of the matrices
/// and their respective class names.
///
/// # Arguments
/// * `path` - The path to the data set
/// * `class_size_limit` - The absolute limit of data that should be populated per class
///
fn load_data(path: &str, class_size_limit: usize) -> Vec<(DMatrix<f64>, OsString)> {
    let classes: Vec<PathBuf> = fs::read_dir(path)
        .unwrap()
        .map(|f| f.unwrap().path())
        .collect();

    let mut dset: Vec<(DMatrix<f64>, OsString)> = Vec::new();
    for class in classes {
        // class is the last directory of the path
        let class_id = OsString::from(class.file_name().unwrap());

        let mut paths: Vec<PathBuf> = fs::read_dir(&class)
            .unwrap()
            .map(|f| f.unwrap().path())
            .collect();

        for _ in 0..class_size_limit {
            let idx = rand::thread_rng().gen_range(0..paths.len());
            let img = ImageReader::open(&paths.remove(idx))
                .unwrap()
                .decode()
                .unwrap()
                .grayscale();
            dset.push((get_pixel_matrix(&img).unwrap(), class_id.clone()));
        }
    }

    dset
}

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
pub fn get_pixel_matrix(image: &DynamicImage) -> Result<DMatrix<f64>, InvalidGrayscaleImageError> {
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

/// Generate set of weights using Xavier initializing.
///
/// # Arguments
/// *input_size* - The size of the input layer
/// *output_size* - The size of the output layer
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

    #[test]
    /// Creating and configuring RCN
    fn rcn_init() {
        let mut model = RCN::new(
            10,
            &[
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Feedforward(&[10, 20, 20]),
            ],
            "images\\mnist_png\\train",
            "images\\mnist_png\\valid",
        );
    }

    #[test]
    /// Testing Convolve2D layers
    fn convolve_2d_test() {
        let mut model = RCN::new(
            10,
            &[
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
            ],
            "images\\mnist_png\\train",
            "images\\mnist_png\\valid",
        );

        model.train(1, 1, 1).unwrap();
    }
}
