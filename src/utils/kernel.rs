// Current nalgebra library supports vector (as in column matrix) convolutions but not 2D matrix convolutions.
// This code is an extension on the nalgebra library to allow this.

// Note: cannot define an implementation outside of an external crate for structs in said crate. Therefore, use
// traits instead to force an external type to be bound by some trait.

use nalgebra::{
    matrix, DMatrix, Dim, Dynamic, Matrix, Matrix1x3, Matrix3, Matrix3x1, Scalar, Storage,
};
use num::{One, Zero};
use std::ops::{AddAssign, Mul, Sub};

/// An operator that can be used in convolutions with true separation (not depthwise)
#[derive(Clone, Copy)]
pub enum SeparableOperator {
    Top,
    Bottom,
    Left,
    Right,
}

/// Determine padding type to be used during convolution
pub enum Padding {
    None,
    Same,
}

/// Determine the pooling type to be used in the pooling layer
pub enum Pooling {
    Average,
    Max,
}

/// Generate Sobel operator built using generics based on the provided operation variant.
fn sobel_separated<N>(op: SeparableOperator) -> (Matrix3x1<N>, Matrix1x3<N>)
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy,
{
    let n_0: N = num::zero::<N>();
    let n_1: N = num::one::<N>();
    let n_2: N = n_1 + n_1;
    let n_1_neg: N = n_0 - n_1;

    match op {
        SeparableOperator::Top => (matrix![n_1; n_0; n_1_neg], matrix![n_1, n_2, n_1]),
        SeparableOperator::Bottom => (matrix![n_1_neg; n_0; n_1], matrix![n_1, n_2, n_1]),
        SeparableOperator::Left => (matrix![n_1; n_2; n_1], matrix![n_1, n_0, n_1_neg]),
        SeparableOperator::Right => (matrix![n_1; n_2; n_1], matrix![n_1_neg, n_0, n_1]),
    }
}

// Sobel operators for testing and benching
pub const __TOP_SOBEL: Matrix3<f64> = matrix![1.0, 2.0, 1.0; 0.0, 0.0, 0.0; -1.0, -2.0, -1.0];
pub const __BOTTOM_SOBEL: Matrix3<f64> = matrix![-1.0, -2.0, -1.0; 0.0, 0.0, 0.0; 1.0, 2.0, 1.0];
pub const __LEFT_SOBEL: Matrix3<f64> = matrix![1.0, 0.0, -1.0; 2.0, 0.0, -2.0; 1.0, 0.0, -1.0];
pub const __RIGHT_SOBEL: Matrix3<f64> = matrix![-1.0, 0.0, 1.0; -2.0, 0.0, 2.0; -1.0, 0.0, 1.0];

pub trait Convolve2D<N, R1, C1, S1>
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy + PartialOrd,
    R1: Dim,
    C1: Dim,
    S1: Storage<N, R1, C1>,
{
    /// Returns the convolution of a target 2D Matrix and a kernel.
    ///
    /// # Arguments
    /// * `kernel` - A Matrix with size > 0.
    /// * `padding` - The padding style to be used. Can be either `Padding::None` or `Padding::Same`.
    ///
    /// # Errors
    /// Kernel dimensions must be less than or equal to target matrix's dimensions.
    ///
    fn convolve_2d<R2, C2, S2>(
        &self,
        kernel: &Matrix<N, R2, C2, S2>,
        padding: &Padding,
    ) -> DMatrix<N>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<N, R2, C2>;

    /// Returns the convolution of a target 2D Matrix and a provided Sobel operator.
    ///
    /// # Arguments
    /// * `op` - An valid operation from the `SeparableOperator` enum.
    /// * `padding` - The padding style to be used. Can be either `Padding::None` or `Padding::Same`.
    ///
    /// # Errors
    /// Default Sobel kernel dimensions are 3x3, meaning target matrix's dimensions must be > 3x3.
    ///
    fn convolve_2d_separated(&self, op: SeparableOperator, padding: &Padding) -> DMatrix<N>;

    /// Activate the convoluted matrix with ReLU
    fn relu(&self) -> DMatrix<N>;
}

impl<N, R1, C1, S1> Convolve2D<N, R1, C1, S1> for Matrix<N, R1, C1, S1>
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy + PartialOrd,
    R1: Dim,
    C1: Dim,
    S1: Storage<N, R1, C1>,
{
    // Credit to GitHub user guissalustiano on his nalgebra pull request (#855) on how the Storage trait is used
    fn convolve_2d<R2, C2, S2>(
        &self,
        kernel: &Matrix<N, R2, C2, S2>,
        padding: &Padding,
    ) -> DMatrix<N>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<N, R2, C2>,
    {
        let target_shape = self.shape();
        let kernel_shape = kernel.shape();

        if kernel_shape == (0, 0)
            || kernel_shape.0 > target_shape.0
            || kernel_shape.1 > target_shape.1
        {
            panic!("convolve_2d expects 'self.shape() >= kernel_shape() > 0', received {:?} and {:?} respectively.", target_shape, kernel_shape);
        }

        // Only allow same padding if both kernel dimensions are odd
        if kernel_shape.0 % 2 == 0 || kernel_shape.1 % 2 == 0 {
            if let Padding::Same = padding {
                panic!("convolve_2d expects kernel dimensions to be odd when padding mode set to 'SAME', got {:?}", kernel_shape);
            }
        }

        match padding {
            Padding::Same => {
                let conv_shape = (target_shape.0, target_shape.1);
                let padding = (kernel_shape.0 / 2, kernel_shape.1 / 2);

                let mut conv = DMatrix::zeros_generic(
                    Dynamic::from_usize(conv_shape.0),
                    Dynamic::from_usize(conv_shape.1),
                );

                // Padding is (f - 1) / 2, where f is the respective dimension of the kernel
                let mut matrix: DMatrix<N> = DMatrix::zeros_generic(
                    Dynamic::from_usize(target_shape.0 + padding.0 * 2),
                    Dynamic::from_usize(target_shape.1 + padding.1 * 2),
                );

                // Recreate self into new matrix with padded borders
                for cy in 1..(target_shape.0 + padding.0) {
                    for cx in 1..(target_shape.1 + padding.1) {
                        matrix[(cy, cx)] = self[(cy - 1, cx - 1)];
                    }
                }

                for cy in 0..conv_shape.0 {
                    for cx in 0..conv_shape.1 {
                        for ky in 0..kernel_shape.0 {
                            for kx in 0..kernel_shape.1 {
                                conv[(cy, cx)] += matrix[(cy + ky, cx + kx)] * kernel[(ky, kx)];
                            }
                        }
                    }
                }
                conv
            }
            Padding::None => {
                let conv_shape = (
                    target_shape.0 - kernel_shape.0 + 1,
                    target_shape.1 - kernel_shape.1 + 1,
                );

                let mut conv = DMatrix::zeros_generic(
                    Dynamic::from_usize(conv_shape.0),
                    Dynamic::from_usize(conv_shape.1),
                );

                for cy in 0..conv_shape.0 {
                    for cx in 0..conv_shape.1 {
                        for ky in 0..kernel_shape.0 {
                            for kx in 0..kernel_shape.1 {
                                conv[(cy, cx)] += self[(cy + ky, cx + kx)] * kernel[(ky, kx)];
                            }
                        }
                    }
                }
                conv
            }
        }
    }

    fn convolve_2d_separated(&self, op: SeparableOperator, padding: &Padding) -> DMatrix<N> {
        let matrix_shape = self.shape();

        if matrix_shape.0 < 3 || matrix_shape.1 < 3 {
            panic!("convolve_2d_separated expects 'self.shape() >= kernel_shape() > 0', received {:?} and {:?} respectively.", matrix_shape, (3, 3));
        }

        let separated_kernel: (Matrix3x1<N>, Matrix1x3<N>) = sobel_separated(op);
        self.convolve_2d(&separated_kernel.0, padding)
            .convolve_2d(&separated_kernel.1, padding)
            .relu()
    }

    fn relu(&self) -> DMatrix<N> {
        DMatrix::from_iterator(
            self.nrows(),
            self.ncols(),
            self.iter()
                .map(|f| if *f >= N::zero() { *f } else { N::zero() }),
        )
    }
}

pub trait Pool2D<N, R, C, S>
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy + PartialOrd,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    /// Returns a condensed matrix resulting from either max or average pooling.
    ///
    /// # Arguments
    /// * `padding` - The padding style to be used. Can be either `Padding::None` or `Padding::Same`.
    /// * `pooling` - The pooling style to be used. Can be either `Pooling::Average` or `Pooling::Max`.
    ///
    /// # Errors
    /// Target matrix must have dimensions greater than 2x2.
    ///
    fn pool_2d(&self, padding: &Padding, pooling: &Pooling) -> DMatrix<N>;
}

impl<N, R, C, S> Pool2D<N, R, C, S> for Matrix<N, R, C, S>
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy + PartialOrd,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    fn pool_2d(&self, padding: &Padding, pooling: &Pooling) -> DMatrix<N> {
        if self.shape().0 < 2 || self.shape().1 < 2 {
            panic!(
                "stride_2d expected a matrix with dimensions greater than (2, 2), got {:?}",
                self.shape()
            );
        }

        if let Padding::Same = padding {
            // Pad original matrix with new size if required
            let row_padding = self.shape().0 % 2;
            let col_padding = self.shape().1 % 2;

            if row_padding != 0 || col_padding != 0 {
                return __pooling_padded(self, (row_padding, col_padding), pooling);
            }
        }

        let mut res: DMatrix<N> = DMatrix::zeros_generic(
            Dynamic::from_usize(self.shape().0 / 2),
            Dynamic::from_usize(self.shape().1 / 2),
        );

        let mut pooler: [N; 4] = [num::zero(); 4];
        for ry in 0..res.shape().0 {
            for rx in 0..res.shape().1 {
                res[(ry, rx)] = match pooling {
                    Pooling::Max => {
                        for px in 0..2 {
                            for py in 0..2 {
                                pooler[py + (px * 2)] = self[(ry * 2 + px, rx * 2 + py)];
                            }
                        }
                        *pooler
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap()
                    }
                    _ => {
                        panic!("Not implemented");
                    }
                };
            }
        }

        res
    }
}

/// *Internal Function*
///
/// Perform pooling operation on the target matrix with same padding. Padding values are given in
/// (row_padding, column_padding) format.
fn __pooling_padded<N, R, C, S>(
    m: &Matrix<N, R, C, S>,
    padding: (usize, usize),
    pooling: &Pooling,
) -> DMatrix<N>
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy + PartialOrd,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    let original_size = m.shape();
    let mut padded_matrix = DMatrix::zeros_generic(
        Dynamic::from_usize(original_size.0 + padding.0),
        Dynamic::from_usize(original_size.1 + padding.1),
    );

    for pmy in 0..original_size.0 {
        for pmx in 0..original_size.1 {
            padded_matrix[(pmy, pmx)] = m[(pmy, pmx)];
        }
    }

    let mut res: DMatrix<N> = DMatrix::zeros_generic(
        Dynamic::from_usize(padded_matrix.shape().0 / 2),
        Dynamic::from_usize(padded_matrix.shape().1 / 2),
    );

    let mut pooler: [N; 4] = [num::zero(); 4];
    for ry in 0..res.shape().0 {
        for rx in 0..res.shape().1 {
            res[(ry, rx)] = match pooling {
                Pooling::Max => {
                    for px in 0..2 {
                        for py in 0..2 {
                            pooler[py + (px * 2)] = padded_matrix[(ry * 2 + px, rx * 2 + py)];
                        }
                    }
                    *pooler
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                }
                _ => {
                    panic!("Not implemented");
                }
            };
        }
    }

    res
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::get_pixel_matrix;
    use image::{io::Reader as ImageReader, ImageError};
    use nalgebra::VecStorage;

    #[test]
    /// View what a convolved image of the number 4 (grayscale) would look like.
    fn convolve_2d() -> Result<(), ImageError> {
        let img = ImageReader::open("images\\mnist_png\\train\\4\\2.png")?.decode()?;
        let matrix = get_pixel_matrix(&img).unwrap();

        let top_conv_result = matrix.convolve_2d(&__TOP_SOBEL, &Padding::Same);
        let bottom_conv_result = matrix.convolve_2d(&__BOTTOM_SOBEL, &Padding::Same);
        let left_conv_result = matrix.convolve_2d(&__LEFT_SOBEL, &Padding::Same);
        let right_conv_result = matrix.convolve_2d(&__RIGHT_SOBEL, &Padding::Same);

        save("test\\top.png", top_conv_result);
        save("test\\bottom.png", bottom_conv_result);
        save("test\\left.png", left_conv_result);
        save("test\\right.png", right_conv_result);

        Ok(())
    }

    #[test]
    /// Perform 2D convolution using separated Sobel operators instead of the full matrices.
    fn convolve_2d_separated() -> Result<(), ImageError> {
        let img = ImageReader::open("images\\mnist_png\\train\\4\\2.png")?.decode()?;
        let matrix = get_pixel_matrix(&img).unwrap();

        let top_conv_result = matrix.convolve_2d_separated(SeparableOperator::Top, &Padding::Same);
        let bottom_conv_result =
            matrix.convolve_2d_separated(SeparableOperator::Bottom, &Padding::Same);
        let left_conv_result =
            matrix.convolve_2d_separated(SeparableOperator::Left, &Padding::Same);
        let right_conv_result =
            matrix.convolve_2d_separated(SeparableOperator::Right, &Padding::Same);

        save("test\\top.png", top_conv_result);
        save("test\\bottom.png", bottom_conv_result);
        save("test\\left.png", left_conv_result);
        save("test\\right.png", right_conv_result);

        Ok(())
    }

    #[test]
    /// Verify that Separated Sobel matrices can be multiplied together to get the desired result.
    fn verify_separated_sobels() {
        let separated_sobels: [(Matrix3x1<f64>, Matrix1x3<f64>); 4] = [
            sobel_separated(SeparableOperator::Top),
            sobel_separated(SeparableOperator::Bottom),
            sobel_separated(SeparableOperator::Left),
            sobel_separated(SeparableOperator::Right),
        ];

        assert_eq!(__TOP_SOBEL, separated_sobels[0].0 * separated_sobels[0].1);
        assert_eq!(__LEFT_SOBEL, separated_sobels[2].0 * separated_sobels[2].1);
        assert_eq!(__RIGHT_SOBEL, separated_sobels[3].0 * separated_sobels[3].1);
        assert_eq!(
            __BOTTOM_SOBEL,
            separated_sobels[1].0 * separated_sobels[1].1
        );
    }

    #[test]
    /// Validate padding calculation
    fn validate_padding_calc() {
        let matrix = DMatrix::from_row_iterator(28, 28, 0..(28 * 28));

        let m_size: (usize, usize) = matrix.shape();
        let k_size: (usize, usize) = __TOP_SOBEL.shape();

        let reg_dims: (usize, usize) = (m_size.0 - k_size.0 + 1, m_size.1 - k_size.1 + 1);
        let padding: (usize, usize) = (m_size.0 - reg_dims.0, m_size.1 - reg_dims.1);

        assert!(padding.0 < k_size.0);
        assert!(padding.1 < k_size.1);
    }

    #[test]
    /// Convovle with SAME padding
    fn convolve_2d_padding_same() {
        let matrix = DMatrix::from_row_iterator(30, 30, 0..(30 * 30));
        let kernel = matrix![0, 0, 0; 0, 1, 0; 0, 0, 0]; // identity

        assert_eq!(matrix, matrix.convolve_2d(&kernel, &Padding::Same));
    }

    #[test]
    /// Test with a huge file
    fn large_convolve_2d() -> Result<(), ImageError> {
        let img = ImageReader::open("images\\dog.jpeg")?.decode()?.grayscale();
        let matrix = get_pixel_matrix(&img).unwrap();

        let top_conv_result = matrix.convolve_2d_separated(SeparableOperator::Top, &Padding::Same);
        let bottom_conv_result =
            matrix.convolve_2d_separated(SeparableOperator::Bottom, &Padding::Same);
        let left_conv_result =
            matrix.convolve_2d_separated(SeparableOperator::Left, &Padding::Same);
        let right_conv_result =
            matrix.convolve_2d_separated(SeparableOperator::Right, &Padding::Same);

        save("test\\dog_top.png", top_conv_result);
        save("test\\dog_bottom.png", bottom_conv_result);
        save("test\\dog_left.png", left_conv_result);
        save("test\\dog_right.png", right_conv_result);

        Ok(())
    }

    #[test]
    /// Test pooling layer
    fn pool_2d() -> Result<(), ImageError> {
        let img = ImageReader::open("images\\dog.jpeg")?.decode()?.grayscale();
        let matrix = get_pixel_matrix(&img).unwrap();

        let top_conv_result = matrix
            .convolve_2d_separated(SeparableOperator::Top, &Padding::Same)
            .pool_2d(&Padding::Same, &Pooling::Max);

        save("test\\dog_pooled_max.jpeg", top_conv_result);

        Ok(())
    }

    #[test]
    /// Test pooling layer
    fn pool_2d_double() -> Result<(), ImageError> {
        let img = ImageReader::open("images\\dog.jpeg")?.decode()?.grayscale();
        let matrix = get_pixel_matrix(&img).unwrap();

        let top_conv_result = matrix
            .convolve_2d_separated(SeparableOperator::Top, &Padding::Same)
            .pool_2d(&Padding::Same, &Pooling::Max)
            .pool_2d(&Padding::Same, &Pooling::Max);

        save("test\\dog_double_pooled_max.jpeg", top_conv_result);

        Ok(())
    }

    ////////////////////////
    // Utility functions //
    ///////////////////////

    /// For debugging purposes, convert a grayscale pixel value to 255 (white) if greater than 255. If less than 0,
    /// set the pixel value to 0 (black). Otherwise, leave the pixel value as is.
    fn normalize(i: f64) -> u8 {
        if i > 255_f64 {
            255 as u8
        } else if i < 0_f64 {
            0 as u8
        } else {
            i as u8
        }
    }

    /// Save an image buffer (matrix form) into a file
    fn save(name: &str, matrix: Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>) {
        // nalgebra matrix iterators are column-major, not row-major, but the ImageBuffer is expecting a row-major
        // collection. Transposing the matrix as of now is not the best for performance, but makes it easier to read
        // when saving the file for debugging.
        image::save_buffer(
            name,
            &matrix
                .transpose()
                .iter()
                .map(|i| normalize(*i))
                .collect::<Vec<u8>>(),
            matrix.shape().1 as u32,
            matrix.shape().0 as u32,
            image::ColorType::L8,
        )
        .unwrap();
    }
}
