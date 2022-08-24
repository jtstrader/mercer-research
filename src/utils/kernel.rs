// Current nalgebra library supports vector (as in column matrix) convolutions but not 2D matrix convolutions.
// This code is an extension on the nalgebra library to allow this.

// Note: cannot define an implementation outside of an external crate for structs in said crate. Therefore, use
// traits instead to force an external type to be bound by some trait.

use nalgebra::{
    matrix, DMatrix, Dim, Dynamic, Matrix, Matrix1x3, Matrix3, Matrix3x1, Scalar, Storage,
};
use num::{One, Zero};
use std::ops::{AddAssign, Mul, Sub};

/// An operator that can be used in convolutions with true Separation (not depthwise)
pub enum SeparableOperator {
    Top,
    Bottom,
    Left,
    Right,
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
pub const __TOP_SOBEL: Matrix3<i16> = matrix![1, 2, 1; 0, 0, 0; -1, -2, -1];
pub const __BOTTOM_SOBEL: Matrix3<i16> = matrix![-1, -2, -1; 0, 0, 0; 1, 2, 1];
pub const __LEFT_SOBEL: Matrix3<i16> = matrix![1, 0, -1; 2, 0, -2; 1, 0, -1];
pub const __RIGHT_SOBEL: Matrix3<i16> = matrix![-1, 0, 1; -2, 0, 2; -1, 0, 1];

pub trait Convolve2D<N, R1, C1, S1>
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy,
    R1: Dim,
    C1: Dim,
    S1: Storage<N, R1, C1>,
{
    /// Returns the convolution of a target 2D Matrix and a kernel.
    ///
    /// # Arguments
    /// * `kernel` - A Matrix with size > 0
    ///
    /// # Errors
    /// Kernel dimensions must be less than or equal to target matrix's dimensions.
    ///
    fn convolve_2d<R2, C2, S2>(&self, kernel: &Matrix<N, R2, C2, S2>) -> DMatrix<N>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<N, R2, C2>;

    /// Returns the convolution of a target 2D Matrix and a provided Sobel operator.
    ///
    /// # Arguments
    /// * `op` - An valid operation from the `SeparableOperator` enum.
    ///
    /// # Errors
    /// Default Sobel kernel dimensions are 3x3, meaning target matrix's dimensions must be > 3x3.
    fn convolve_2d_separated(&self, op: SeparableOperator) -> DMatrix<N>;
}

impl<N, R1, C1, S1> Convolve2D<N, R1, C1, S1> for Matrix<N, R1, C1, S1>
where
    N: Scalar + Zero + One + AddAssign + Sub<Output = N> + Mul<Output = N> + Copy,
    R1: Dim,
    C1: Dim,
    S1: Storage<N, R1, C1>,
{
    // Credit to GitHub user guissalustiano on his nalgebra pull request (#855) on how the Storage trait is used
    fn convolve_2d<R2, C2, S2>(&self, kernel: &Matrix<N, R2, C2, S2>) -> DMatrix<N>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<N, R2, C2>,
    {
        let matrix_shape = self.shape();
        let kernel_shape = kernel.shape();

        if kernel_shape == (0, 0)
            || kernel_shape.0 > matrix_shape.0
            || kernel_shape.1 > matrix_shape.1
        {
            panic!("convolve_2d expects 'self.shape() >= kernel_shape() > 0', received {:?} and {:?} respectively.", matrix_shape, kernel_shape);
        }

        let conv_shape = (
            matrix_shape.0 - kernel_shape.0 + 1,
            matrix_shape.1 - kernel_shape.1 + 1,
        );

        let mut conv = DMatrix::zeros_generic(
            Dynamic::from_usize(conv_shape.0),
            Dynamic::from_usize(conv_shape.1),
        );

        for c_x in 0..conv_shape.0 {
            for c_y in 0..conv_shape.1 {
                for k_x in 0..kernel_shape.0 {
                    for k_y in 0..kernel_shape.1 {
                        conv[(c_x, c_y)] += self[(c_x + k_x, c_y + k_y)] * kernel[(k_x, k_y)];
                    }
                }
            }
        }

        conv
    }

    fn convolve_2d_separated(&self, op: SeparableOperator) -> DMatrix<N> {
        let matrix_shape = self.shape();

        if matrix_shape.0 < 3 || matrix_shape.1 < 3 {
            panic!("convolve_2d_separated expects 'self.shape() >= kernel_shape() > 0', received {:?} and {:?} respectively.", matrix_shape, (3, 3));
        }

        let separated_kernel: (Matrix3x1<N>, Matrix1x3<N>) = sobel_separated(op);
        self.convolve_2d(&separated_kernel.0)
            .convolve_2d(&separated_kernel.1)
    }
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

        let top_conv_result = matrix.convolve_2d(&__TOP_SOBEL);
        let bottom_conv_result = matrix.convolve_2d(&__BOTTOM_SOBEL);
        let left_conv_result = matrix.convolve_2d(&__LEFT_SOBEL);
        let right_conv_result = matrix.convolve_2d(&__RIGHT_SOBEL);

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

        let top_conv_result = matrix.convolve_2d_separated(SeparableOperator::Top);
        let bottom_conv_result = matrix.convolve_2d_separated(SeparableOperator::Bottom);
        let left_conv_result = matrix.convolve_2d_separated(SeparableOperator::Left);
        let right_conv_result = matrix.convolve_2d_separated(SeparableOperator::Right);

        save("test\\top.png", top_conv_result);
        save("test\\bottom.png", bottom_conv_result);
        save("test\\left.png", left_conv_result);
        save("test\\right.png", right_conv_result);

        Ok(())
    }

    #[test]
    /// Verify that Separated Sobel matrices can be multiplied together to get the desired result.
    fn verify_separated_sobels() {
        let separated_sobels: [(Matrix3x1<i16>, Matrix1x3<i16>); 4] = [
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

    ////////////////////////
    // Utility functions //
    ///////////////////////

    /// For debugging purposes, convert a grayscale pixel value to 255 (white) if greater than 255. If less than 0,
    /// set the pixel value to 0 (black). Otherwise, leave the pixel value as is.
    fn normalize(i: i16) -> u8 {
        if i > 255 {
            255 as u8
        } else if i < 0 {
            0 as u8
        } else {
            i as u8
        }
    }

    /// Save an image buffer (matrix form) into a file
    fn save(name: &str, matrix: Matrix<i16, Dynamic, Dynamic, VecStorage<i16, Dynamic, Dynamic>>) {
        // nalgebra matrix iterators are column-major, not row-major, but the ImageBuffer is expecting a row-major
        // collection. Transposing the matrix as of now is not the best for performance, but makes it easier to read
        // when saving the file for debugging.
        let buf: Vec<u8> = matrix.transpose().iter().map(|i| normalize(*i)).collect();
        image::save_buffer(
            name,
            &buf,
            matrix.shape().0 as u32,
            matrix.shape().1 as u32,
            image::ColorType::L8,
        )
        .unwrap();
    }
}
