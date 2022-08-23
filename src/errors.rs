#[derive(Debug, Clone)]
pub struct InvalidGrayscaleImageError;

impl std::fmt::Display for InvalidGrayscaleImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InvalidGrayscaleImageError: Image provided was not Luma8 (grayscaled image)"
        )
    }
}
