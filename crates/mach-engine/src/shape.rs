//! Tensor shapes.

use core::fmt;

/// A tensor shape: an ordered list of dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Builds a shape from dimension sizes.
    #[must_use]
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Rank: number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Number of elements (product of dims; 1 for a scalar/empty shape).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Dimension sizes as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Byte size for a given element type, checked against overflow.
    pub fn byte_size(&self, dtype: crate::DType) -> Result<usize, crate::Error> {
        self.numel()
            .checked_mul(dtype.size())
            .ok_or_else(|| crate::Error::InvalidArgument("shape byte size overflow".into()))
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn numel_and_bytes() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.byte_size(DType::F32).unwrap(), 96);
        assert_eq!(s.to_string(), "[2, 3, 4]");
    }
}
