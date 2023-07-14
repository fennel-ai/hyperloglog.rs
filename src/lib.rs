//! Implementations of the HyperLogLog algorithm for cardinality estimation.
//!
//! HyperLogLog is an probabilistic algorithm for estimating the number of
//! *distinct* elements (*cardinality*) of a multiset. The original algorithm,
//! described by P. Flajolet et al. in *HyperLogLog: the analysis of a
//! near-optimal cardinality estimation algorithm*, can estimate cardinalities
//! well beyond 10<sup>9</sup> with a typical accuracy of 2% while using memory
//! of 1.5 kilobytes. HyperLogLog variants can improve on those results.
//!
//! All HyperLogLog variants should implement the [`HyperLogLog`] trait.
//!
//! Current implementations:
//!
//! * [`HyperLogLogPF`]
//! * [`HyperLogLogPlus`]

#![cfg_attr(feature = "bench-units", feature(test))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
extern crate core;

mod common;
#[cfg(feature = "std")]
mod constants;
#[cfg(feature = "std")]
mod encoding;
#[cfg(feature = "std")]
mod hyperloglogplus;
#[cfg(not(feature = "std"))]
mod log;

use core::borrow::Borrow;
use core::fmt;
use core::hash::Hash;
use bytes::Bytes;

// Available only with std.
#[cfg(feature = "std")]
pub use crate::hyperloglogplus::HyperLogLogPlus;

/// A trait that should be implemented by any HyperLogLog variant.
pub trait HyperLogLog<H: Hash + ?Sized> {
    /// Inserts a new value to the multiset.
    fn insert<Q>(&mut self, value: &Q) -> Result<(), HyperLogLogError>
    where
        H: Borrow<Q>,
        Q: Hash + ?Sized;

    /// Deletes a value from the multiset.
    fn delete<Q>(&mut self, value: &Q) -> Result<(), HyperLogLogError>
    where
        H: Borrow<Q>,
        Q: Hash + ?Sized;

    /// Estimates the cardinality of the multiset.
    fn count(&mut self) -> Result<f64, HyperLogLogError>;

    /// Merges the multiset with another multiset.
    fn merge(&mut self, other: &Self) -> Result<(), HyperLogLogError>;

    /// Merges another HyperLogLogPlus into this one without using any counter data.
    fn merge_compact(&mut self, other: &HyperLogLogPlus<H>) ->  Result<(), HyperLogLogError>;

    /// Serializes the HyperLogLogPlus into a byte array.
    fn serialize(&mut self) -> Result<Bytes, HyperLogLogError>;

    /// Deserializes a byte array into a HyperLogLogPlus.
    fn deserialize(bytes: &[u8]) -> Result<Self, HyperLogLogError> where Self: Sized;

    /// Deserializes a byte array into a HyperLogLogPlus without loading any counter data.
    fn deserialize_compact(bytes: &[u8]) -> Result<Self, HyperLogLogError> where Self: Sized;

    /// Returns the estimate of the memory used by the multiset.
    fn mem_size(&mut self) -> usize;
}

#[derive(Debug, PartialEq)]
pub enum HyperLogLogError {
    InvalidPrecision,
    IncompatiblePrecision,
    InvalidSparseDelete(u32, u32),
    InvalidSparseInsert(String),
    InvalidDenseDelete(u32, u32),
    EmptyBuffer,
    InvalidCodec(u8),
    DeserializationError(String),
    SerializationError(String),
}

impl fmt::Display for HyperLogLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperLogLogError::InvalidPrecision => "precision is out of bounds".fmt(f),
            HyperLogLogError::IncompatiblePrecision => "precisions must be equal".fmt(f),
            HyperLogLogError::InvalidSparseInsert(msg) => {
                write!(f, "invalid insert in sparse format: {}", msg)
            }
            HyperLogLogError::InvalidSparseDelete(index, size) => {
                write!(
                    f,
                    "invalid index to delete in sparse format: {}, current length: {}",
                    index, size
                )
            }
            HyperLogLogError::InvalidDenseDelete(index, size) => {
                write!(
                    f,
                    "invalid index to delete in dense format: {}, current length: {}",
                    index, size
                )
            }
            HyperLogLogError::EmptyBuffer => "Trying to read from an empty buffer".fmt(f),
            HyperLogLogError::InvalidCodec(codec) => {
                write!(f, "Invalid codec version: {}", codec)
            }
            HyperLogLogError::DeserializationError(err) => {
                write!(f, "Deserialization error: {}", err)
            }
            HyperLogLogError::SerializationError(err) => {
                write!(f, "Serialization error: {}", err)
            }
        }
    }
}

impl std::error::Error for HyperLogLogError {}
