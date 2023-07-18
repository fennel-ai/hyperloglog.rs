use bytes::Bytes;
use core::fmt::Debug;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::zip;
use std::marker::PhantomData;

use crate::common::*;
use crate::constants;
use crate::encoding::{DifIntVec, ZeroCountMap, VarIntVec};
use crate::HyperLogLog;
use crate::HyperLogLogError;



mod same_module {
    include!("serde.rs");
}

/// Implements the HyperLogLog++ algorithm for cardinality estimation.
///
/// This implementation is based on the paper:
///
/// *HyperLogLog in Practice: Algorithmic Engineering of a State of The Art
/// Cardinality Estimation Algorithm.*
///
/// - Uses 6-bit registers, packed in a 32-bit unsigned integer. Thus, every
///   five registers 2 bits are not used.
/// - In small cardinalities, a sparse representation is used which allows
///   for higher precision in estimations.
/// - Performs bias correction using the empirical data provided by Google
///   (can be found [here](http://goo.gl/iU8Ig)).
/// - Supports serialization/deserialization implemented in serde.rs
///
/// # Examples
///
/// ```
/// use std::collections::hash_map::RandomState;
/// use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
///
/// let mut hllp: HyperLogLogPlus<u32, _> = HyperLogLogPlus::new(16, RandomState::new()).unwrap();
///
/// hllp.insert(&12345);
/// hllp.insert(&23456);
///
/// assert_eq!(hllp.count().unwrap().trunc() as u32, 2);
/// ```
///
/// # References
///
/// - ["HyperLogLog: the analysis of a near-optimal cardinality estimation
///   algorithm", Philippe Flajolet, Éric Fusy, Olivier Gandouet and Frédéric
///   Meunier.](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
/// - ["HyperLogLog in Practice: Algorithmic Engineering of a State of The Art
///   Cardinality Estimation Algorithm", Stefan Heule, Marc Nunkesser and
///   Alexander Hall.](https://research.google/pubs/pub40671/)
/// - ["Appendix to HyperLogLog in Practice: Algorithmic Engineering of a State
///   of the Art Cardinality Estimation Algorithm", Stefan Heule, Marc
///   Nunkesser and Alexander Hall.](https://goo.gl/iU8Ig)
///
#[derive(Clone, Debug)]
pub struct HyperLogLogPlus<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    precision: u8,
    builder: B,
    counts: (usize, usize, usize),

    // Data structures for sparse representation.

    insert_tmpset: HashMap<u32, u32>,
    del_tmpset: HashMap<u32, u32>,
    sparse: DifIntVec,
    sparse_counters: VarIntVec,

    // Data structures for dense representation.

    registers: Option<RegistersPlus>,
    register_counters: HashMap<u16, ZeroCountMap>,

    phantom: PhantomData<H>,
}

impl<H, B> HyperLogLogCommon for HyperLogLogPlus<H, B>
    where H: Hash + ?Sized,
          B: BuildHasher, {}

impl<H, B> HyperLogLog<H, B> for HyperLogLogPlus<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    /// Inserts a new value to the multiset.
    fn insert<Q>(&mut self, value: &Q) -> Result<(), HyperLogLogError>
    where
        H: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        self.insert_impl(value)
    }

    /// Deletes a value from the multiset.
    fn delete<Q>(&mut self, value: &Q) -> Result<(), HyperLogLogError>
    where
        H: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        self.delete_impl(value)
    }

    /// Estimates the cardinality of the multiset.
    fn count(&mut self) -> Result<f64, HyperLogLogError> {
        self.count_impl()
    }

    /// Merges another HyperLogLogPlus into this one.
    fn merge(&mut self, other: &HyperLogLogPlus<H, B>) -> Result<(), HyperLogLogError> {
        self.merge_impl(other)
    }

    /// Merges another HyperLogLogPlus into this one without using any counter data.
    fn merge_compact(&mut self, other: &HyperLogLogPlus<H, B>) -> Result<(), HyperLogLogError> {
        self.compact_merge_impl(other)
    }

    /// Serializes the HyperLogLogPlus into a byte array.
    fn serialize(&mut self) -> Result<Bytes, HyperLogLogError> {
        self.to_bytes()
    }

    /// Deserializes a byte array into a HyperLogLogPlus.
    fn deserialize(bytes: &[u8], builder: B) -> Result<Self, HyperLogLogError>
    where
        Self: Sized,
    {
        HyperLogLogPlus::from_bytes(bytes, builder)
    }

    /// Deserializes a byte array into a HyperLogLogPlus without loading any counter data.
    fn deserialize_compact(bytes: &[u8], builder: B) -> Result<Self, HyperLogLogError>
    where
        Self: Sized,
    {
        HyperLogLogPlus::from_bytes_compact(bytes, builder)
    }

    /// Returns the estimate of the memory used by the multiset.
    fn mem_size(&mut self) -> usize {
        self.mem_size_impl()
    }
}

impl<H, B> HyperLogLogPlus<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    // Minimum precision allowed.
    const MIN_PRECISION: u8 = 4;
    // Maximum precision allowed.
    const MAX_PRECISION: u8 = 16;
    // Maximum precision in sparse representation.
    const PRIME_PRECISION: u8 = 25;

    /// Creates a new HyperLogLogPlus instance.
    pub fn new(precision: u8, builder: B) -> Result<Self, HyperLogLogError> {
        // Ensure the specified precision is within bounds.
        if precision < Self::MIN_PRECISION || precision > Self::MAX_PRECISION {
            return Err(HyperLogLogError::InvalidPrecision);
        }

        let count = Self::register_count(precision);
        let counts = (
            count,
            Self::register_count(Self::PRIME_PRECISION - 1),
            RegistersPlus::size_in_bytes(count),
        );

        Ok(HyperLogLogPlus {
            precision,
            builder,
            counts,
            insert_tmpset: HashMap::new(),
            del_tmpset: HashMap::new(),
            sparse: DifIntVec::new(),
            registers: None,
            register_counters: HashMap::new(),
            sparse_counters: VarIntVec::new(),
            phantom: PhantomData,
        })
    }

    /// Size of the HyperLogLogPlus instance in bytes.
    fn mem_size_impl(&mut self) -> usize {
        self.merge_sparse().unwrap();
        let self_size = std::mem::size_of_val(self);
        let insert_tmpset_size = self.insert_tmpset.len() * (std::mem::size_of::<u32>() * 2);
        let del_tmpset_size = self.del_tmpset.len() * (std::mem::size_of::<u32>() * 2);
        let sparse_size = self.sparse.mem_size();
        let sparse_counters_size = self.sparse_counters.mem_size();

        let registers_size = match &self.registers {
            Some(registers) => registers.mem_size(),
            None => 0,
        };

        let mut register_counters_size = self.register_counters.len() * std::mem::size_of::<u16>();
        for (_, counter) in &self.register_counters {
            register_counters_size += counter.mem_size();
        }
        register_counters_size += std::mem::size_of_val(&self.register_counters);

        self_size
            + insert_tmpset_size
            + del_tmpset_size
            + sparse_size
            + sparse_counters_size
            + registers_size
            + register_counters_size
    }

    #[inline] // Returns true if the HyperLogLog is using the
    // sparse representation.
    pub fn is_sparse(&self) -> bool {
        self.registers.is_none()
    }


    fn count_impl(&mut self) -> Result<f64, HyperLogLogError> {
        // Merge tmpset into sparse representation.
        if self.is_sparse() {
            self.merge_sparse()?;
        }

        match self.registers.as_mut() {
            Some(registers) => {
                // We use normal representation.

                let zeros = registers.zeros();

                if zeros != 0 {
                    let correction = Self::linear_count(self.counts.0, zeros);

                    // Use linear counting only if value below threshold.
                    if correction <= Self::threshold(self.precision) {
                        Ok(correction)
                    } else {
                        // Calculate the raw estimate.
                        let mut raw = Self::estimate_raw_plus(registers.iter(), self.counts.0);

                        // Apply correction if required.
                        if raw <= 5.0 * self.counts.0 as f64 {
                            raw -= self.estimate_bias(raw);
                        }

                        Ok(raw)
                    }
                } else {
                    // Calculate the raw estimate.
                    let mut raw = Self::estimate_raw_plus(registers.iter(), self.counts.0);

                    // Apply correction if required.
                    if raw <= 5.0 * self.counts.0 as f64 {
                        raw -= self.estimate_bias(raw);
                    }

                    Ok(raw)
                }
            }
            None => {
                // We use sparse representation.
                // Calculate number of registers set to zero.
                let zeros = self.counts.1 - self.sparse.count();
                // Use linear counting to approximate.
                Ok(Self::linear_count(self.counts.1, zeros))
            }
        }
    }

    /// Merges the `other` HyperLogLogPlus instance into `self`.
    ///
    /// Both sketches must have the same precision. Merge can trigger
    /// the transition from sparse to normal representation.
    ///
    fn merge_impl<S, T>(&mut self, other: &HyperLogLogPlus<S, T>) -> Result<(), HyperLogLogError>
    where
        S: Hash + ?Sized,
        T: BuildHasher,
    {
        if self.precision != other.precision() {
            return Err(HyperLogLogError::IncompatiblePrecision);
        }

        // If any of the sketches is in dense representation, we need to
        // merge them into a dense representation.

        if other.is_sparse() {
            if self.is_sparse() {
                // Self -> Sparse, Other -> Sparse => Sparse
                // Both sketches are in sparse representation.
                //
                // Insert all the hash codes of other into `tmpset`.
                for (hash_code, cnt) in other.insert_tmpset.iter() {
                    // Update the counter in tmpset.
                    let tmpset_cnt = self.insert_tmpset.entry(*hash_code).or_insert(0);
                    *tmpset_cnt += cnt;
                }
                for (hash_code, cnt) in
                    zip(other.sparse.into_iter(), other.sparse_counters.into_iter())
                {
                    let tmpset_cnt = self.insert_tmpset.entry(hash_code).or_insert(0);
                    *tmpset_cnt += cnt;
                }

                // Merge del_tmpset
                for (hash_code, cnt) in other.del_tmpset.iter() {
                    // Update the counter in tmpset.
                    let tmpset_cnt = self.del_tmpset.entry(*hash_code).or_insert(0);
                    *tmpset_cnt += cnt;
                }

                // Merge temporary del set into sparse representation.
                if self.del_tmpset.len() * 100 > self.counts.2 {
                    self.merge_deletes_sparse()?;
                }

                // Merge temporary set into sparse representation.
                if self.insert_tmpset.len() * 100 > self.counts.2 {
                    self.merge_inserts_sparse()?;
                }

                self.convert_to_dense_if_required();
            } else {
                // Self -> Dense, Other -> Sparse => Dense

                // The other sketch is in sparse representation but not self.
                //
                // Decode all the hash codes and update the self's
                // corresponding Registers.

                // Update the counts from the sparse representation into the dense representation.
                let registers = self.registers.as_mut().unwrap();

                // Handle the temporary insert and delete sets
                for (hash_code, cnt) in other.insert_tmpset.iter() {
                    let (zeros, index) = other.decode_hash(*hash_code);
                    let counter_map = self
                        .register_counters
                        .entry(index as u16)
                        .or_insert(ZeroCountMap::new());
                    counter_map.increase_count_at_index(zeros as u8, *cnt);
                    registers.set_greater(index, zeros);
                }

                // Empty the delete set
                for (hash_code, cnt) in other.del_tmpset.iter() {
                    let (zeros, index) = other.decode_hash(*hash_code);
                    if !self.register_counters.contains_key(&(index as u16)) {
                        continue;
                    }
                    let counter_map = self.register_counters.get_mut(&(index as u16)).unwrap();
                    if counter_map.decrease_count_at_index(zeros as u8, *cnt)? {
                        let new_max_zeros = counter_map.arg_max();
                        registers.set_register(index, new_max_zeros);
                        if new_max_zeros == 0 {
                            self.register_counters.remove(&(index as u16));
                        }
                    }
                }

                // Handle the main sparse representation
                for hash_code in other.sparse.into_iter() {
                    let (zeros, index) = other.decode_hash(hash_code);
                    registers.set_greater(index, zeros);
                }
            }
        } else {
            //  Other -> Dense => Dense

            // Convert self from sparse to normal representation.
            if self.is_sparse() {
                // The other sketch is in normal representation but self
                // is in sparse representation.
                //
                // Turn sparse into normal.
                self.merge_sparse()?;

                if self.is_sparse() {
                    self.sparse_to_normal();
                }
            }

            // Merge registers from both sketches.
            let registers = self.registers.as_mut().unwrap();
            let other_registers_iter = other.registers_iter().unwrap();
            for (i, val) in other_registers_iter.enumerate() {
                registers.set_greater(i, val);
            }
        }

        Ok(())
    }

    /// Compact merge is similar to merge but does not use counters.
    fn compact_merge_impl<S, T>(&mut self, other: &HyperLogLogPlus<S, T>) -> Result<(), HyperLogLogError>
    where
        S: Hash + ?Sized,
        T: BuildHasher,
    {
        if self.precision != other.precision() {
            return Err(HyperLogLogError::IncompatiblePrecision);
        }
        if other.is_sparse() {
            if self.is_sparse() {
                // Self -> Sparse, Other -> Sparse
                // Insert all the hash codes of other into `tmpset`.
                for hash_code in other.sparse.into_iter() {
                    // Update the counter in tmpset by 1
                    self.insert_tmpset.entry(hash_code).or_insert(1);
                }
                // Merge temporary set into sparse representation.
                if self.insert_tmpset.len() * 100 > self.counts.2 {
                    self.merge_inserts_sparse()?;
                }
            } else {
                // Self -> Dense, Other -> Sparse
                let registers = self.registers.as_mut().unwrap();
                assert!(self.del_tmpset.is_empty());

                for hash_code in other.sparse.into_iter() {
                    let (zeros, index) = other.decode_hash(hash_code);
                    registers.set_greater(index, zeros);
                }
            }
        } else {
            //  Other -> Dense

            // Convert self from sparse to normal representation.
            if self.is_sparse() {
                // The other sketch is in normal representation but self
                // is in sparse representation.
                //
                // Turn sparse into normal.
                self.merge_sparse()?;
                if self.is_sparse() {
                    self.sparse_to_normal();
                }
            }

            // Merge registers from both sketches.
            let registers = self.registers.as_mut().unwrap();
            let other_registers_iter = other.registers_iter().unwrap();
            for (i, val) in other_registers_iter.enumerate() {
                registers.set_greater(i, val);
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn insert_impl<R>(&mut self, value: &R) -> Result<(), HyperLogLogError>
    where
        R: Hash + ?Sized,
    {
        // Create a new hasher.
        let mut hasher = self.builder.build_hasher();
        value.hash(&mut hasher);
        // Use a 64-bit hash value.
        let mut hash: u64 = hasher.finish();

        match &mut self.registers {
            Some(registers) => {
                // We use normal representation.

                // Calculate the register's index.
                let index: usize = (hash >> (64 - self.precision)) as usize;

                // Shift left the bits of the index.
                hash = (hash << self.precision) | (1 << (self.precision - 1));

                // Count leading zeros.
                let zeros: u32 = 1 + hash.leading_zeros();

                // Update the register with the max leading zeros counts.
                registers.set_greater(index, zeros);

                // Insert into register_counters

                // Check if index exists in register_counters else create a new ZeroCountMap
                let counter_map = self
                    .register_counters
                    .entry(index as u16)
                    .or_insert(ZeroCountMap::new());

                counter_map.increase_count_at_index(zeros as u8, 1);
            }
            None => {
                // We use sparse representation.

                // Encode hash value.
                let hash_code = self.encode_hash(hash);

                // Increment the counter for the hash code in the sparse by 1.
                self.insert_tmpset
                    .entry(hash_code)
                    .and_modify(|e| *e += 1)
                    .or_insert(1);

                // Merge temporary set into sparse representation.
                if self.insert_tmpset.len() * 100 > self.counts.2 {
                    self.merge_inserts_sparse()?;
                    self.convert_to_dense_if_required();
                }
            }
        }
        Ok(())
    }

    pub fn delete_any<R>(&mut self, value: &R) -> Result<(), HyperLogLogError>
    where
        R: Hash + ?Sized,
    {
        self.delete_impl(value)
    }

    #[inline(always)]
    fn delete_impl<R>(&mut self, value: &R) -> Result<(), HyperLogLogError>
    where
        R: Hash + ?Sized,
    {
        // Create a new hasher.
        let mut hasher = self.builder.build_hasher();
        value.hash(&mut hasher);
        // Use a 64-bit hash value.
        let mut hash: u64 = hasher.finish();

        match &mut self.registers {
            Some(registers) => {
                // Calculate the register's index.
                let index: usize = (hash >> (64 - self.precision)) as usize;

                // Shift left the bits of the index.
                hash = (hash << self.precision) | (1 << (self.precision - 1));

                // Count leading zeros.
                let zeros: u32 = 1 + hash.leading_zeros();

                // Delete from register_counters
                if !self.register_counters.contains_key(&(index as u16)) {
                    return Err(HyperLogLogError::InvalidDenseDelete(
                        index as u32,
                        self.register_counters.len() as u16 as u32,
                    ));
                }

                let counter_map = self.register_counters.get_mut(&(index as u16)).unwrap();
                // The register count is 0, so we update the max register or delete the register.
                if counter_map.decrease_count_at_index(zeros as u8, 1)? {
                    let new_max_zeros = counter_map.arg_max();
                    registers.set_register(index, new_max_zeros as u32);
                    if new_max_zeros == 0 {
                        self.register_counters.remove(&(index as u16));
                    }
                }
            }
            None => {
                // We use sparse representation.

                // Encode hash value.
                let hash_code = self.encode_hash(hash);

                // Increment the count of the hash code.
                self.del_tmpset
                    .entry(hash_code)
                    .and_modify(|e| *e += 1)
                    .or_insert(1);

                // Merge temporary set into sparse representation.
                if self.del_tmpset.len() * 100 > self.counts.2 {
                    self.merge_deletes_sparse()?
                }
            }
        }
        Ok(())
    }

    #[inline] // Returns the precision of the HyperLogLogPF instance.
    fn precision(&self) -> u8 {
        self.precision
    }

    #[inline] // Returns an iterator to the Registers' values.
    fn registers_iter(&self) -> Option<impl Iterator<Item = u32> + '_> {
        self.registers
            .as_ref()
            .and_then(|registers| Some(registers.iter()))
    }



    #[inline] // Encodes the hash value as a u32 integer.
    fn encode_hash(&self, mut hash: u64) -> u32 {
        let index: u64 = u64::extract(hash, 64, 64 - Self::PRIME_PRECISION);

        let dif: u64 = u64::extract(hash, 64 - self.precision, 64 - Self::PRIME_PRECISION);

        if dif == 0 {
            // Shift left the bits of the index.
            hash = (hash << Self::PRIME_PRECISION) | (1 << Self::PRIME_PRECISION - 1);

            // Count leading zeros.
            let zeros: u32 = 1 + hash.leading_zeros();

            return ((index as u32) << 7) | (zeros << 1) | 1;
        }

        (index << 1) as u32
    }

    #[inline] // Extracts the index from a encoded hash.
    fn index(&self, hash_code: u32) -> usize {
        if hash_code & 1 == 1 {
            return u32::extract(hash_code, 32, 32 - self.precision) as usize;
        }

        u32::extract(
            hash_code,
            Self::PRIME_PRECISION + 1,
            Self::PRIME_PRECISION - self.precision + 1,
        ) as usize
    }

    #[inline] // Decodes a hash into the number of leading zeros and
              // the index of the correspondingn hash.
    fn decode_hash(&self, hash_code: u32) -> (u32, usize) {
        if hash_code & 1 == 1 {
            return (
                u32::extract(hash_code, 7, 1) + (Self::PRIME_PRECISION - self.precision) as u32,
                self.index(hash_code),
            );
        }

        let hash = hash_code << (32 - Self::PRIME_PRECISION + self.precision - 1);

        (hash.leading_zeros() + 1, self.index(hash_code))
    }

    // Creates a set of Registers for the given precision and copies the
    // register values from the sparse representation to the normal one.
    // The function assumes that the temporary sets are already flushed out.
    fn sparse_to_normal(&mut self) {
        let mut registers: RegistersPlus = RegistersPlus::with_count(self.counts.0);

        for (hash_code, cnt) in zip(self.sparse.into_iter(), self.sparse_counters.into_iter()) {
            let (zeros, index) = self.decode_hash(hash_code);

            registers.set_greater(index, zeros);
            let counter_map = self
                .register_counters
                .entry(index as u16)
                .or_insert(ZeroCountMap::new());
            counter_map.increase_count_at_index(zeros as u8, cnt);
        }

        self.registers = Some(registers);
        self.insert_tmpset.clear();
        self.del_tmpset.clear();
        self.sparse.clear();
    }

    fn merge_sparse(&mut self) -> Result<(), HyperLogLogError> {
        if !self.insert_tmpset.is_empty() {
            self.merge_inserts_sparse()?;
        }

        if !self.del_tmpset.is_empty() {
            self.merge_deletes_sparse()?;
        }

        self.convert_to_dense_if_required();

        Ok(())
    }

    fn convert_to_dense_if_required(&mut self) {
        if self.sparse.len() > self.counts.2 {
            self.sparse_to_normal();
        }
    }

    // Merges the hash codes stored in the temporary set to the sparse
    // representation.
    fn merge_inserts_sparse(&mut self) -> Result<(), HyperLogLogError> {
        if self.insert_tmpset.is_empty() {
            return Ok(());
        }

        let mut set_codes: Vec<(u32, u32)> = self.insert_tmpset.clone().into_iter().collect();
        set_codes.sort();

        let mut buf = DifIntVec::with_capacity(self.sparse.len());
        let mut sparse_counts = VarIntVec::with_capacity(self.sparse.len());

        let (mut set_iter, mut buf_iter, mut cnt_iter) = (
            set_codes.iter(),
            self.sparse.into_iter(),
            self.sparse_counters.into_iter(),
        );

        let (mut set_hash_option, mut buf_hash_option, mut cnt_option) =
            (set_iter.next(), buf_iter.next(), cnt_iter.next());

        while set_hash_option.is_some() || buf_hash_option.is_some() {
            if set_hash_option.is_none() {
                // Exists only in the sparse representation.
                buf.push(buf_hash_option.unwrap());
                let cnt = cnt_option.map_or(
                    Err(HyperLogLogError::InvalidSparseInsert(
                        "Sparse insert is empty".to_string(),
                    )),
                    |cnt| Ok(cnt),
                )?;
                sparse_counts.push(cnt);
                buf_hash_option = buf_iter.next();
                cnt_option = cnt_iter.next();
                continue;
            }

            if buf_hash_option.is_none() {
                let (set_hash_code, set_cnt) = set_hash_option.unwrap();
                // Exists only in the temporary set.
                buf.push(*set_hash_code);
                sparse_counts.push(*set_cnt);
                set_hash_option = set_iter.next();
                continue;
            }

            let ((set_hash_code, set_cnt), buf_hash_code) =
                (*set_hash_option.unwrap(), buf_hash_option.unwrap());

            if set_hash_code == buf_hash_code {
                // Exists in both the sparse representation and the temporary set.
                buf.push(set_hash_code);
                let cnt = cnt_option.map_or(
                    Err(HyperLogLogError::InvalidSparseInsert(
                        "Counter during sparse insert is empty".to_string(),
                    )),
                    |cnt| Ok(cnt),
                )?;
                sparse_counts.push(set_cnt + cnt);
                set_hash_option = set_iter.next();
                buf_hash_option = buf_iter.next();
                cnt_option = cnt_iter.next();
            } else if set_hash_code > buf_hash_code {
                buf.push(buf_hash_code);
                let cnt = cnt_option.map_or(
                    Err(HyperLogLogError::InvalidSparseInsert(
                        "Counter during sparse insert is empty".to_string(),
                    )),
                    |cnt| Ok(cnt),
                )?;
                sparse_counts.push(cnt);
                buf_hash_option = buf_iter.next();
                cnt_option = cnt_iter.next();
            } else {
                buf.push(set_hash_code);
                sparse_counts.push(set_cnt);
                set_hash_option = set_iter.next();
            }
        }

        self.sparse = buf;
        self.sparse_counters = sparse_counts;
        self.insert_tmpset.clear();

        Ok(())
    }

    fn merge_deletes_sparse(&mut self) -> Result<(), HyperLogLogError> {
        if self.del_tmpset.is_empty() {
            return Ok(());
        }

        let mut set_codes: Vec<(u32, u32)> = self.del_tmpset.clone().into_iter().collect();

        set_codes.sort();

        let mut buf = DifIntVec::with_capacity(self.sparse.len());
        let mut sparse_counts = VarIntVec::with_capacity(self.sparse.len());

        let (mut set_iter, mut buf_iter, mut cnt_iter) = (
            set_codes.iter(),
            self.sparse.into_iter(),
            self.sparse_counters.into_iter(),
        );

        let (mut set_hash_option, mut buf_hash_option, mut cnt_option) =
            (set_iter.next(), buf_iter.next(), cnt_iter.next());

        while set_hash_option.is_some() || buf_hash_option.is_some() {
            if set_hash_option.is_none() {
                // Exists only in the sparse representation.
                buf.push(buf_hash_option.unwrap());
                sparse_counts.push(cnt_option.unwrap());
                buf_hash_option = buf_iter.next();
                cnt_option = cnt_iter.next();
                continue;
            }

            if buf_hash_option.is_none() {
                // Exists only in the temporary set, dont care.
                continue;
            }

            let ((set_hash_code, set_cnt), buf_hash_code) =
                (*set_hash_option.unwrap(), buf_hash_option.unwrap());

            if set_hash_code == buf_hash_code {
                // Exists in both the sparse representation and the temporary set.
                let cnt = cnt_option.map_or(
                    Err(HyperLogLogError::InvalidSparseDelete(
                        set_cnt,
                        0,
                        "Counter is empty".to_string(),
                    )),
                    |cnt| Ok(cnt),
                )?;

                if set_cnt > cnt {
                    return Err(HyperLogLogError::InvalidSparseDelete(
                        set_cnt,
                        cnt,
                        "Delete count is greater than set count".to_string(),
                    ));
                } else if set_cnt < cnt_option.unwrap() {
                    buf.push(set_hash_code);
                    sparse_counts.push(cnt_option.unwrap() - set_cnt);
                } else {
                    // Delete the entry.
                }

                set_hash_option = set_iter.next();
                buf_hash_option = buf_iter.next();
                cnt_option = cnt_iter.next();
            } else if set_hash_code > buf_hash_code {
                buf.push(buf_hash_code);
                sparse_counts.push(cnt_option.unwrap());

                buf_hash_option = buf_iter.next();
                cnt_option = cnt_iter.next();
            } else {
                // Advance the set iterator.
                set_hash_option = set_iter.next();
            }
        }

        self.sparse = buf;
        self.sparse_counters = sparse_counts;

        self.del_tmpset.clear();
        Ok(())
    }

    // Returns an estimated bias correction based on empirical data.
    fn estimate_bias(&self, raw: f64) -> f64 {
        // Get a reference to raw estimates/biases for precision.
        let biases = &constants::BIAS_DATA[(self.precision - Self::MIN_PRECISION) as usize];
        let estimates =
            &constants::RAW_ESTIMATE_DATA[(self.precision - Self::MIN_PRECISION) as usize];

        // Raw estimate is first/last in estimates. Return the first/last bias.
        if raw <= estimates[0] {
            return biases[0];
        } else if estimates[estimates.len() - 1] <= raw {
            return biases[biases.len() - 1];
        }

        // Raw estimate is somewhere in between estimates.
        // Binary search for the calculated raw estimate.
        //
        // Here we unwrap because neither the values in `estimates`
        // nor `raw` are going to be NaN.
        let res = estimates.binary_search_by(|est| est.partial_cmp(&raw).unwrap());

        let (prv, idx) = match res {
            Ok(idx) => (idx - 1, idx),
            Err(idx) => (idx - 1, idx),
        };

        // Return linear interpolation between raw's neighboring points.
        let ratio = (raw - estimates[prv]) / (estimates[idx] - estimates[prv]);

        // Calculate bias.
        biases[prv] + ratio * (biases[idx] - biases[prv])
    }

    #[inline] // Returns an empirically determined threshold to decide on
              // the use of linear counting.
    fn threshold(precision: u8) -> f64 {
        match precision {
            4 => 10.0,
            5 => 10.0,
            6 => 40.0,
            7 => 80.0,
            8 => 220.0,
            9 => 400.0,
            10 => 900.0,
            11 => 1800.0,
            12 => 3100.0,
            13 => 6500.0,
            14 => 11500.0,
            15 => 20000.0,
            16 => 50000.0,
            17 => 120000.0,
            18 => 350000.0,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::prelude::IteratorRandom;
    use rand::Rng;
    use std::collections::HashSet;
    use std::hash::Hasher;
    use rand::prelude::SliceRandom;
    use siphasher::sip::SipHasher13;

    const SEEDED_HASH1: u64 = 0x9b8d7e6f;
    const SEEDED_HASH2: u64 = 0xdeadbeef;

    struct PassThroughHasher(u64);

    impl Hasher for PassThroughHasher {
        #[inline]
        fn finish(&self) -> u64 {
            self.0
        }

        #[inline]
        fn write(&mut self, _: &[u8]) {}

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.0 = i;
        }
    }

    struct SipHasher;

    impl BuildHasher for SipHasher {
        type Hasher = siphasher::sip::SipHasher13;

        #[inline]
        fn build_hasher(&self) -> Self::Hasher {
            SipHasher13::new_with_keys(SEEDED_HASH1, SEEDED_HASH2)
        }
    }

    fn setup_identical_hlls(num_elements: usize) -> (HyperLogLogPlus<u32,SipHasher>, HyperLogLogPlus<u32, SipHasher>) {
        let mut hll1 = HyperLogLogPlus::<u32,SipHasher>::new(14, SipHasher{}).unwrap();
        let mut hll2 = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        let mut rng = rand::thread_rng();

        // Generate random elements in the range 1-500
        for _ in 0..num_elements {
            let x = rng.gen_range(1, 50);
            hll1.insert(&x).unwrap();
            hll2.insert(&x).unwrap();
        }

        (hll1, hll2)
    }

    fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
        if a == 0.0 {
            return b.abs() < epsilon;
        }
        (a - b).abs() / a < epsilon
    }

    #[test]
    fn test_merge_both_sparse_equal() {
        let (mut hll1, hll2) = setup_identical_hlls(5000);

        assert!(hll1.is_sparse());
        assert!(hll2.is_sparse());

        let pre_merge_count = hll1.count().unwrap();
        hll1.merge(&hll2).unwrap();

        assert!(approx_equal(hll1.count().unwrap(), pre_merge_count, 0.01));
    }

    #[test]
    fn test_merge_self_sparse_other_dense_equal() {
        let (mut hll1, mut hll2) = setup_identical_hlls(50000);

        hll2.sparse_to_normal();

        assert!(hll1.is_sparse());
        assert!(!hll2.is_sparse());

        let pre_merge_count = hll1.count().unwrap();
        hll1.merge(&hll2).unwrap();
        assert!(approx_equal(hll1.count().unwrap(), pre_merge_count, 0.02));
    }

    #[test]
    fn test_merge_both_dense_equal() {
        let (mut hll1, mut hll2) = setup_identical_hlls(10000);

        hll1.sparse_to_normal();
        hll2.sparse_to_normal();

        assert!(!hll1.is_sparse());
        assert!(!hll2.is_sparse());

        let pre_merge_count = hll1.count().unwrap();
        hll1.merge(&hll2).unwrap();

        assert_eq!(hll1.count().unwrap(), pre_merge_count);
    }

    fn setup_different_hlls(
        num_elements: usize,
    ) -> (
        HashSet<u32>,
        HyperLogLogPlus<u32, SipHasher>,
        HashSet<u32>,
        HyperLogLogPlus<u32, SipHasher>,
    ) {
        let mut set1 = HashSet::new();
        let mut hll1 = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        let mut set2 = HashSet::new();
        let mut hll2 = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        let mut rng = rand::thread_rng();

        // Generate random elements in the range 1-50000 for hll1
        for _ in 0..num_elements {
            let x = rng.gen_range(1, 50000);
            set1.insert(x);
            hll1.insert(&x).unwrap();
        }

        // Generate different random elements in the range 50001-100000 for hll2
        for _ in 0..num_elements {
            let x = rng.gen_range(50001, 100000);
            set2.insert(x);
            hll2.insert(&x).unwrap();
        }

        (set1, hll1, set2, hll2)
    }

    #[test]
    fn test_merge_both_sparse_different() {
        let (mut set1, mut hll1, set2, hll2) = setup_different_hlls(5000);

        assert!(hll1.is_sparse());
        assert!(hll2.is_sparse());

        hll1.merge(&hll2).unwrap();

        let post_merge_estimate = hll1.count().unwrap();
        set1.extend(&set2);
        let post_merge_exact_count = set1.len();

        assert!(approx_equal(
            post_merge_estimate,
            post_merge_exact_count as f64,
            0.02
        ));
    }

    #[test]
    fn test_merge_one_sparse_other_dense_different() {
        let (mut set1, mut hll1, set2, mut hll2) = setup_different_hlls(5000);

        assert!(hll1.is_sparse());
        hll2.sparse_to_normal();
        assert!(!hll2.is_sparse());

        hll1.merge(&hll2).unwrap();

        let post_merge_estimate = hll1.count().unwrap();
        set1.extend(&set2);
        let post_merge_exact_count = set1.len();

        assert!(approx_equal(
            post_merge_estimate,
            post_merge_exact_count as f64,
            0.04
        ));
    }

    #[test]
    fn test_merge_both_dense_different() {
        let (mut set1, mut hll1, set2, mut hll2) = setup_different_hlls(5000);

        hll1.sparse_to_normal();
        hll2.sparse_to_normal();
        assert!(!hll1.is_sparse());
        assert!(!hll2.is_sparse());

        hll1.merge(&hll2).unwrap();

        let post_merge_estimate = hll1.count().unwrap();
        set1.extend(&set2);
        let post_merge_exact_count = set1.len();

        assert!(approx_equal(
            post_merge_estimate,
            post_merge_exact_count as f64,
            0.05
        ));
    }

    #[test]
    fn test_merge_one_dense_other_sparse_different() {
        let (mut set1, mut hll1, set2, hll2) = setup_different_hlls(5000);

        hll1.sparse_to_normal();
        assert!(!hll1.is_sparse());
        assert!(hll2.is_sparse());

        hll1.merge(&hll2).unwrap();

        let post_merge_estimate = hll1.count().unwrap();
        set1.extend(&set2);
        let post_merge_exact_count = set1.len();

        assert!(approx_equal(
            post_merge_estimate,
            post_merge_exact_count as f64,
            0.03
        ));
    }

    #[test]
    fn test_insert_any() {
        let mut hll = HyperLogLogPlus::<i32, SipHasher>::new(14, SipHasher{}).unwrap();
        let elements: Vec<i32> = (1..=1000).chain(1..=1000).collect();

        // Insert elements into the HyperLogLogPlus
        for element in elements.iter() {
            hll.insert(element).unwrap();
        }

        // Check the estimate
        let estimate = hll.count().unwrap();
        let num_elements = elements.len() / 2;
        let error = (estimate as f64 - num_elements as f64).abs() / num_elements as f64;

        assert!(error <= 0.02, "Relative error is more than 2%");
    }

    #[test]
    fn test_insert_delete_simple() {
        let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();

        let elements = vec![1, 3, 7];

        for element in elements.iter() {
            hll.insert(element).unwrap();
        }

        assert_eq!(hll.count().unwrap().round() as u64, elements.len() as u64);

        hll.delete(&7).unwrap();
        hll.delete(&1).unwrap();
        hll.insert(&9).unwrap();
        hll.insert(&10).unwrap();
        hll.insert(&11).unwrap();

        assert_eq!(hll.count().unwrap().round() as i64, 4);
    }

    #[test]
    fn test_insert_delete_sparse() {
        let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        let mut rng = rand::thread_rng();
        let mut test_set: Vec<u32> = vec![];

        for _ in 0..5000 {
            let val: u32 = rng.gen_range(0, 1000);
            test_set.push(val);
            hll.insert(&val).unwrap();
        }
        let count = hll.count().unwrap();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert_eq!(count as usize, actual_count);

        for i in 0..2000 {
            hll.delete_any(&test_set[i]).unwrap();
        }

        // Delete first 2000 elements from test set
        test_set = test_set.into_iter().skip(2000).collect::<Vec<u32>>();

        assert!(hll.is_sparse());

        let count = hll.count().unwrap();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert_eq!(count as usize, actual_count);

        for val in test_set.clone().into_iter() {
            hll.delete_any(&val).unwrap();
        }

        let count = hll.count().unwrap();
        assert_eq!(count as usize, 0);

        test_set.clear();

        for _ in 0..5000 {
            let val: u32 = rng.gen_range(0, 50);
            test_set.push(val);
            hll.insert(&val).unwrap();
        }

        assert!(hll.is_sparse());
        let count = hll.count().unwrap();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert_eq!(count as usize, actual_count);

        let mut iter = test_set.clone().into_iter();

        for _ in 0..2500 {
            if let Some(val) = iter.next() {
                hll.delete_any(&val).unwrap();
            }
        }

        let test_set = test_set.into_iter().skip(2500).collect::<Vec<u32>>();

        let count = hll.count().unwrap();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert_eq!(count as usize, actual_count);

        for val in iter {
            hll.delete_any(&val).unwrap();
        }

        assert_eq!(hll.count().unwrap(), 0.0);
    }

    #[test]
    fn test_insert_delete_dense() {
        let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        let mut rng = rand::thread_rng();
        let mut test_set: Vec<u32> = vec![];

        for _ in 0..20000 {
            let val: u32 = rng.gen_range(0, 10000);
            test_set.push(val);
            hll.insert(&val).unwrap();
        }

        assert!(!hll.is_sparse());
        let count = hll.count().unwrap();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert!(approx_equal(count, actual_count as f64, 0.02));

        for i in 0..15000 {
            hll.delete_any(&test_set[i]).unwrap();
        }
        let old_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        // Delete first 15000 elements from test set
        test_set = test_set.into_iter().skip(15000).collect::<Vec<u32>>();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert!(old_count > actual_count);
        let count = hll.count().unwrap();
        assert!(approx_equal(count, actual_count as f64, 0.02));

        for val in test_set.clone().into_iter() {
            hll.delete_any(&val).unwrap();
        }

        let count = hll.count().unwrap();
        assert_eq!(count as usize, 0);

        test_set.clear();
        assert_eq!(hll.register_counters.len(), 0);

        for _ in 0..30000 {
            let val: u32 = rng.gen_range(0, 10000);
            test_set.push(val);
            hll.insert(&val).unwrap();
        }

        assert!(!hll.is_sparse());
        let count = hll.count().unwrap();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert!(approx_equal(count, actual_count as f64, 0.01));

        for i in 0..20000 {
            hll.delete_any(&test_set[i]).unwrap();
        }

        let old_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        let test_set = test_set.into_iter().skip(20000).collect::<Vec<u32>>();
        let actual_count = test_set.clone().into_iter().collect::<HashSet<u32>>().len();
        assert!(actual_count < old_count);

        let count = hll.count().unwrap();
        assert!(approx_equal(count, actual_count as f64, 0.02));

        for val in test_set.clone().into_iter() {
            hll.delete_any(&val).unwrap();
        }

        assert_eq!(hll.count().unwrap(), 0.0);
    }

    #[test]
    fn test_insert_any_with_random_inputs() {
        let mut rng = rand::thread_rng(); // Create a random number generator
        let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();

        // Generate 1000 random elements in the range 1-500
        let mut elements: Vec<u32> = Vec::with_capacity(100000);
        for _ in 0..10000 {
            let x = rng.gen_range(1, 5000);
            elements.push(x);
        }

        // Insert elements into the HyperLogLogPlus
        for element in elements.iter() {
            hll.insert(element).unwrap();
        }

        // Calculate the actual cardinality by deduplicating the elements
        let actual_cardinality = elements
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .len();

        // Check the estimate
        let estimate = hll.count().unwrap();
        let error = (estimate as f64 - actual_cardinality as f64).abs() / actual_cardinality as f64;

        assert!(error <= 0.02, "Relative error is more than 2%");
        assert_eq!(hll.is_sparse(), true);
    }

    #[test]
    fn test_sparse_encode_hash() {
        let hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(14, SipHasher{}).unwrap();

        //                 < ... 14 ... > .. 25 .. >
        let index: u64 = 0b0000000000111000000000000;

        let hash: u64 = 0b1101;

        let hash_code = hll.encode_hash((index << 64 - 25) | hash);

        assert_eq!(hash_code, (index << 7) as u32 | (35 + 1 << 1) | 1);

        //                 < ... 14 ... > .. 25 .. >
        let index: u64 = 0b0000000000111000000000010;

        let hash: u64 = 0b1101;

        let hash_code = hll.encode_hash((index << 64 - 25) | hash);

        assert_eq!(hash_code, (index << 1) as u32);
    }

    #[test]
    fn test_sparse_decode_hash() {
        let hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(8, SipHasher{}).unwrap();

        let (zeros, index) = hll.decode_hash(hll.encode_hash(0xffffff8000000000));

        assert_eq!((zeros, index), (1, 0xff));

        let (zeros, index) = hll.decode_hash(hll.encode_hash(0xff00000000000000));

        assert_eq!((zeros, index), (57, 0xff));

        let (zeros, index) = hll.decode_hash(hll.encode_hash(0xff30000000000000));

        assert_eq!((zeros, index), (3, 0xff));

        let (zeros, index) = hll.decode_hash(hll.encode_hash(0xaa10000000000000));

        assert_eq!((zeros, index), (4, 0xaa));

        let (zeros, index) = hll.decode_hash(hll.encode_hash(0xaa0f000000000000));

        assert_eq!((zeros, index), (5, 0xaa));
    }

    #[test]
    fn test_sparse_merge_sparse() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        let hashes: [u64; 3] = [0xf000017000000000, 0x000fff8f00000000, 0x0f00017000000000];

        // Insert a couple of hashes.
        hll.insert(&hashes[0]).unwrap();

        hll.insert(&hashes[1]).unwrap();

        assert_eq!(hll.insert_tmpset.len(), 2);

        assert_eq!(hll.sparse.len(), 0);

        // Merge and check hashes.
        hll.merge_sparse().unwrap();

        assert_eq!(hll.sparse.count(), 2);

        assert_eq!(hll.insert_tmpset.len(), 0);

        // Insert another hash.
        hll.insert(&hashes[2]).unwrap();

        // Merge and check hashes again.
        hll.merge_sparse().unwrap();

        assert_eq!(hll.sparse.count(), 3);

        assert_eq!(hll.insert_tmpset.len(), 0);
    }

    #[test]
    fn test_sparse_trigger_sparse_to_normal() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(4, SipHasher{}).unwrap();

        for i in 1..4 {
            hll.insert(&(1 << i)).unwrap();
        }

        assert!(hll.registers.is_none());
        assert!(hll.is_sparse());

        hll.insert(&(1 << 5)).unwrap();

        assert!(!hll.is_sparse());
        assert!(hll.registers.is_some());

        assert_eq!(hll.insert_tmpset.len(), 0);

        assert_eq!(hll.sparse.len(), 0);
    }

    #[test]
    fn test_sparse_to_normal_complex() {
        let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        let mut rng = rand::thread_rng();

        // Generate 10000 random elements in the range 1-5000
        let mut elements: Vec<u32> = Vec::with_capacity(1000000);
        for _ in 0..10000 {
            let x = rng.gen_range(1, 5000);
            elements.push(x);
        }
        // Insert elements into the HyperLogLogPlus
        for element in &elements {
            hll.insert(element).unwrap();
        }

        // Check the estimate
        assert!(hll.is_sparse());
        let estimate = hll.count().unwrap();
        let num_elements = elements.clone().into_iter().collect::<HashSet<u32>>().len();
        let error = (estimate as f64 - num_elements as f64).abs() / num_elements as f64;

        assert!(error <= 0.01, "Relative error is more than 1%");

        hll.sparse_to_normal();

        // Check the estimate
        let estimate = hll.count().unwrap();
        let error = (estimate as f64 - num_elements as f64).abs() / num_elements as f64;

        assert!(error <= 0.02, "Relative error is more than 2%");
    }

    #[test]
    fn test_sparse_to_normal_complex_with_deletes() {
        let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        let mut rng = rand::thread_rng();
        // Generate 10000 random elements in the range 1-5000
        let mut elements: Vec<u32> = Vec::with_capacity(10000);
        for _ in 0..10000 {
            let x = rng.gen_range(1, 1000);
            elements.push(x);
        }

        // Insert elements into the HyperLogLogPlus
        for element in &elements {
            hll.insert(element).unwrap();
        }

        // Generate 2000 random indices for deletion
        let mut delete_indices: HashSet<usize> = HashSet::with_capacity(2000);
        while delete_indices.len() < 2000 {
            let i = rng.gen_range(0, elements.len());
            delete_indices.insert(i);
        }

        for &i in &delete_indices {
            assert!(hll.delete_any(&elements[i]).is_ok());
        }

        // Remove the deleted elements from `elements`
        let elements: Vec<u32> = elements
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !delete_indices.contains(i))
            .map(|(_, element)| element)
            .collect();

        let num_elements = elements.clone().into_iter().collect::<HashSet<u32>>().len();

        // Check the estimate
        assert!(hll.is_sparse());
        let estimate = hll.count().unwrap();
        let error = (estimate as f64 - num_elements as f64).abs() / num_elements as f64;

        // HyperLogLogPlus has a probabilistic error rate.
        // The following assert checks if the relative error is within expected bounds (e.g., 2% for p=14).
        assert!(error <= 0.02, "Relative error is more than 2%");

        hll.sparse_to_normal();

        // Check the estimate
        assert!(!hll.is_sparse());
        let estimate = hll.count().unwrap();
        let error = (estimate as f64 - num_elements as f64).abs() / num_elements as f64;
        assert!(error <= 0.02, "Relative error is more than 2%");
    }

    #[test]
    fn test_sparse_to_normal_simple() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        hll.insert(&1).unwrap();

        assert_eq!(hll.count().unwrap() as u64, 1);

        hll.merge_sparse().unwrap();

        hll.sparse_to_normal();

        assert_eq!(hll.count().unwrap() as u64, 1);

        assert!(hll.registers.is_some());

        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        hll.insert(&2).unwrap();
        hll.insert(&3).unwrap();
        hll.insert(&4).unwrap();

        assert_eq!(hll.count().unwrap() as u64, 3);

        hll.sparse_to_normal();

        assert_eq!(hll.count().unwrap() as u64, 3);
    }

    #[test]
    fn test_sparse_count() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        let hashes: [u64; 6] = [
            0x00010fffffffffff,
            0x00020fffffffffff,
            0x00030fffffffffff,
            0x00040fffffffffff,
            0x00050fffffffffff,
            0x00050fffffffffff,
        ];

        for hash in &hashes {
            hll.insert(hash).unwrap();
        }

        // Calls a merge_sparse().
        hll.count().unwrap();

        let hash_codes: Vec<u32> = hll.sparse.into_iter().collect();

        let sip_hasher = SipHasher13::new_with_keys(SEEDED_HASH1, SEEDED_HASH2);
        let post_hash_codes: Vec<u64> = hashes
            .iter()
            .map(|hash| {
                let mut hasher = sip_hasher.clone();
                hash.hash(&mut hasher);
                hasher.finish()
            })
            .collect();

        let expected_hash_codes: Vec<u32> = post_hash_codes
            .iter()
            .map(|hash| hll.encode_hash(*hash))
            .collect();
        assert_eq!(hll.count().unwrap() as u64, 5);
        // Do not check the order of the hash codes.
        assert_eq!(
            hash_codes.into_iter().collect::<HashSet<u32>>(),
            expected_hash_codes.into_iter().collect::<HashSet<u32>>()
        );
    }

    #[test]
    fn test_estimate_bias() {
        let hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(4, SipHasher{}).unwrap();

        let bias = hll.estimate_bias(14.0988);

        assert!((bias - 7.5988).abs() <= 1e-5);

        let bias = hll.estimate_bias(10.0);

        assert!((bias - 10.0).abs() < 1e-5);

        let bias = hll.estimate_bias(80.0);

        assert!((bias - (-1.7606)).abs() < 1e-5);

        let hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        let bias = hll.estimate_bias(55391.4373);

        assert!((bias - 39416.9373).abs() < 1e-5);
    }

    #[test]
    fn test_estimate_bias_count() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(4, SipHasher{}).unwrap();

        hll.sparse_to_normal();

        for i in 0..10 {
            hll.insert(&i).unwrap();
        }

        assert!((10.0 - hll.count().unwrap()).abs() < 1.0);
    }

    #[test]
    fn test_merge_error() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();
        let other: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(12, SipHasher{}).unwrap();

        assert_eq!(
            hll.merge(&other),
            Err(HyperLogLogError::IncompatiblePrecision)
        );
    }

    #[test]
    fn test_merge_both_sparse() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();
        let mut other: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        other.insert(&0x00010ffffffffff).unwrap();
        other.insert(&0x00020ffffffffff).unwrap();
        other.insert(&0x00030ffffffffff).unwrap();
        other.insert(&0x00040ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();

        assert_eq!(other.count().unwrap().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 5);

        assert!(hll.is_sparse() && other.is_sparse());

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 5);

        assert!(hll.is_sparse() && other.is_sparse());

        other.insert(&0x00060ffffffffff).unwrap();
        other.insert(&0x00070ffffffffff).unwrap();
        other.insert(&0x00080ffffffffff).unwrap();
        other.insert(&0x00090ffffffffff).unwrap();
        other.insert(&0x000a0ffffffffff).unwrap();

        assert_eq!(other.count().unwrap().trunc() as u64, 10);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 10);

        assert!(hll.is_sparse() && other.is_sparse());
    }

    #[test]
    fn test_merge_both_normal() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();
        let mut other: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        hll.sparse_to_normal();
        other.sparse_to_normal();

        other.insert(&0x00010ffffffffff).unwrap();
        other.insert(&0x00020ffffffffff).unwrap();
        other.insert(&0x00030ffffffffff).unwrap();
        other.insert(&0x00040ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();

        assert_eq!(other.count().unwrap().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 5);

        assert!(!hll.is_sparse() && !other.is_sparse());

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 5);

        assert!(!hll.is_sparse() && !other.is_sparse());

        other.insert(&0x00060fffffffffff).unwrap();
        other.insert(&0x00070fffffffffff).unwrap();
        other.insert(&0x00080fffffffffff).unwrap();
        other.insert(&0x00090fffffffffff).unwrap();
        other.insert(&0x000a0fffffffffff).unwrap();

        assert_eq!(other.count().unwrap().trunc() as u64, 10);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 10);

        assert!(!hll.is_sparse() && !other.is_sparse());
    }

    #[test]
    fn test_merge_sparse_to_normal() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();
        let mut other: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        hll.sparse_to_normal();

        other.insert(&0x00010ffffffffff).unwrap();
        other.insert(&0x00020ffffffffff).unwrap();
        other.insert(&0x00030ffffffffff).unwrap();
        other.insert(&0x00040ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();

        assert_eq!(other.count().unwrap().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 5);

        assert!(!hll.is_sparse() && other.is_sparse());

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 5);

        assert!(!hll.is_sparse() && other.is_sparse());

        other.insert(&0x00060ffffffffff).unwrap();
        other.insert(&0x00070ffffffffff).unwrap();
        other.insert(&0x00080ffffffffff).unwrap();
        other.insert(&0x00090ffffffffff).unwrap();
        other.insert(&0x000a0ffffffffff).unwrap();

        assert_eq!(other.count().unwrap().trunc() as u64, 10);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 10);

        assert!(!hll.is_sparse() && other.is_sparse());
    }

    #[test]
    fn test_merge_normal_to_sparse() {
        let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();
        let mut other: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

        other.sparse_to_normal();

        other.insert(&0x00010ffffffffff).unwrap();
        other.insert(&0x00020ffffffffff).unwrap();
        other.insert(&0x00030ffffffffff).unwrap();
        other.insert(&0x00040ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();
        other.insert(&0x00050ffffffffff).unwrap();

        assert_eq!(other.count().unwrap().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().unwrap().trunc() as u64, 5);

        assert!(!hll.is_sparse() && !other.is_sparse());
    }

    #[test]
    fn test_serialization_deserialization_sparse() {
        let mut hll1 = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();

        let mut rng = rand::thread_rng();

        // Generate 100 random elements in the range 1-500
        let mut elements: HashSet<u32> = HashSet::with_capacity(100);
        for _ in 0..100 {
            let x = rng.gen_range(1, 500);
            elements.insert(x);
        }

        // Insert elements into the HyperLogLogPlus
        for element in elements.iter() {
            hll1.insert(element).unwrap();
        }

        // Serialize
        let bytes = hll1.to_bytes().unwrap();

        // Deserialize
        let mut hll2: HyperLogLogPlus<i32, SipHasher> = HyperLogLogPlus::from_bytes_compact(&bytes, SipHasher{}).unwrap();

        // Verify counts are the same
        assert_eq!(hll1.count().unwrap(), hll2.count().unwrap());

        hll2.insert(&501).unwrap();

        assert!(hll2.count().is_err());

        // Deserialize
        let mut hll3: HyperLogLogPlus<i32, SipHasher> = HyperLogLogPlus::deserialize(&bytes, SipHasher{}).unwrap();

        // Verify counts are the same
        assert_eq!(hll1.count().unwrap(), hll3.count().unwrap());
        let cur_cnt = hll1.count().unwrap();

        hll3.insert(&501).unwrap();
        assert!((hll3.count().unwrap() - (cur_cnt + 1.0)).abs() < 0.0001);
        hll3.insert(&501).unwrap();
        assert!((hll3.count().unwrap() - (cur_cnt + 1.0)).abs() < 0.0001);

        hll3.delete_any(&501).unwrap();
        assert!((hll3.count().unwrap() - (cur_cnt + 1.0)).abs() < 0.0001);
        hll3.delete_any(&501).unwrap();
        assert!((hll3.count().unwrap() - cur_cnt).abs() < 0.0001);

        let mut hll4: HyperLogLogPlus<i32, SipHasher> = HyperLogLogPlus::deserialize_compact(&bytes, SipHasher{}).unwrap();
        assert_eq!(hll1.count().unwrap(), hll4.count().unwrap());
    }

    #[test]
    fn test_serialization_deserialization_dense() {
        let mut hll1 = HyperLogLogPlus::<i32, SipHasher>::new(14, SipHasher{}).unwrap();

        let mut rng = rand::thread_rng();

        // Generate 10000 random elements in the range 1-50000
        let mut elements: HashSet<i32> = HashSet::with_capacity(100);
        for _ in 0..10000 {
            let x = rng.gen_range(1, 500);
            elements.insert(x);
        }

        hll1.sparse_to_normal();

        // Insert elements into the HyperLogLogPlus
        for element in elements.iter() {
            hll1.insert(element).unwrap();
        }

        // Serialize
        let bytes = hll1.serialize().unwrap();

        // Deserialize
        let mut hll2: HyperLogLogPlus<i32, SipHasher> = HyperLogLogPlus::deserialize_compact(&bytes, SipHasher{}).unwrap();

        // Verify counts are the same
        assert_eq!(hll1.count().unwrap(), hll2.count().unwrap());

        let cur_cnt = hll1.count().unwrap();

        // Deserialize
        let mut hll3: HyperLogLogPlus<i32, SipHasher> = HyperLogLogPlus::<i32, SipHasher>::deserialize(&bytes, SipHasher{}).unwrap();

        // Verify counts are the same
        assert_eq!(hll1.count().unwrap(), hll3.count().unwrap());

        hll3.insert(&50011).unwrap();
        assert!((hll3.count().unwrap() - (cur_cnt + 1.0)) < 0.1);
        hll3.insert(&50011).unwrap();
        assert!((hll3.count().unwrap() - (cur_cnt + 1.0)) < 0.1);

        hll3.delete_any(&50011).unwrap();
        assert!((hll3.count().unwrap() - (cur_cnt + 1.0)).abs() < 0.1);
        hll3.delete_any(&50011).unwrap();
        assert!((hll3.count().unwrap() - cur_cnt).abs() < 0.1);

        let mut hll4: HyperLogLogPlus<i32, SipHasher> = HyperLogLogPlus::deserialize_compact(&bytes, SipHasher{}).unwrap();
        assert_eq!(hll1.count().unwrap(), hll4.count().unwrap());
    }

    #[test]
    fn test_merge_multiple_hyperloglogs() {
        let num_hlls = 50;
        let mut hlls = Vec::with_capacity(num_hlls);
        let mut set: HashMap<u32, u32> = HashMap::new();
        let mut cnt_sparse_hlls = 0;
        let mut cnt_dense_hlls = 0;
        let mut hll_vals = Vec::with_capacity(num_hlls);
        let mut rng = rand::thread_rng();
        for _i in 0..num_hlls {
            let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
            let num_points = rand::thread_rng().gen_range(1, 30000);
            let mut tmp_set: HashMap<u32, u32> = HashMap::new();
            for _j in 0..num_points {
                let x = rng.gen_range(1, 100000);
                hll.insert(&x).unwrap();
                // Increment count in set by 1
                let count = set.entry(x).or_insert(0);
                *count += 1;
                let count = tmp_set.entry(x).or_insert(0);
                *count += 1;
            }
            hll_vals.push(tmp_set);

            match hll.is_sparse() {
                true => cnt_sparse_hlls += 1,
                false => cnt_dense_hlls += 1,
            }
            hlls.push(hll);
        }
        assert!(cnt_sparse_hlls > 0);
        assert!(cnt_dense_hlls > 0);
        assert_eq!(hll_vals.len(), num_hlls);

        let mut serialized_hlls = vec![];
        // Merge all the HyperLogLogs
        let mut merged_hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        for mut hll in hlls.into_iter() {
            merged_hll.merge(&hll).unwrap();
            serialized_hlls.push(hll.serialize().unwrap());
        }

        // Verify the merged HyperLogLog has the correct count
        assert!(approx_equal(
            merged_hll.count().unwrap(),
            set.len() as f64,
            0.02
        ));

        // Deserialize and merge .
        let mut merged_hll2 = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        for serialized_hll in serialized_hlls.iter() {
            let hll = HyperLogLogPlus::<u32, SipHasher>::deserialize(serialized_hll, SipHasher{}).unwrap();
            merged_hll2.merge(&hll).unwrap();
        }

        assert!(approx_equal(
            merged_hll2.count().unwrap(),
            set.len() as f64,
            0.02
        ));

        // Compact deserialize and merge.
        let mut merged_hll3 = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        for serialized_hll in serialized_hlls.iter() {
            let hll = HyperLogLogPlus::<u32, SipHasher>::deserialize_compact(serialized_hll, SipHasher{}).unwrap();
            merged_hll3.merge_compact(&hll).unwrap();
        }

        assert!(approx_equal(
            merged_hll3.count().unwrap(),
            set.len() as f64,
            0.02
        ));

        // Delete some elements from the merged HyperLogLog
        let insert_count = set.len();
        let mut merged_hll_post_del = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
        for i in 0..num_hlls {
            // Pick a random subset from hlls_vals[i] to delete.
            let mut keys: Vec<_> = hll_vals[i].keys().collect::<Vec<_>>();
            keys.shuffle(&mut rng);
            let mut hll = HyperLogLogPlus::<u32, SipHasher>::deserialize(&serialized_hlls[i], SipHasher{}).unwrap();
            assert!(approx_equal(hll.count().unwrap(), hll_vals[i].len() as f64, 0.03));
            let num_to_delete = rng.gen_range(1, keys.len());
            for j in 0..num_to_delete {
                // Delete from set too
                let count = set.get_mut(&keys[j]).unwrap();
                *count -= 1;
                if *count == 0 {
                    set.remove(&keys[j]);
                }
                hll.delete(&keys[j]).unwrap();
            }
            merged_hll_post_del.merge_compact(&hll).unwrap();
        }

        assert!(set.len() < insert_count);
        assert!(approx_equal(
            merged_hll_post_del.count().unwrap(),
            set.len() as f64,
            0.02
        ));
    }

    #[test]
    fn test_delete_post_deserialization() {
        let mode = vec![false, true];
        for keep_sparse in mode {
            let mut hll = HyperLogLogPlus::<u32, SipHasher>::new(14, SipHasher{}).unwrap();
            let mut counter = HashMap::new();
            for _j in 0..500 {
                let x = rand::thread_rng().gen_range(1, 100);
                hll.insert(&x).unwrap();
                // Increment counter of x in counter
                let count = counter.entry(x).or_insert(0);
                *count += 1;
            }

            let bytes = hll.serialize().unwrap();

            let mut hll2 = HyperLogLogPlus::<u32, SipHasher>::deserialize(&bytes, SipHasher{}).unwrap();
            // Verify the merged HyperLogLog has the correct count
            assert!(approx_equal(
                hll2.count().unwrap(),
                counter.len() as f64,
                0.02
            ));

            if !keep_sparse {
                hll2.merge_sparse().unwrap();
                hll2.sparse_to_normal();
            }

            // Delete elements from counter and HyperLogLog
            while counter.len() > 0 {
                // Pick a random element in counter
                let key = counter
                    .keys()
                    .choose(&mut rand::thread_rng())
                    .unwrap()
                    .clone();
                let val = counter.get(&key).unwrap();
                // Delete from counter
                hll.delete_any(&key).unwrap();
                if val == &1 {
                    counter.remove(&key);
                } else {
                    let count = counter.entry(key).or_insert(0);
                    *count -= 1;
                }
                // Check counts
                assert!(approx_equal(
                    hll.count().unwrap(),
                    counter.len() as f64,
                    0.03
                ));
            }
        }
    }

    #[cfg(feature = "bench-units")]
    mod benches {
        extern crate test;

        use super::*;
        use rand::prelude::*;
        use test::{black_box, Bencher};

        #[bench]
        fn bench_plus_insert_normal(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

            hll.sparse_to_normal();

            b.iter(|| {
                for i in 0u64..1000 {
                    hll.insert(&(u64::max_value() - i));
                }
            })
        }

        #[bench]
        fn bench_insert_normal_with_hash(b: &mut Bencher) {
            let mut rng = rand::thread_rng();

            let workload: Vec<String> = (0..2000)
                .map(|_| format!("- {} - {} -", rng.gen::<u64>(), rng.gen::<u64>()))
                .collect();

            b.iter(|| {
                let mut hll: HyperLogLogPlus<&String, DefaultBuildHasher> =
                    HyperLogLogPlus::new(16, DefaultBuildHasher {}).unwrap();

                hll.sparse_to_normal();

                for val in &workload {
                    hll.insert(&val);
                }

                let val = hll.count().unwrap();

                black_box(val);
            })
        }

        #[bench]
        fn bench_plus_count_normal(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

            hll.sparse_to_normal();

            b.iter(|| {
                let count = hll.count().unwrap();
                black_box(count);
            })
        }

        #[bench]
        fn bench_plus_merge_sparse(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(16, SipHasher{}).unwrap();

            for i in 0u64..500 {
                hll.insert(&(i << 39));
            }

            assert_eq!(hll.insert_tmpset.len(), 500);

            let set = hll.insert_tmpset.clone();

            b.iter(|| {
                hll.insert_tmpset = set.clone();
                hll.merge_sparse()
            });

            assert!(hll.registers.is_none());

            assert_eq!(hll.insert_tmpset.len(), 0);
        }

        #[bench]
        fn bench_estimate_bias(b: &mut Bencher) {
            let hll: HyperLogLogPlus<u64, SipHasher> = HyperLogLogPlus::new(18).unwrap();

            b.iter(|| {
                let bias = hll.estimate_bias(275468.768);
                black_box(bias);
                let bias = hll.estimate_bias(587532.522);
                black_box(bias);
                let bias = hll.estimate_bias(1205430.993);
                black_box(bias);
                let bias = hll.estimate_bias(1251260.649);
                black_box(bias);
            });
        }
    }
}
