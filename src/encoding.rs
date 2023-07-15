use crate::HyperLogLogError;
use serde::{Deserialize, Serialize};

const SEVEN_LSB_MASK: u32 = (1 << 7) - 1;

const VAR_INT_MASK: u32 = !SEVEN_LSB_MASK;

const MSB_MASK: u8 = 1 << 7;

// A Vector of bytes containing variable length encoded unsigned integers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarIntVec(Vec<u8>);

// A Vector containing difference encoded unsigned integers
// stored in a `VarIntVec`.
//
// Numbers stored are assumed to be in increasing order, hence the
// difference between a new number and `last` will always be positive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifIntVec {
    // The count of numbers stored.
    count: usize,
    // The last number inserted.
    last: u32,
    // The inner Varint encoded vector.
    buf: VarIntVec,
}

pub struct VarIntVecIntoIter<'a> {
    index: usize,
    inner: &'a VarIntVec,
}

pub struct DifIntVecIntoIter<'a> {
    index: usize,
    last: u32,
    inner: &'a DifIntVec,
}

impl VarIntVec {
    pub fn new() -> Self {
        VarIntVec(Vec::new())
    }

    pub fn mem_size(&self) -> usize {
        self.0.len() * std::mem::size_of::<u8>()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.clone()
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, HyperLogLogError> {
        Ok(VarIntVec(bytes))
    }

    pub fn with_capacity(cap: usize) -> Self {
        VarIntVec(Vec::with_capacity(cap))
    }

    #[inline] // Varint encodes `val` and pushes it into the vector.
    pub fn push(&mut self, mut val: u32) {
        while val & VAR_INT_MASK != 0 {
            self.0.push((val & SEVEN_LSB_MASK) as u8 | MSB_MASK);

            val >>= 7;
        }

        self.0.push((val & SEVEN_LSB_MASK) as u8);
    }

    #[inline] // Varint decodes a number starting at `index`.
              //
              // Returns the decoded number and the index of the next
              // number in the vector.
    fn decode(&self, index: usize) -> (u32, usize) {
        let (mut i, mut val) = (0, 0);

        while self.0[index + i] & MSB_MASK != 0 {
            val |= ((self.0[index + i] as u32) & SEVEN_LSB_MASK) << (i * 7);

            i += 1;
        }

        val |= (self.0[index + i] as u32) << (i * 7);

        (val, index + i + 1)
    }
}

/// Map of the number of leading zeros to the count of numbers with that many leading zeros.
/// Hence it is map of u8 to u32.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZeroCountMap(Vec<u8>);

impl ZeroCountMap {
    pub fn new() -> Self {
        ZeroCountMap(Vec::new())
    }

    pub fn increase_count_at_index(&mut self, key: u8, val: u32) {
        if val == 0 {
            return;
        }
        assert!(key > 0);

        let mut values = self.convert_to_vec();
        let len: u8 = values.len() as u8;
        if key < len {
            values[key as usize] += val;
        } else if key == len {
            values.push(val);
        } else {
            self.0.push(0);
            self.0.push(key - len);
            self.push_non_zero(val);
            return;
        }

        self.encode_vec_to_buf(values);
    }

    // Returns true if the count at index is zero after decreasing it by val.
    pub fn decrease_count_at_index(&mut self, key: u8, val: u32) -> Result<bool, HyperLogLogError> {
        if val == 0 {
            return Ok(false);
        }
        assert!(key > 0);

        let mut values = self.convert_to_vec();

        let val_is_zero;
        if values.len() > key as usize {
            if values[key as usize] < val {
                return Err(HyperLogLogError::InvalidDenseDelete(
                    val,
                    values[key as usize],
                ));
            }
            values[key as usize] -= val;
            val_is_zero = values[key as usize] == 0;
        } else {
            return Err(HyperLogLogError::InvalidDenseDelete(
                key as u32,
                values.len() as u32,
            ));
        }


        self.encode_vec_to_buf(values);

        Ok(val_is_zero)
    }

    pub fn arg_max(&self) -> u32 {
        let values = self.convert_to_vec();
        // The last value is guaranteed to be non zero.
        (values.len() - 1) as u32
    }

    pub fn serialize(&self) -> Vec<u8> {
        self.0.clone()
    }

    pub fn deserialize(bytes: Vec<u8>) -> Result<Self, HyperLogLogError> {
        Ok(ZeroCountMap(bytes))
    }

    pub fn mem_size(&self) -> usize {
        self.0.len() * std::mem::size_of::<u8>()
    }

    // -------------------------------- Internal Functions -----------------------------------------
    // Encoding Scheme:
    // A Vector of bytes containing run length and variable length encoded unsigned integers.
    // The vector is similar to a VarIntVec, but it also encodes runs of zeros.
    // Hence if a number in the vector is zero, the next byte stores the count of consecutive zeros.
    // The vector also always terminates with a non zero number.
    // All numbers after the last key are assumed to be zero.

    // The terminology used in the implementation is as follows:
    // `index` is the index of the byte in the  byte vector.
    // `key` is the index of the key we are looking at. For our purpose the key is a number between 0 and 50.

    #[inline] // Varint encodes `val` and pushes it into the vector.
    fn push_non_zero(&mut self, mut val: u32) {
        if val == 0 {
            return;
        }

        while val & VAR_INT_MASK != 0 {
            self.0.push((val & SEVEN_LSB_MASK) as u8 | MSB_MASK);

            val >>= 7;
        }

        self.0.push((val & SEVEN_LSB_MASK) as u8);
    }

    fn convert_to_vec(&self) -> Vec<u32> {
        // Decode all the values
        let mut values = Vec::new();
        // The first value is always zero since the count of leading zeros is calculated as
        // let zeros: u32 = 1 + hash.leading_zeros();
        values.push(0);
        let mut index = 0;
        let mut zero_index = 0;
        while index < self.0.len() {
            let (decoded_val, new_index, new_zero_index) = self.decode(index, zero_index);
            values.push(decoded_val);
            index = new_index;
            zero_index = new_zero_index;
        }
        values
    }

    fn encode_vec_to_buf(&mut self, values: Vec<u32>) {
        self.0 = Vec::with_capacity(values.len());
        let mut zero_run_length: u8 = 0;

        for v in values.into_iter().skip(1) {
            if v == 0 {
                zero_run_length += 1;
            } else {
                if zero_run_length > 0 {
                    self.0.push(0);
                    self.0.push(zero_run_length);
                    zero_run_length = 0;
                }
                self.push_non_zero(v);
            }
        }
    }

    #[inline] // Varint decodes a number starting at `index`.
              //
              // Returns the decoded number and the index of the next
              // number in the vector.
              // zero_index is the number of consecutive zeros we have seen.
    fn decode(&self, index: usize, zero_index: usize) -> (u32, usize, usize) {
        let (mut i, mut val) = (0, 0);

        if self.0[index] == 0 {
            assert!(self.0.len() > index + 1);
            let run_length = self.0[index + 1];
            if zero_index != run_length as usize {
                return (0, index, zero_index + 1);
            } else {
                // We have reached the end of the zero run. But we are guaranteed that the next number is not zero.
                return self.decode(index + 2, 0);
            }
        }

        while self.0[index + i] & MSB_MASK != 0 {
            val |= ((self.0[index + i] as u32) & SEVEN_LSB_MASK) << (i * 7);

            i += 1;
        }

        val |= (self.0[index + i] as u32) << (i * 7);

        (val, index + i + 1, 0)
    }
}

impl DifIntVec {
    pub fn new() -> Self {
        DifIntVec {
            count: 0,
            last: 0,
            buf: VarIntVec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        DifIntVec {
            count: 0,
            last: 0,
            buf: VarIntVec::with_capacity(cap),
        }
    }

    pub fn mem_size(&self) -> usize {
        self.buf.mem_size()
    }

    #[inline] // Difference encodes `val` and pushes it into a `VarIntVec`.
    pub fn push(&mut self, val: u32) {
        self.buf.push(val - self.last);
        self.last = val;
        self.count += 1;
    }

    #[inline] // Returns the count of numbers in the vector.
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline] // Returns the inner vector's length.
    pub fn len(&self) -> usize {
        self.buf.0.len()
    }

    // Clear the inner vector.
    pub fn clear(&mut self) {
        self.count = 0;
        self.last = 0;
        self.buf.0.clear();
    }
}

impl<'a> IntoIterator for &'a VarIntVec {
    type Item = u32;
    type IntoIter = VarIntVecIntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        VarIntVecIntoIter {
            index: 0,
            inner: self,
        }
    }
}

impl<'a> IntoIterator for &'a DifIntVec {
    type Item = u32;
    type IntoIter = DifIntVecIntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DifIntVecIntoIter {
            index: 0,
            last: 0,
            inner: self,
        }
    }
}

impl<'a> Iterator for VarIntVecIntoIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.inner.0.len() {
            None
        } else {
            let (val, index) = self.inner.decode(self.index);

            self.index = index;

            Some(val)
        }
    }
}

impl<'a> Iterator for DifIntVecIntoIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.inner.buf.0.len() {
            None
        } else {
            let (dif, index) = self.inner.buf.decode(self.index);

            self.index = index;

            self.last += dif;

            Some(self.last)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encode() {
        let mut nums = VarIntVec::with_capacity(2);

        nums.push(7);
        nums.push(128);
        nums.push(300);

        assert_eq!(nums.0, vec![7, 128, 1, 172, 2]);

        nums.push(127);

        assert_eq!(nums.0, vec![7, 128, 1, 172, 2, 127]);

        nums.push(255);

        assert_eq!(nums.0, vec![7, 128, 1, 172, 2, 127, 255, 1]);

        nums.push(0xffffffff);

        assert_eq!(
            nums.0,
            vec![7, 128, 1, 172, 2, 127, 255, 1, 255, 255, 255, 255, 15]
        );
    }

    #[test]
    fn test_varintvec() {
        let mut varint_vec = VarIntVec::new();

        // Insert some numbers
        varint_vec.push(1);
        varint_vec.push(128);
        varint_vec.push(10000);
        varint_vec.push(12345678);

        // Decode and check the numbers
        let (val, next_index) = varint_vec.decode(0);
        assert_eq!(val, 1);
        let (val, next_index) = varint_vec.decode(next_index);
        assert_eq!(val, 128);
        let (val, next_index) = varint_vec.decode(next_index);
        assert_eq!(val, 10000);
        let (val, _next_index) = varint_vec.decode(next_index);
        assert_eq!(val, 12345678);
    }

    #[test]
    fn test_varint_zeros() {
        let mut varint_vec = VarIntVec::new();
        assert_eq!(varint_vec.0.capacity() * std::mem::size_of::<u8>(), 0);
        varint_vec.push(0);
        varint_vec.push(0);
        varint_vec.push(0);
        varint_vec.push(0);
        varint_vec.push(0);
        varint_vec.push(0);

        varint_vec.push(1);

        assert_eq!(varint_vec.0.capacity() * std::mem::size_of::<u8>(), 8);
    }

    #[test]
    fn test_arg_max() {
        // Test with non-zero elements
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        v.increase_count_at_index(2, 1);
        v.increase_count_at_index(3, 9);
        assert_eq!(v.arg_max(), 3);

        // Test with all elements being zeros except the last one
        let mut v = ZeroCountMap::new();
        for i in 1..63 {
            v.increase_count_at_index(i, 0);
        }
        v.increase_count_at_index(63, 9);
        assert_eq!(v.arg_max(), 63);

        // Test with all elements being the same non-zero number
        let mut v = ZeroCountMap::new();
        for i in 1..64 {
            v.increase_count_at_index(i, 7);
        }
        assert_eq!(v.arg_max(), 63);

        // Test with empty ZeroCountMap
        let v = ZeroCountMap::new();
        assert_eq!(v.arg_max(), 0);

        // Insert and delete some elements
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        v.increase_count_at_index(2, 1);
        v.increase_count_at_index(3, 9);
        v.increase_count_at_index(4, 7);
        assert!(v.decrease_count_at_index(4, 7).is_ok()); // This makes 3rd index to 0
        assert_eq!(v.arg_max(), 3);

        // Insert elements, then delete all of them
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        v.increase_count_at_index(2, 1);
        assert!(v.decrease_count_at_index(1, 5).is_ok());
        assert!(v.decrease_count_at_index(2, 1).is_ok());
        assert_eq!(v.arg_max(), 0);

        // Delete an element that doesn't exist
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        assert!(v.decrease_count_at_index(2, 1).is_err());
        assert_eq!(v.arg_max(), 1);
    }

    #[test]
    fn test_run_encoded_vec_increase_count() {
        // Test with non-zero elements
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        v.increase_count_at_index(2, 1);
        v.increase_count_at_index(3, 9);
        assert_eq!(v.convert_to_vec(), vec![0, 5, 1, 9]);

        // Test with zeros
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 0);
        v.increase_count_at_index(2, 0);
        v.increase_count_at_index(3, 0);
        // All zeros, so we dont need to store anything
        assert_eq!(v.convert_to_vec(), vec![0]);

        // Test with mix of zeros and non-zeros
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 3);
        v.increase_count_at_index(2, 0);
        v.increase_count_at_index(3, 0);
        v.increase_count_at_index(4, 0);
        v.increase_count_at_index(5, 2);
        v.increase_count_at_index(6, 0);
        v.increase_count_at_index(7, 0);
        assert_eq!(v.convert_to_vec(), vec![0, 3, 0, 0, 0, 2]);
    }

    #[test]
    fn test_decrease_count_at_index() {
        // Test with non-zero elements
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        v.increase_count_at_index(2, 1);
        v.increase_count_at_index(3, 9);
        assert!(v.decrease_count_at_index(1, 2).is_ok());
        assert!(v.decrease_count_at_index(3, 3).is_ok());
        assert_eq!(v.convert_to_vec(), vec![0, 3, 1, 6]);

        // Test decrease to zero
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        v.increase_count_at_index(2, 1);
        assert!(v.decrease_count_at_index(1, 5).is_ok());
        assert!(v.decrease_count_at_index(2, 1).is_ok());
        assert_eq!(v.convert_to_vec(), vec![0]);

        // Test with mix of zeros and non-zeros
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 3);
        v.increase_count_at_index(2, 0);
        v.increase_count_at_index(3, 0);
        v.increase_count_at_index(4, 0);
        v.increase_count_at_index(5, 2);
        v.increase_count_at_index(6, 0);
        v.increase_count_at_index(7, 0);
        assert!(v.decrease_count_at_index(1, 1).is_ok());
        assert!(v.decrease_count_at_index(5, 1).is_ok());
        assert_eq!(v.convert_to_vec(), vec![0, 2, 0, 0, 0, 1]);

        // Test with out of range key and decrease more than available
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 3);
        assert!(v.decrease_count_at_index(2, 1).is_err());
        assert!(v.decrease_count_at_index(1, 4).is_err());
    }

    #[test]
    fn test_run_encoded_vec_increase_count_complex() {
        // Test with a large number of zeros in between non-zeros
        let mut v = ZeroCountMap::new();
        v.increase_count_at_index(1, 5);
        for i in 1..50 {
            v.increase_count_at_index(i, 0);
        }
        v.increase_count_at_index(50, 1);
        let mut expected_output = vec![0, 5];
        expected_output.extend(vec![0; 48]);
        expected_output.push(1);
        assert_eq!(v.convert_to_vec(), expected_output);
        // Check the memory usage
        // 2 bytes for 5 and 1 and 2 bytes for the 49 zeros
        assert_eq!(v.mem_size(), 4);

        // Test with all elements being zeros except the last one
        let mut v = ZeroCountMap::new();
        for i in 0..63 {
            v.increase_count_at_index(i, 0);
        }
        v.increase_count_at_index(63, 9);
        let mut expected_output = vec![0; 63];
        expected_output.push(9);
        assert_eq!(v.convert_to_vec(), expected_output);
        // Check the memory usage
        // 1 byte for 9 and 2 bytes for the 63 zeros
        assert_eq!(v.mem_size(), 3);

        // Test with all elements being the same non-zero number
        let mut v = ZeroCountMap::new();
        for i in 1..64 {
            v.increase_count_at_index(i, 7);
        }
        let mut expected_output = vec![0];
        expected_output.extend(vec![7; 63]);

        assert_eq!(v.convert_to_vec(), expected_output);
        // Check the memory usage
        // 1 bytes for each of the 63 7s
        assert_eq!(v.mem_size(), 63);
    }

    #[test]
    fn test_varint_decode() {
        let input: Vec<u32> = (1..256).chain(16400..16500).collect();

        let mut nums = VarIntVec::with_capacity(100);

        for num in &input {
            nums.push(*num);
        }

        let output: Vec<u32> = nums.into_iter().collect();

        assert_eq!(input, output);
    }

    #[test]
    fn test_difvarint_encode() {
        let mut nums = DifIntVec::with_capacity(2);

        // push: 7
        nums.push(7);
        // push: 128 - 7 = 121
        nums.push(128);
        // push: 300 - 128 = 172
        nums.push(300);

        assert_eq!(nums.buf.0, vec![7, 121, 172, 1]);

        // push: 555 - 300 = 255
        nums.push(555);

        assert_eq!(nums.buf.0, vec![7, 121, 172, 1, 255, 1]);

        // push: 556 - 555 = 1
        nums.push(556);

        assert_eq!(nums.buf.0, vec![7, 121, 172, 1, 255, 1, 1]);
    }

    #[test]
    fn test_difvarint_decode() {
        let input: Vec<u32> = (1..256).chain(16400..16500).collect();

        let mut nums = DifIntVec::with_capacity(100);

        for num in &input {
            nums.push(*num);
        }

        let output: Vec<u32> = nums.into_iter().collect();

        assert_eq!(input, output);
    }

    #[cfg(feature = "bench-units")]
    mod benches {
        extern crate test;

        use super::*;
        use test::Bencher;

        #[bench]
        fn bench_varint_push_min(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(1000);

            b.iter(|| buf.push(u32::min_value()));
        }

        #[bench]
        fn bench_varint_push_max(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(1000);

            b.iter(|| buf.push(u32::max_value()));
        }

        #[bench]
        fn bench_varint_decode_min(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(10);

            for _ in 0..10 {
                buf.push(u32::min_value());
            }

            b.iter(|| buf.decode(0));
        }

        #[bench]
        fn bench_varint_decode_max(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(10);

            for _ in 0..10 {
                buf.push(u32::max_value());
            }

            b.iter(|| buf.decode(0));
        }
    }
}
