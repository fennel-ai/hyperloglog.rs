use std::collections::HashMap;
use std::hash::Hash;
use std::io::{Read, Write};
use std::marker::PhantomData;
use crate::common::RegistersPlus;
use crate::encoding::{DifIntVec, ZeroCountMap, VarIntVec};
use crate::{HyperLogLogError, HyperLogLogPlus};
use serde::{Deserialize, Serialize};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use byteorder::{ReadBytesExt, WriteBytesExt};

const CODEC_VERSION: u8 = 0x01;


#[derive(Clone, Debug, Serialize, Deserialize)]
struct HyperLogLogPlusSerializable<H>
    where
        H: Hash + ?Sized,
{
    precision: u8,
    counts:    (usize, usize, usize),
    sparse:    DifIntVec,
    registers: Option<RegistersPlus>,
    phantom:   PhantomData<H>,
}

impl<H> HyperLogLogPlus<H>
    where
        H: Hash + ?Sized,
{
    // Serialization format:
    // 1. CODEC_VERSION: u8
    // 2. Length of non-counter data: u32
    // 3. Non-counter data: HyperLogLogPlusSerializable
    // 4. Sparse or Dense: bool
    // 5. Counter data for either sparse or dense: VarIntVec or HashMap<u32, u32>
    pub fn to_bytes(&mut self) -> Result<Bytes, HyperLogLogError> {
        if self.is_sparse() {
            self.merge_sparse()?;
        }
        let serializable = HyperLogLogPlusSerializable {
            precision: self.precision,
            counts: self.counts,
            sparse: self.sparse.clone(),
            registers: self.registers.clone(),
            phantom: PhantomData::<H>,
        };
        let non_counter_data = bincode::serialize(&serializable).map_err(|e| HyperLogLogError::SerializationError(e.to_string()))?;
        let counter_data = match self.is_sparse() {
            true => self.sparse_counters.to_bytes(),
            false => serialize_register_counters(&self.register_counters)?.to_vec(),
        };
        let buf = BytesMut::new();
        let mut w = buf.writer();
        w.write_u8(CODEC_VERSION).map_err(|e| HyperLogLogError::SerializationError(e.to_string()))?;
        w.write_u32::<byteorder::BigEndian>(non_counter_data.len() as u32).map_err(|e| HyperLogLogError::SerializationError(e.to_string()))?;
        w.write_all(&non_counter_data).map_err(|e| HyperLogLogError::SerializationError(e.to_string()))?;
        w.write_u8(self.is_sparse() as u8).map_err(|e| HyperLogLogError::SerializationError(e.to_string()))?;
        w.write_all(&counter_data).map_err(|e| HyperLogLogError::SerializationError(e.to_string()))?;
        Ok(w.into_inner().freeze())
    }


    // This function only reads the non-counter data and is meant to be used during read operations to prevent large allocations.
    pub fn from_bytes_compact(bytes: &[u8]) -> Result<Self, HyperLogLogError> {
        if bytes.is_empty() {
            return Err(HyperLogLogError::EmptyBuffer);
        }

        let r = std::io::Cursor::new(bytes);
        let mut r = r.reader();
        let codec = r.read_u8().map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
        if codec != CODEC_VERSION {
            return Err(HyperLogLogError::InvalidCodec(codec));
        }

        let len = r.read_u32::<byteorder::BigEndian>().map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
        let mut bytes = vec![0; len as usize];
        r.read_exact(&mut bytes).map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;

        let HyperLogLogPlusSerializable::<H> {
            precision,
            counts,
            sparse,
            registers,
            phantom: _,
        } = bincode::deserialize(&bytes).map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;

        Ok(HyperLogLogPlus {
            builder: Self::default_hasher(),
            precision,
            counts,
            insert_tmpset: HashMap::new(),
            del_tmpset: HashMap::new(),
            sparse,
            registers,
            sparse_counters: VarIntVec::new(),
            register_counters: HashMap::new(),
            phantom: PhantomData::<H>,
        })
    }


    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HyperLogLogError> {
        if bytes.is_empty() {
            return Err(HyperLogLogError::EmptyBuffer);
        }

        let mut num_bytes_read = 0;

        let r = std::io::Cursor::new(bytes);
        let mut r = r.reader();
        let codec = r.read_u8().map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
        num_bytes_read += 1;
        if codec != CODEC_VERSION {
            return Err(HyperLogLogError::InvalidCodec(codec));
        }

        let non_counter_bytes_len = r.read_u32::<byteorder::BigEndian>().map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
        let mut non_counter_bytes = vec![0; non_counter_bytes_len as usize];
        r.read_exact(&mut non_counter_bytes).map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
        num_bytes_read += 4 + non_counter_bytes_len as usize;


        let HyperLogLogPlusSerializable::<H> {
            precision,
            counts,
            sparse,
            registers,
            phantom: _,
        } = bincode::deserialize(&non_counter_bytes).map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;

        let is_sparse = r.read_u8().map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
        num_bytes_read += 1;

        let counter_bytes_len = bytes.len() - num_bytes_read;
        let mut counter_bytes: Vec<u8> = vec![0; counter_bytes_len as usize];
        r.read_exact(&mut counter_bytes).map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;

        if is_sparse as u8 == 1 {
            let sparse_counters = VarIntVec::from_bytes(counter_bytes)?;
            Ok(HyperLogLogPlus {
                builder: Self::default_hasher(),
                precision,
                counts,
                insert_tmpset: HashMap::new(),
                del_tmpset: HashMap::new(),
                sparse,
                registers,
                sparse_counters,
                register_counters: HashMap::new(), // Initialize an empty register_counters
                phantom: PhantomData::<H>,
            })
        } else {
            let register_counters = deserialize_register_counters(Bytes::from(counter_bytes))?;
            Ok(HyperLogLogPlus {
                builder: Self::default_hasher(),
                precision,
                counts,
                insert_tmpset: HashMap::new(),
                del_tmpset: HashMap::new(),
                sparse,
                registers,
                sparse_counters: VarIntVec::new(),
                register_counters,
                phantom: PhantomData::<H>,
            })
        }
    }
}


// Extract the required bits from the indicator byte using the last 6 bits.
const SIX_BIT_INFO_MASK: u8 = 0b0011_1111;

// Indicator mask for NULL registers counts that require 6 bits.
const SIX_BIT_NULL_MASK: u8 = 0b0000_0000;

// Indicator mask for NULL registers counts that require 14 bits.
const FOURTEEN_BIT_NULL_MASK: u8 = 0b0100_0000;

// Indicator mask for NON NULL registers counts that require 14 bits.
const FOURTEEN_BIT_NON_NULL_MASK: u8 = 0b1100_0000;

// Indicator mask for NULL registers counts that require 6 bits.
const SIX_BIT_NOT_NULL_MASK : u8 = 0b1000_0000;

// The following functions are used to serialize the dense counters which is a hashmap of u16 -> ZeroCountMap.
// The hashmap is serialized as a vector of bytes, where there is an indicator byte followed by the bytes to represent the ZeroCountMap.
// The indicator byte works as follows:
// If the first bit is 0, then we are looking at a series of NULL registers counts.
//      A. If the second bit is 0, then the next 6 bits represent the number of NULL registers counts.
//      B. If the second bit is 1, then the next 14 ( 6 + 8 )  bits represent number of NULL registers counts.
// If the first bit is 1, then we are looking at a non-NULL register count.
//      A. As before, if the second bit is 0, then the next 6 bits represent the number of bytes taken by the ZeroCountMap.
//      B. If the second bit is 1, then the next 14 ( 6 + 8 ) bits represent the number of bytes taken by the ZeroCountMap
fn serialize_register_counters(register_counters: &HashMap<u16, ZeroCountMap>) -> Result<Bytes, HyperLogLogError> {
    // The buffer is initialized with minimum capacity of threes byte ( one for indicator and two for value) per register.
    let mut buffer = Vec::with_capacity(register_counters.len() * 3);
    let mut null_count: i32 = 0;
    let mut last_key: i32 = -1;

    // Sort the keys so that we can keep track of null registers.
    let mut keys: Vec<&u16> = register_counters.keys().collect();
    keys.sort();

    for key in keys {
        // Find the number of null registers till the current key.
        while *key as i32 > (last_key + null_count) + 1 {
            null_count += 1;
        }

        // Write any remaining null counts.
        if null_count > 0 {
            if null_count < 64 {
                buffer.push(SIX_BIT_NULL_MASK | (null_count as u8));
            } else {
                buffer.push(FOURTEEN_BIT_NULL_MASK | ((null_count >> 8) as u8));
                buffer.push((null_count & 0xFF) as u8);
            }
            null_count = 0;
        }

        let run_encoded = register_counters.get(key).unwrap();
        let bytes = run_encoded.serialize();
        let byte_len = bytes.len() as u16;

        // Write the indicator byte and length.
        if byte_len < 64 {
            // Length fits within 6 bits.
            buffer.push(SIX_BIT_NOT_NULL_MASK | (byte_len as u8));
        } else {
            // Length requires 14 bits.
            buffer.push(FOURTEEN_BIT_NON_NULL_MASK | ((byte_len >> 8) as u8));
            // Store the remaining 8 bits.
            buffer.push((byte_len & 0xFF) as u8);
        }

        // Write the bytes of the ZeroCountMap.
        buffer.extend_from_slice(&bytes);
        last_key = *key as i32;
    }

    Ok(Bytes::from(buffer))
}

fn deserialize_register_counters(bytes: Bytes) -> Result<HashMap<u16, ZeroCountMap>, HyperLogLogError> {
    let mut index: u16 = 0;
    let mut register_counters = HashMap::new();
    let r = std::io::Cursor::new(bytes);
    let mut r = r.reader();
    while let Ok(indicator_byte) = r.read_u8() {
        if indicator_byte & SIX_BIT_NOT_NULL_MASK == 0 {
            // We are looking at a series of null registers.
            let count = if indicator_byte & FOURTEEN_BIT_NULL_MASK == 0 {
                // The count is represented in the next 6 bits.
                (indicator_byte & SIX_BIT_INFO_MASK) as u16
            } else {
                let next_byte = r.read_u8().map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
                // The count is represented in the next 14 bits.
                ((indicator_byte as u16 & SIX_BIT_INFO_MASK as u16) << 8) | (next_byte as u16)
            };

            index += count;
        } else {
            // We are looking at a non-null register.
            let byte_len = if indicator_byte & FOURTEEN_BIT_NULL_MASK == 0 {
                // The length is represented in the next 6 bits.
                (indicator_byte & SIX_BIT_INFO_MASK) as usize
            } else {
                let next_byte = r.read_u8().map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
                // The length is represented in the next 14 bits.
                ((indicator_byte as usize & SIX_BIT_INFO_MASK as usize) << 8) | (next_byte as usize)
            };
            let mut buffer: Vec<u8> = vec![0; byte_len];
            // Get the bytes of the ZeroCountMap.
            r.read_exact(&mut buffer).map_err(|e| HyperLogLogError::DeserializationError(e.to_string()))?;
            // Add the ZeroCountMap to the HashMap.

            register_counters.insert(index, ZeroCountMap::deserialize(buffer.to_vec())?);
            index += 1;
        }
    }

    Ok(register_counters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::prelude::SliceRandom;

    fn generate_zero_count_map() -> ZeroCountMap {
        let mut v = ZeroCountMap::new();
        let mut rng = rand::thread_rng();
        let mut index_val = HashMap::new();
        for _i in 0..30 {
            let num = rng.gen_range(1, 63);
            let val = rng.gen_range(0, 10000000);
            v.increase_count_at_index(num, val);
            let count = index_val.entry(num).or_insert(0);
            *count += val;
        }
        // Pick a random subset of keys and decrease the count by a random amount.
        let mut keys: Vec<u8> = index_val.keys().map(|x| *x).collect();
        keys.shuffle(&mut rng);
        for i in 0..rng.gen_range(1, keys.len()) {
            let key = keys[i];
            let val = rng.gen_range(0, index_val[&key]);
            v.decrease_count_at_index(key, val).unwrap();
            let count = index_val.entry(key).or_insert(0);
            *count -= val;
        }
        v
    }

    fn setup() -> HashMap<u16, ZeroCountMap> {
        let mut register_counters = HashMap::new();
        let mut rng = rand::thread_rng();

        let num_registers = rng.gen_range(1, 16000);
        let mut set = std::collections::HashSet::new();
        for i in 0..num_registers {
            // Skip random registers.
            if rng.gen_range(0, 100) < 60 {
                continue;
            }
            set.insert(i);
            register_counters.insert(i, generate_zero_count_map());
        }
        register_counters
    }

    fn compare_maps(map1: &HashMap<u16, ZeroCountMap>, map2: &HashMap<u16, ZeroCountMap>) {
        assert_eq!(map1.len(), map2.len());
        for (key, val) in map1 {
            match map2.get(key) {
                Some(other_val) => assert_eq!(val, other_val),
                None => assert!(false),
            }
        }
    }

    #[test]
    fn test_serialize_deserialize_register_counters() {
        let register_counters = setup();
        let serialized = serialize_register_counters(&register_counters).unwrap();
        let deserialized = deserialize_register_counters(serialized).unwrap();

        compare_maps(&register_counters, &deserialized);
    }

}
