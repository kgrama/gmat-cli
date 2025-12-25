//! I/O serialization for UnifiedBlock

use super::super::configs::BlockConfig;
use super::super::traits::ElementMask;
use super::{EncodeHelper, UnifiedBlock};
use half::f16;
use std::io::{Read, Result, Write};
use std::marker::PhantomData;

impl<const ROWS: usize, C: BlockConfig + EncodeHelper> UnifiedBlock<ROWS, C> {
    /// Write block to binary stream
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        // Write all zero_maps first (for ROWS rows)
        for i in 0..ROWS {
            self.zero_map[i].write_le(w)?;
        }

        // If empty, stop
        if self.is_empty() {
            return Ok(());
        }

        // Write shared scale_log
        w.write_all(&self.scale_log.to_bits().to_le_bytes())?;

        // Write data for each row
        for i in 0..ROWS {
            self.signs[i].write_le(w)?;
            self.octave_shift[i].write_le(w)?;
            w.write_all(&self.magnitudes[i][..C::MAG_ARRAY_SIZE])?;
        }

        Ok(())
    }

    /// Read block from binary stream
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        // Read zero_maps for all rows
        let mut zero_map = [C::Mask::default(); 2];
        for zm in zero_map.iter_mut().take(ROWS) {
            *zm = C::Mask::read_le(r)?;
        }

        // Check if empty
        let is_empty = (0..ROWS).all(|i| zero_map[i] == C::Mask::default());

        if is_empty {
            return Ok(Self::new_empty());
        }

        // Read shared scale_log
        let mut scale_bytes = [0u8; 2];
        r.read_exact(&mut scale_bytes)?;
        let scale_log = f16::from_bits(u16::from_le_bytes(scale_bytes));

        // Read data for each row
        let mut signs = [C::Mask::default(); 2];
        let mut octave_shift = [C::Mask::default(); 2];
        let mut magnitudes = [[0u8; 16]; 2];

        for i in 0..ROWS {
            signs[i] = C::Mask::read_le(r)?;
            octave_shift[i] = C::Mask::read_le(r)?;
            r.read_exact(&mut magnitudes[i][..C::MAG_ARRAY_SIZE])?;
        }

        Ok(Self {
            scale_log,
            zero_map,
            signs,
            octave_shift,
            magnitudes,
            _phantom: PhantomData,
        })
    }
}
