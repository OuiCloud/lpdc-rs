//! This module contains the available LDPC codes

use super::*;

#[derive(Copy, Clone, PartialEq, Eq)]

enum Code {
    TM1280,
    TM1536,
    TM2048,
    TM5120,
    TM6144,
    TM8192,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ECC<const K: usize, const N: usize> {
    code: Code,
}

pub const TM1280: ECC<1024, 1280> = ECC { code: Code::TM1280 };
pub const TM1536: ECC<1024, 1536> = ECC { code: Code::TM1536 };
pub const TM2048: ECC<1024, 2048> = ECC { code: Code::TM2048 };
pub const TM5120: ECC<4096, 5120> = ECC { code: Code::TM5120 };
pub const TM6144: ECC<4096, 6144> = ECC { code: Code::TM6144 };
pub const TM8192: ECC<4096, 8192> = ECC { code: Code::TM8192 };

impl<const K: usize, const N: usize> ECC<K, N> {
    /// Block length aka code length.
    #[inline(always)]
    #[must_use]
    pub const fn n(&self) -> usize {
        N
    }

    /// Data length aka code dimension.
    #[inline(always)]
    #[must_use]
    pub const fn k(&self) -> usize {
        K
    }

    /// Number of parity bits not transmitted.
    #[inline(always)]
    #[must_use]
    pub const fn punctured_bits(&self) -> usize {
        (N - K) / 2
    }

    /// Sub-matrix size (used in parity check matrix construction).
    #[inline(always)]
    #[must_use]
    pub const fn submatrix_size(&self) -> usize {
        (N - K) / 2
    }

    /// Circulant block size (used in generator matrix construction).
    #[inline(always)]
    #[must_use]
    pub const fn circulant_size(&self) -> usize {
        (N - K) / 8
    }

    /// Sum of the parity check matrix (number of parity check edges).
    #[inline]
    #[must_use]
    pub const fn paritycheck_sum(&self) -> usize {
        (K + 7 * (N - K) / 8) * 4
    }

    /// Length of the working area required for the message-passing decoder.
    /// Equal to 2 * paritycheck_sum + 3*n + 3*p - 2*k
    #[inline]
    #[must_use]
    pub const fn decode_working_len(&self) -> usize {
        2 * self.paritycheck_sum() + 3 * N + 3 * self.punctured_bits() - 2 * N
    }

    /// Length of the u8 working area required for the message-passing decoder.
    /// Equal to (n + punctured_bits - k)/8.
    #[inline]
    #[must_use]
    pub const fn decode_ms_working_u8_len(&self) -> usize {
        (N + self.punctured_bits() - K) / 8
    }

    /// Length of output required from any decoder.
    /// Equal to (n+punctured_bits)/8.
    #[inline]
    #[must_use]
    pub const fn output_len(&self) -> usize {
        (N + self.punctured_bits()) / 8
    }

    /// Get a  reference to the generator matrix for this code
    pub const fn compact_generator(&self) -> &[u64] {
        match self.code {
            Code::TM1280 => &TM1280_G,
            Code::TM1536 => &TM1536_G,
            Code::TM2048 => &TM2048_G,
            Code::TM5120 => &TM5120_G,
            Code::TM6144 => &TM6144_G,
            Code::TM8192 => &TM8192_G,
        }
    }

    /// Get an iterator over all parity check matrix edges for this code.
    ///
    /// All included codes have a corresponding parity check matrix, which is defined
    /// using a very compact representation that can be expanded into the full parity
    /// check matrix. This function returns an efficient iterator over all edges in
    /// the parity check matrix, in a deterministic but otherwise unspecified order.
    ///
    /// The iterator yields (check, variable) pairs, corresponding to the index of a
    /// row and column in the parity check matrix which contains a 1.
    pub fn iter(self) -> ParityIter {
        let phi = match self.code {
            Code::TM1280 => &PHI_J_K_M128,
            Code::TM1536 => &PHI_J_K_M256,
            Code::TM2048 => &PHI_J_K_M512,
            Code::TM5120 => &PHI_J_K_M512,
            Code::TM6144 => &PHI_J_K_M1024,
            Code::TM8192 => &PHI_J_K_M2048,
        };

        let m = self.submatrix_size();
        let prototype_cols = (N + self.punctured_bits()) / m;
        let prototype = match prototype_cols {
            5 => &TM_R12_H,
            7 => &TM_R23_H,
            11 => &TM_R45_H,
            _ => unreachable!(),
        };

        let subm = prototype[0][0][0];

        ParityIter {
            phi,
            prototype,
            m,
            logmd4: (m / 4).trailing_zeros() as usize,
            modm: m - 1,
            modmd4: (m / 4) - 1,
            rowidx: 0,
            colidx: 0,
            sub_mat_idx: 0,
            check: 0,
            sub_mat: subm,
            sub_mat_val: (subm & 0x3F) as usize,
        }
    }
}

/// Iterator over a code's parity check matrix.
///
/// Iterating gives values `(check, variable)` which are the indices
/// of an edge on the parity check matrix, where `check` is the row
/// and `variable` is the column.
///
/// `ParityIter` is obtained from `LDPCCode::iter_paritychecks()`.
pub struct ParityIter {
    phi: &'static [[u16; 26]; 4],
    prototype: &'static [[[u8; 11]; 4]; 3],
    m: usize,
    logmd4: usize, // log2(M/4), used to multiply and divide by M/4
    modm: usize,   // the bitmask to AND with to accomplish "mod M", equals m-1
    modmd4: usize, // the bitmask to AND with to accomplish "mod M/4", equals (m/4)-1
    rowidx: usize,
    colidx: usize,
    sub_mat_idx: usize,
    sub_mat: u8,
    sub_mat_val: usize,
    check: usize,
}

impl Iterator for ParityIter {
    type Item = (usize, usize);

    /// Compute the next parity edge.
    ///
    /// This function really really wants to be inlined for performance. It does almost no
    /// computation but returns thousands of times, so the overhead of a function call
    /// completely dominates its runtime if not inlined.
    #[inline(always)]
    fn next(&mut self) -> Option<(usize, usize)> {
        // This function demands careful optimisation. Not only will it be the hottest inner loop
        // of any algorithm on the parity check matrix, but because it's an iterator it's called
        // from the start thousands of times. We use this annoying loop structure so that the
        // hot path, entering and returning almost right away, is as simple as possible.
        //
        // Terms:
        //  * prototype is the set of 3 4x11 design matrices
        //  * sub_mat_idx chooses one of those 3 design matrices
        //  * rowidx and colidx choose an element from that design matrix
        //  * sub_mat is set to that element, and represents an MxM block of the full parity check
        //  * check ranges 0..M and is the row inside that MxM block
        //
        // For each check in 0..M we compute the corresponding column inside that MxM block,
        // either using a rotated identity matrix or using the phi and theta lookups,
        // add the offset to get to this block (rowidx*M, colidx*M), and return the result.

        // Loop over rows of the prototype
        loop {
            // Loop over columns of the prototype
            loop {
                // Loop over the three sub-prototypes we have to sum for each cell of the prototype
                loop {
                    // If we have not yet yielded enough edges for this sub_mat
                    if self.check < self.m {
                        // Weirdly doing this & operation every loop is faster than doing it just
                        // when we update self.sub_mat. Presumably the hint helps it match.
                        match self.sub_mat & (HP | HI) {
                            HI => {
                                // Identity matrix with a right-shift
                                let chk = self.rowidx * self.m + self.check;
                                let var = self.colidx * self.m
                                    + ((self.check + self.sub_mat_val) & self.modm);
                                self.check += 1;
                                return Some((chk, var));
                            }
                            HP => {
                                // Permutation matrix using theta and phi lookup tables
                                let pi = (((THETA_K[self.sub_mat_val] as usize
                                    + (self.check >> self.logmd4))
                                    % 4)
                                    << self.logmd4)
                                    + ((self.phi[self.check >> self.logmd4][self.sub_mat_val]
                                        as usize
                                        + self.check)
                                        & self.modmd4);
                                let chk = self.rowidx * self.m + self.check;
                                let var = self.colidx * self.m + pi;
                                self.check += 1;
                                return Some((chk, var));
                            }
                            _ => (),
                        }
                    }

                    // Once we're done yielding results for this cell, reset check to 0.
                    self.check = 0;

                    // Advance which of the three sub-matrices we're summing.
                    // If sub_mat is 0, there won't be any new ones to sum, so stop then too.
                    if self.sub_mat != 0 && self.sub_mat_idx < 2 {
                        self.sub_mat_idx += 1;
                        self.sub_mat = self.prototype[self.sub_mat_idx][self.rowidx][self.colidx];
                        self.sub_mat_val = (self.sub_mat & 0x3F) as usize;
                    } else {
                        self.sub_mat_idx = 0;
                        break;
                    }
                }

                // Advance colidx. The number of active columns depends on the prototype.
                if self.colidx < 10 {
                    self.colidx += 1;
                    self.sub_mat = self.prototype[self.sub_mat_idx][self.rowidx][self.colidx];
                    self.sub_mat_val = (self.sub_mat & 0x3F) as usize;
                } else {
                    self.colidx = 0;
                    break;
                }
            }

            // Advance rowidx. The number of rows depends on the prototype.
            if self.rowidx < 3 {
                self.rowidx += 1;
                self.sub_mat = self.prototype[self.sub_mat_idx][self.rowidx][self.colidx];
                self.sub_mat_val = (self.sub_mat & 0x3F) as usize;
            } else {
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32_u16(crc: u32, data: u32) -> u32 {
        let mut crc = crc ^ data;
        for _ in 0..16 {
            let mask = if crc & 1 == 0 { 0 } else { 0xFFFFFFFFu32 };
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
        crc
    }

    #[test]
    fn test_iter_parity() {
        test_parity(0, TM1280);
        test_parity(1, TM1536);
        test_parity(2, TM2048);
        test_parity(3, TM5120);
        test_parity(4, TM6144);
        test_parity(5, TM8192);
    }

    fn test_parity<const K: usize, const N: usize>(i: usize, code: ECC<K, N>) {
        // These CRC results have been manually verified and should only change if
        // the ordering of checks returned from the iterator changes.
        let crc_results = [
            0xB643C99E, 0x8169E0CF, 0x599A0807, 0xD0E794B1, 0xBD0AB764, 0x9003014C,
        ];

        let mut count = 0;
        let mut crc = 0xFFFFFFFFu32;
        for (check, var) in code.iter() {
            count += 1;
            crc = crc32_u16(crc, check as u32);
            crc = crc32_u16(crc, var as u32);
        }
        assert_eq!(
            count,
            code.paritycheck_sum(),
            "❌ Invalid parity checksum for {i}"
        );
        assert_eq!(crc, crc_results[i], "❌ Invalid crc for {i}");
    }
}
