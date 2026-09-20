//! The data plane: a memory-mapped file holding two alternating frame buffers.
//!
//! A file rather than POSIX shared memory because the Blender addon must read
//! it with the Python standard library on 3.11 through 3.13, where
//! `SharedMemory` cleanup semantics differ. The double-buffer protocol is
//! identical, so this can be swapped for real shared memory later without
//! changing either end's API.
//!
//! Protocol: the writer fills buffer `seq % 2`, then increments `seq`. A reader
//! samples `seq`, reads buffer `(seq - 1) % 2`, and re-samples `seq`; if it
//! changed, the writer lapped it and the read is retried.
//!
//! # Soundness caveat
//!
//! This is a seqlock, and a seqlock is not fully soundly expressible in safe
//! Rust. The buffer copies in [`FrameWriter::publish`] and
//! [`FrameReader::read_latest`] are ordinary, non-atomic accesses to the same
//! memory, made through safe indexing APIs. When the writer begins
//! overwriting the exact buffer a reader is mid-copy of, both threads are
//! touching overlapping memory without synchronization on those bytes
//! themselves — only `seq` is atomic. Under the strict Rust/C++ abstract
//! memory model, that is a data race, which is formally undefined behavior,
//! even though the sequence-number recheck afterwards always detects and
//! discards any torn read: no incorrect data is ever returned to a caller.
//! This is why the design cannot be expressed with only safe accesses to the
//! buffer bytes; it works in practice on real hardware and compilers and is a
//! well-known pattern in production systems code, but it is a residual,
//! model-level unsoundness inherent to the seqlock design itself, not a bug
//! in this implementation. The alternative, a lock around the buffer, was
//! rejected: the writer must never block on a slow reader, and a lock would
//! make that possible.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// `"ELFC"` as a little-endian u32.
pub const CHANNEL_MAGIC: u32 = 0x4346_4C45;
pub const CHANNEL_VERSION: u32 = 1;
/// Fixed header size. Buffers start here; the tail is reserved padding.
pub const CHANNEL_HEADER_BYTES: usize = 64;

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 4;
const OFF_DIMS: usize = 8; // three u32
const OFF_CHANNELS: usize = 20;
const OFF_BUFFER_BYTES: usize = 24; // u64
const OFF_SEQ: usize = 32; // u64, must be 8-byte aligned for atomic access

#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("not an Elements frame channel (magic {found:#010x})")]
    BadMagic { found: u32 },
    #[error("unsupported channel version {0}")]
    UnsupportedVersion(u32),
    #[error("expected {expected} values, got {got}")]
    LengthMismatch { expected: usize, got: usize },
    #[error("the writer lapped the reader repeatedly")]
    Torn,
}

/// The fixed-size channel header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelHeader {
    pub magic: u32,
    pub version: u32,
    pub dims: [u32; 3],
    pub channels: u32,
    pub buffer_bytes: u64,
    pub seq: u64,
}

fn value_count(dims: [u32; 3], channels: u32) -> usize {
    dims[0] as usize * dims[1] as usize * dims[2] as usize * channels as usize
}

fn read_u32(map: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(map[offset..offset + 4].try_into().expect("in-bounds"))
}

fn read_u64(map: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(map[offset..offset + 8].try_into().expect("in-bounds"))
}

/// Read the `seq` field with acquire ordering.
///
/// # Safety
/// `map` must be at least `OFF_SEQ + 8` bytes long. Its base address must be
/// 8-byte aligned, which holds because `memmap2` mappings begin at a
/// page-aligned address (page sizes are always multiples of 8) and `OFF_SEQ`
/// (32) is itself a multiple of 8, so `map.as_ptr().add(OFF_SEQ)` is 8-byte
/// aligned. No `&mut` reference to these bytes may be alive concurrently with
/// this read (readers only ever take `&self`).
unsafe fn load_seq(map: &[u8]) -> u64 {
    let ptr = map.as_ptr().wrapping_add(OFF_SEQ) as *const AtomicU64;
    unsafe { (*ptr).load(Ordering::Acquire) }
}

/// Store the `seq` field with release ordering, publishing prior buffer writes.
///
/// # Safety
/// Same alignment/length requirements as [`load_seq`]. Additionally, the
/// caller must be the sole writer of this field (enforced by `FrameWriter`
/// owning the only `MmapMut` for the file), so this store never races another
/// store.
unsafe fn store_seq(map: &mut [u8], value: u64) {
    let ptr = map.as_mut_ptr().wrapping_add(OFF_SEQ) as *const AtomicU64;
    unsafe { (*ptr).store(value, Ordering::Release) }
}

/// The engine side: owns the file and publishes frames.
pub struct FrameWriter {
    map: memmap2::MmapMut,
    path: PathBuf,
    dims: [u32; 3],
    channels: u32,
    values: usize,
    buffer_bytes: usize,
    seq: u64,
}

impl FrameWriter {
    /// Create (or truncate) the channel file and write its header.
    pub fn create(path: &Path, dims: [u32; 3], channels: u32) -> Result<Self, ChannelError> {
        let values = value_count(dims, channels);
        let buffer_bytes = values * std::mem::size_of::<f32>();
        let total = CHANNEL_HEADER_BYTES + 2 * buffer_bytes;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total as u64)?;

        // SAFETY: the file was just sized to `total` bytes above, so the
        // mapping cannot read past the end of the underlying file (the usual
        // mmap hazard of a file being truncated by another process
        // afterwards is out of scope: this file is private to this channel
        // and nothing else in this codebase resizes it). No other mapping of
        // this file exists yet *from this call*, at this instant — a
        // `FrameReader::open` racing this `create` (deliberately exercised by
        // the concurrency test) will map the same file concurrently once this
        // call returns, which is expected and is exactly what the seqlock
        // protocol above is designed to make safe.
        let mut map = unsafe { memmap2::MmapMut::map_mut(&file)? };

        map[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(&CHANNEL_MAGIC.to_le_bytes());
        map[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&CHANNEL_VERSION.to_le_bytes());
        for (i, d) in dims.iter().enumerate() {
            let at = OFF_DIMS + i * 4;
            map[at..at + 4].copy_from_slice(&d.to_le_bytes());
        }
        map[OFF_CHANNELS..OFF_CHANNELS + 4].copy_from_slice(&channels.to_le_bytes());
        map[OFF_BUFFER_BYTES..OFF_BUFFER_BYTES + 8]
            .copy_from_slice(&(buffer_bytes as u64).to_le_bytes());
        // SAFETY: see `store_seq`; `map` is at least `CHANNEL_HEADER_BYTES`
        // (64) long because `total >= CHANNEL_HEADER_BYTES`, and this
        // `FrameWriter` is the only writer of the file.
        unsafe { store_seq(&mut map, 0) };
        map.flush()?;

        Ok(Self {
            map,
            path: path.to_path_buf(),
            dims,
            channels,
            values,
            buffer_bytes,
            seq: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Write `values` into the next buffer and publish it. Returns the new sequence.
    pub fn publish(&mut self, values: &[f32]) -> Result<u64, ChannelError> {
        if values.len() != self.values {
            return Err(ChannelError::LengthMismatch {
                expected: self.values,
                got: values.len(),
            });
        }

        let index = (self.seq % 2) as usize;
        let start = CHANNEL_HEADER_BYTES + index * self.buffer_bytes;
        let bytes: &[u8] = bytemuck::cast_slice(values);
        self.map[start..start + self.buffer_bytes].copy_from_slice(bytes);

        self.seq += 1;
        // SAFETY: see `store_seq`. `self.map` is at least `CHANNEL_HEADER_BYTES`
        // long (checked once in `create`, and the mapping's length never
        // changes), and `self` is the only `FrameWriter` for this file.
        unsafe { store_seq(&mut self.map, self.seq) };
        Ok(self.seq)
    }
}

/// The client side: maps the file read-only and samples the latest frame.
#[derive(Debug)]
pub struct FrameReader {
    map: memmap2::Mmap,
    header: ChannelHeader,
    values: usize,
    buffer_bytes: usize,
}

impl FrameReader {
    pub fn open(path: &Path) -> Result<Self, ChannelError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: a concurrent writer may modify these bytes through its own
        // mapping of the same file, which is the point of this channel. It is
        // not only `seq`'s atomic access and the creation of this mapping
        // that are unsafe here: the buffer copies in `read_latest` are also
        // concurrently-racing accesses with the writer's buffer copies in
        // `publish`, even though they go through safe indexing APIs — see the
        // "Soundness caveat" section of the module docs. `load_seq` (an
        // atomic acquire load) and `read_latest`'s seqlock retry validate the
        // sequence number before and after copying out a buffer, so no
        // reader ever returns a partially-written value to its caller as if
        // it were complete, but the copy itself is not race-free under the
        // abstract memory model.
        let map = unsafe { memmap2::Mmap::map(&file)? };

        if map.len() < CHANNEL_HEADER_BYTES {
            return Err(ChannelError::BadMagic { found: 0 });
        }
        let magic = read_u32(&map, OFF_MAGIC);
        if magic != CHANNEL_MAGIC {
            return Err(ChannelError::BadMagic { found: magic });
        }
        let version = read_u32(&map, OFF_VERSION);
        if version != CHANNEL_VERSION {
            return Err(ChannelError::UnsupportedVersion(version));
        }

        let dims = [
            read_u32(&map, OFF_DIMS),
            read_u32(&map, OFF_DIMS + 4),
            read_u32(&map, OFF_DIMS + 8),
        ];
        let channels = read_u32(&map, OFF_CHANNELS);
        let buffer_bytes = read_u64(&map, OFF_BUFFER_BYTES) as usize;
        let values = value_count(dims, channels);

        // `buffer_bytes` was just read from the file, which this crate did
        // not necessarily write (a crashed process may have left a truncated
        // file mid-write, or a test/fixture may hand-build one with a wrong
        // size). It must be validated against the real mapping length before
        // it is ever used to slice into `map`, or a malformed file turns
        // into an out-of-bounds panic in `read_latest` instead of an error.
        let expected = CHANNEL_HEADER_BYTES + 2 * buffer_bytes;
        if map.len() < expected {
            return Err(ChannelError::LengthMismatch {
                expected,
                got: map.len(),
            });
        }

        Ok(Self {
            header: ChannelHeader {
                magic,
                version,
                dims,
                channels,
                buffer_bytes: buffer_bytes as u64,
                seq: 0,
            },
            map,
            values,
            buffer_bytes,
        })
    }

    pub fn header(&self) -> ChannelHeader {
        // SAFETY: see `load_seq`; `self.map` was validated to be at least
        // `CHANNEL_HEADER_BYTES` long in `open`, and that length is fixed for
        // the lifetime of the mapping.
        let seq = unsafe { load_seq(&self.map) };
        ChannelHeader { seq, ..self.header }
    }

    /// Copy the most recently published frame.
    ///
    /// Returns sequence 0 and zeros if nothing has been published yet.
    pub fn read_latest(&self) -> Result<(u64, Vec<f32>), ChannelError> {
        for _ in 0..3 {
            // SAFETY: see `load_seq`; length/alignment established in `open`.
            let before = unsafe { load_seq(&self.map) };
            let index = if before == 0 {
                0
            } else {
                ((before - 1) % 2) as usize
            };
            let start = CHANNEL_HEADER_BYTES + index * self.buffer_bytes;

            let mut out = vec![0.0f32; self.values];
            bytemuck::cast_slice_mut::<f32, u8>(&mut out)
                .copy_from_slice(&self.map[start..start + self.buffer_bytes]);

            // SAFETY: see `load_seq`; same reasoning as `before` above.
            let after = unsafe { load_seq(&self.map) };
            if after == before {
                return Ok((before, out));
            }
        }
        Err(ChannelError::Torn)
    }
}
