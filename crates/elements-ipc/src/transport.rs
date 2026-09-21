//! Stream transport: Unix domain sockets on macOS and Linux, named pipes on
//! Windows. `interprocess` provides both behind one API; the Python client
//! reimplements the same two cases with its standard library.

use std::io::{Read, Write};

use interprocess::TryClone;
use interprocess::local_socket::{
    GenericFilePath, GenericNamespaced, ListenerOptions, Stream as RawStream, ToFsName, ToNsName,
    prelude::*,
};

use crate::ProtocolError;

/// A conventional endpoint for `name`: a socket file on Unix, a pipe on Windows.
pub fn default_endpoint(name: &str) -> String {
    if cfg!(windows) {
        format!("\\\\.\\pipe\\{name}")
    } else {
        let base = std::env::var("XDG_RUNTIME_DIR")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        format!("{base}/{name}.sock")
    }
}

fn to_name(endpoint: &str) -> Result<interprocess::local_socket::Name<'_>, ProtocolError> {
    let name = if cfg!(windows) && !endpoint.contains('\\') {
        endpoint.to_ns_name::<GenericNamespaced>()?
    } else {
        endpoint.to_fs_name::<GenericFilePath>()?
    };
    Ok(name)
}

/// A connected control-plane stream.
pub struct Stream(RawStream);

impl Stream {
    pub fn connect(endpoint: &str) -> Result<Self, ProtocolError> {
        Ok(Self(RawStream::connect(to_name(endpoint)?)?))
    }

    /// A second handle to the same stream, so one side can be wrapped in a
    /// `BufReader` while the other is written to.
    pub fn try_clone(&self) -> Result<Self, ProtocolError> {
        Ok(Self(self.0.try_clone()?))
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

/// A bound listener.
pub struct Listener {
    inner: interprocess::local_socket::Listener,
}

impl Listener {
    pub fn bind(endpoint: &str) -> Result<Self, ProtocolError> {
        // A stale socket file from a crashed daemon would make bind fail.
        if !cfg!(windows) {
            let _ = std::fs::remove_file(endpoint);
        }
        let inner = ListenerOptions::new()
            .name(to_name(endpoint)?)
            .create_sync()?;
        Ok(Self { inner })
    }

    pub fn accept(&self) -> Result<Stream, ProtocolError> {
        Ok(Stream(self.inner.accept()?))
    }
}
