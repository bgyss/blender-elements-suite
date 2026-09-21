#![forbid(unsafe_code)]

//! The Elements engine daemon.

mod daemon;

use std::io::{BufReader, Write};

use clap::Parser;
use elements_ipc::{Command, Listener, Response, read_message, write_message};

#[derive(Parser)]
#[command(name = "elementsd", version, about = "Elements Suite engine daemon")]
struct Cli {
    /// Socket path (Unix) or pipe name (Windows) to listen on.
    #[arg(long)]
    endpoint: String,
    /// Path of the memory-mapped frame channel.
    #[arg(long)]
    channel: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let listener = Listener::bind(&cli.endpoint)?;

    // Announce readiness on stdout so supervisors need not poll or sleep.
    println!("ready {}", cli.endpoint);
    std::io::stdout().flush()?;

    loop {
        let stream = listener.accept()?;
        // One client at a time: Core v1 has a single consumer, and serialising
        // sessions keeps the GPU context single-threaded.
        if let Err(e) = serve(stream, &cli.channel) {
            eprintln!("session ended: {e:#}");
        }
    }
}

fn serve(mut stream: elements_ipc::Stream, channel: &std::path::Path) -> anyhow::Result<()> {
    let mut session = match daemon::Session::new(channel) {
        Ok(s) => s,
        Err(e) => {
            write_message(&mut stream, &Response::Error(e))?;
            return Ok(());
        }
    };

    let mut reader = BufReader::new(stream.try_clone()?);
    // A `ProtocolError::OversizedFrame` from `read_message` leaves the stream
    // mid-line; its own doc comment requires the caller to stop reading and
    // close the connection rather than resume, since the next read would
    // start at the wrong byte. Propagating the error out of `serve` does
    // exactly that: the stream (and its cloned handle) are dropped when this
    // function returns, and the outer loop moves on to the next `accept`.
    while let Some(command) = read_message::<_, Command>(&mut reader)? {
        let shutting_down = matches!(command, Command::Shutdown);
        let response = daemon::handle(&mut session, command);
        write_message(&mut stream, &response)?;
        if shutting_down {
            break;
        }
    }
    Ok(())
}
