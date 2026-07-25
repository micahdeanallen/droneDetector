use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use crate::types::{DetectionResult, GpsFix};

// Lightweight telementry payload handed from the display loop to the sender thread. Pre-extracted
// from DetectionResult so the sender never touches actual frame buffers
pub struct Telementry {
    pub total_objects: usize,
    pub drone_count: usize,
    // one tuple per object: (id, is_drone, px, py, vx, vy) where p is position and v is velocity
    pub objects: Vec<(u64, bool, f32, f32, f32, f32)>
}

// Handle held by the display loop. Dropping telementry when the channel is full is
// intentional: detection/display must never block on a slow or absent dashboard
pub struct TelementrySink {
    tx: Sender<Telementry>
}
impl TelementrySink {
    // Best effort send. Returns silently on a full or disconnected channel
    pub fn offer(&self, t: Telementry) {
        match self.tx.try_send(t) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}            // dashboard behind; drop frame
            Err(TrySendError::Disconnected(_)) => {}    // sender thread gone; ignore
        }
    }
}

// Build a Telementry from a DetectionResult without cloning the frame
pub fn extract(result: &DetectionResult) -> Telementry {
    let mut objects = Vec::with_capacity(result.tracks.len());
    for t in &result.tracks {
        let (cx, cy) = t.center();
        objects.push((t.id, t.is_drone, cx, cy, t.vx, t.vy));
    }
    Telementry {
        total_objects: result.total_objects,
        drone_count: result.drone_count,
        objects
    }
}

// Spawn the telementry server. Binds to the LAN IP of the Jetson, not 0.0.0.0. Accepts ONE client
// and streams newline-delimited JSON. The header line is sent once per client connection
// so late/reconnecting dashboards will still get the anchor. Returns a sink for the display loop;
// all socket I/O happens on the spawned thread
pub fn spawn(addr: &str, anchor: GpsFix, frame_w: u32, frame_h: u32, cap: usize) -> std::io::Result<TelementrySink> {
    let listener = TcpListener::bind(addr)?;
    eprintln!("dashboard: listening on {addr} (single client, others refused)");

    let (tx, rx) = bounded::<Telementry>(cap);
    let rx = Arc::new(rx);
    let header = Arc::new(format!(
        "{{\"lat\":{:.6},\"lon\":{:.6},\"w\":{},\"h\":{}}}\n",
        anchor.lat, anchor.lon, frame_w, frame_h
    ));
    let busy = Arc::new(AtomicBool::new(false));

    std::thread::spawn(move || {
        for incoming in listener.incoming() {
            let mut stream = match incoming {
                Ok(s) => s,
                Err(e) => { eprintln!("dashboard: accept error: {e}"); continue; }
            };
            let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "?".into());

            // Claim the single connection slot. swap returns previous value. If it was already
            // true, someone else holds the connection, so the requested connection is refused
            if busy.swap(true, Ordering::SeqCst) {
                eprintln!("dashboard: refused second client({peer})");
                let _ = stream.write_all(b"{\"error\":\"busy\"}\n");
                continue;
            }

            eprintln!("dashboard: client connected({peer})");
            let rx = Arc::clone(&rx);
            let header = Arc::clone(&header);
            let busy_release = Arc::clone(&busy);

            // Serve on its own thread so the accept loop keeps running and can refuse concurrent
            // connections immediately.
            std::thread::spawn(move || {
                if let Err(e) = serve_client(&mut stream, &header, &rx) {
                    eprintln!("dashboard: client {peer} disconnected ({e})");
                }
                // Drain telementry queued during this session so the next client starts clean
                while rx.try_recv().is_ok() {}
                // Release the slot: the next connection attempt will now be accepted
                busy_release.store(false, Ordering::SeqCst);
            });
        }
    });
    
    Ok(TelementrySink { tx })
}

// Serve a single connected client until a write fails (disconnect)
fn serve_client(stream: &mut TcpStream, header: &str, rx: &crossbeam_channel::Receiver<Telementry>) -> 
std::io::Result<()> {
    // Small writes, send imediately rather than coalescing (telementry is latency sensitive)
    let _ = stream.set_nodelay(true);
    stream.write_all(header.as_bytes())?;

    // Block on the channel; each recv is one frame's worth of telementry
    for t in rx.iter() {
        let mut line = String::with_capacity(32 + t.objects.len() * 32);
        line.push_str(&format!("{{\"n\":{},\"d\":{},\"o\":[", t.total_objects, t.drone_count));
        for (i, (id, drone, px, py, vx, vy)) in t.objects.iter().enumerate() {
            if i > 0 { line.push(','); }
            line.push_str(&format!(
                "[{},{},{:.0},{:.0},{:.0},{:.0}]",
                id, if *drone { 1 } else { 0 }, px, py, vx, vy
            ));
        }
        line.push_str("]}\n");
        stream.write_all(line.as_bytes())?;
    }
    Ok(())
}
