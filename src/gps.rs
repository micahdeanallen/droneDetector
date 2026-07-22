use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::mpsc;
use std::time::Duration;
use crate::types::GpsFix;

// Block until a tangible ping arrives and account for potential inconsistencies in the first few
// pings.
pub fn acquire(device: &str, timeout: Duration, min_satellites: u32) -> Option<GpsFix> {
    let (tx, rx) = mpsc::channel::<GpsFix>();
    let device_owned = device.to_string();

    // Runs on its own thread with no way to interrupt it. Harmless at startup, but don't call again on retry loop.
    std::thread::spawn(move || {
        let file = match File::open(&device_owned) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("gps: cannot open {device_owned}: {e}");
                return;
            }
        };
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let Ok(line) = line else { continue };
            let line = line.trim();
            if !line.starts_with('$') || !checksum_ok(line) {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            if f.is_empty() { continue; }

            // Prefix matching & wait for valid log
            let kind = &f[0][f[0].len().saturating_sub(3)..];
            if kind == "GGA" && f.len() > 9 {
                if f[6] == "0" { continue; }
                let satellites = f[7].parse::<u32>().unwrap_or(0);
                if satellites < min_satellites { continue; }
                let Some((lat, lon)) = parse_lat_lon(&f, 2, 3, 4, 5) else { continue; };
                let _ = tx.send(GpsFix {
                    lat,
                    lon,
                    altitude_m: f[9].parse::<f64>().ok(),
                    satellites: Some(satellites),
                });
                return;
            }
        }
        eprintln!("gps: stream ended before a usable ping");
    });

    match rx.recv_timeout(timeout) {
        Ok(fix) => Some(fix),
        Err(_) => {
            eprintln!("gps: no ping with >= {min_satellites} satellites within {timeout:?}");
            None
        }
    }
}

// Parse latitude & longitude in valid format
fn parse_lat_lon(f: &[&str], lat_i: usize, ns_i: usize, lon_i: usize, ew_i: usize) -> Option<(f64, f64)> {
    let lat = nmea_to_degrees(f.get(lat_i)?, 2)?;
    let lon = nmea_to_degrees(f.get(lon_i)?, 3)?;
    let lat = if *f.get(ns_i)? == "S" { -lat } else { lat };
    let lon = if *f.get(ew_i)? == "W" { -lon } else { lon };
    Some((lat, lon))
}
fn nmea_to_degrees(raw: &str, deg_digits: usize) -> Option<f64> {
    if raw.len() < deg_digits + 1 {
        return None;
    }
    let deg: f64 = raw[..deg_digits].parse().ok()?;
    let min: f64 = raw[deg_digits..].parse().ok()?;
    Some(deg + min / 60.0)
}

// NMEA checksum: XOR of everything between '$' and '*', hex after '*'.
fn checksum_ok(line: &str) -> bool {
    let Some(star) = line.rfind('*') else { return false; };
    let Some(body) = line.get(1..star) else { return false; };
    let Some(given) = line.get(star + 1..star + 3) else { return false; };
    let Ok(given) = u8::from_str_radix(given, 16) else { return false; };
    let computed = body.bytes().fold(0u8, |acc, b| acc ^ b);
    computed == given
}
