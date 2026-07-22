mod capture;
mod detect;
mod gps;
mod track;
mod types;
use std::sync::Arc;
use std::time::Duration;
use anyhow::{Result, bail};
use crossbeam_channel::unbounded;
use crate::capture::Source;
use crate::types::{DetectionResult, FrameQueue, GpsFix};

const DEFAULT_MODEL: &str = "models/drone_yolov8s.onnx";
const CLASS_NAMES: [&str; 4] = ["airplane", "bird", "drone", "helicopter"];
const DRONE_CLASS_INDEX: usize = 2;
const CONF_THRESH: f32 = 0.35;
const NMS_IOU_THRESH: f32 = 0.;
const TRACK_IOU_THRESH: f32 = 0.3;
const TRACK_MAX_MISSES: u32 = 15;
const TRACK_CONFIRM_HITS: u32 = 3;
const PREDICT_HORIZON_SECS: f32 = 2.0;
const GPS_DEVICE: &str = "/dev/ttyACM0";
const GPS_TIMEOUT: Duration = Duration::from_secs(30);
const GPS_MIN_SATELLITES: u32 = 4;
const PC_CAMERA: (&str, u32, u32, u32, Option<&str>) = ("/dev/video0", 1280, 720, 30, None);
const JETSON_CAMERA: (&str, u32, u32, u32, Option<&str>) = ("/dev/video0", 1920, 1200, 30, Some("UYVY"));

fn print_usage() {
    eprintln!(
        "usage: drone_pipeline [options]

  --source camera         live V4L2 camera (default)
  --source <file.mp4>     video file backtest (processes every frame)
  --source <file.png>     single still image, held
  --jetson-cam            use Arducam UYVY 1920x1200 caps instead of PC webcam
  --model <path>          ONNX model (default: {DEFAULT_MODEL})
  --gps                   acquire position once from {GPS_DEVICE} at startup
  --at <lat>,<lon>        hardcode position, skip the GPS device entirely
                          (use this on the PC, where there's no receiver)
  --profile               print per-stage timings
  --help"
    );
}

fn main() -> Result<()> {
    // Hand-rolled arg parsing; not worth a dependency for seven flags.
    let mut source_arg = String::from("camera");
    let mut model_path = String::from(DEFAULT_MODEL);
    let mut jetson_cam = false;
    let mut enable_gps = false;
    let mut manual_pos: Option<(f64, f64)> = None;
    let mut profile = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                i += 1;
                let Some(v) = args.get(i) else { bail!("--source needs a value") };
                source_arg = v.clone();
            }
            "--model" => {
                i += 1;
                let Some(v) = args.get(i) else { bail!("--model needs a value") };
                model_path = v.clone();
            }
            "--at" => {
                i += 1;
                let Some(v) = args.get(i) else { bail!("--at needs <lat>,<lon>") };
                let Some((a, b)) = v.split_once(',') else { bail!("--at expects <lat>,<lon>") };
                manual_pos = Some((a.trim().parse()?, b.trim().parse()?));
            }
            "--jetson-cam" => jetson_cam = true,
            "--gps" => enable_gps = true,
            "--profile" => profile = true,
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => bail!("unknown argument: {other}")
        }
        i += 1;
    }

    // Get GPS position once before everything else starts. This is a startup constant and lasts
    // the entire lifetime of the process.
    let position: Option<GpsFix> = if let Some((lat, lon)) = manual_pos {
        Some(GpsFix { lat, lon, altitude_m: None, satellites: None })
    } else if enable_gps {
        eprintln!("gps: waiting for fix on {GPS_DEVICE} (up to {GPS_TIMEOUT:?})...");
        gps::acquire(GPS_DEVICE, GPS_TIMEOUT, GPS_MIN_SATELLITES)
    } else {
        None
    };
    match position {
        Some(p) => eprintln!(
            "position: {:.6}, {:.6}{}",
            p.lat,
            p.lon,
            match p.satellites {
                Some(n) => format!(" ({n} satellites)"),
                None => " (manual)".to_string()
            }
        ),
        None => eprintln!("postion: unknown -- detection runs, dashboard has no anchor")
    }

    let source = if source_arg == "camera" {
        let (device, width, height, fps, format) = if jetson_cam { JETSON_CAMERA } else { PC_CAMERA };
        Source::Camera {
            device: device.to_string(),
            width, height, fps,
            format: format.map(|s| s.to_string())
        }
    } else {
        let lower = source_arg.to_lowercase();
        if lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            Source::Image(source_arg.clone())
        } else {
            Source::VideoFile(source_arg.clone())
        }
    };

    // Live feed drops stale frames to bound latency; file sources process every frame so backtests
    // are deterministic and reproducible
    let drop_stale = source.drop_stale();
    eprintln!(
        "source: {source:?}\nmode: {}",
        if drop_stale {
            "live (dropping stale frames)"
        } else {
            "backtest (every frame, backpressured)"
        }
    );
    let queue = Arc::new(FrameQueue::new(drop_stale));
    let cap_queue = Arc::clone(&queue);
    std::thread::spawn(move || {
        if let Err(e) = capture::run_capture(source, Arc::clone(&cap_queue)) {
            eprintln!("capture thread died: {e:?}");
            cap_queue.close();
        }
    });
    
    let (tx, rx) = unbounded::<DetectionResult>();
    std::thread::spawn(move || {
        let use_cuda = cfg!(feature = "cuda");
        let use_trt = cfg!(feature = "tensorrt");
        let mut detector = detect::Detector::new(
            &model_path,
            CONF_THRESH,
            NMS_IOU_THRESH,
            DRONE_CLASS_INDEX,
            use_cuda,
            use_trt,
            profile
        ).expect("failed to build detector");
        let mut tracker = track::Tracker::new(TRACK_IOU_THRESH, TRACK_MAX_MISSES, TRACK_CONFIRM_HITS, DRONE_CLASS_INDEX);

        while let Some(frame) = queue.take() {
            match detector.detect(&frame) {
                Ok(dets) => {
                    let tracks = tracker.update(&dets, frame.captured_at, frame.width as f32, frame.height as f32);
                    let drone_count = tracks.iter().filter(|t| t.is_drone).count();
                    let result = DetectionResult {
                        seq: frame.seq,
                        captured_at: frame.captured_at,
                        total_objects: tracks.len(),
                        drone_count,
                        detections: dets,
                        tracks,
                        frame,
                    };
                    if tx.send(result).is_err() { break; }
                }
                Err(e) => eprintln!("inference error: {e:?}"),
            }
        }
        eprintln!("inference: source exhausted, exiting");
    });

    for result in rx {
        if result.total_objects == 0 {
            continue;
        }
        println!(
            "seq {:>6} | {} object(s), {} drone(s)",
            result.seq, result.total_objects, result.drone_count
        );
        for t in &result.tracks {
            let name = CLASS_NAMES.get(t.class_id).copied().unwrap_or("?");
            let tag = if t.is_drone { "DRONE" } else { "not-drone" };
            let (px, py) = t.predict(PREDICT_HORIZON_SECS);
            println!(
                "    #{:<3} {name:<10} [{tag}] conf={:.2} at=({:.0},{:.0}) \
                 v=({:.0},{:.0})px/s pred@{PREDICT_HORIZON_SECS}s=({px:.0},{py:.0})",
                t.id,
                t.confidence,
                t.center().0,
                t.center().1,
                t.vx,
                t.vy
            );
        }
    }
    
    Ok(())
}
