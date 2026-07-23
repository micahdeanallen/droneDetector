use std::sync::Arc;
use anyhow::{anyhow, Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use crate::types::{Frame, FrameQueue};

#[derive(Clone, Debug)]
pub enum Source {
    // Live V4L2 camera that works on both PC webcam and Arducam AR0234
    Camera {
        device: String,
        width: u32,
        height: u32,
        fps: u32,
        format: Option<String> // UYVY for Arducam AR0234; None on PC webcam
    },
    VideoFile(String),
    Image(String)
}
impl Source {
    // Live sources drop stale frames, file sources must not.
    pub fn drop_stale(&self) -> bool {
        matches!(self, Source::Camera { .. })
    }
}

// Pipeline capture loop. Builds GStreamer pipeline from description string (Jetson, laptop, etc),
// pulls samples out of its appsink, and pushes each frame into the FrameQueue. Requests
// RGBA from caps filter so there isn't row-stride padding, stripping alpha to RGB.
fn pipeline_for(source: &Source) -> String {
    match source {
        Source::Camera { device, width, height, fps, format } => {
            let caps = match format {
                Some(f) => format!("video/x-raw,format={f},width={width},height={height},framerate={fps}/1"),
                None => format!("video/x-raw,width={width},height={height},framerate={fps}/1")
            };
            format!(
                "v4l2src device={device} ! {caps} \
                ! videoconvert ! video/x-raw,format=RGBA \
                ! appsink name=sink drop=true max-buffers=1 sync=false"
            )
        }
        Source::VideoFile(path) => format!(
            "filesrc location=\"{path}\" ! decodebin ! videoconvert \
            ! video/x-raw,format=RGBA \
            ! appsink name=sink drop=false max-buffers=4 sync=false"
        ),
        Source::Image(path) => format!(
            "filesrc location=\"{path}\" ! decodebin ! imagefreeze ! videoconvert \
            ! video/x-raw,format=RGBA \
            ! appsink name=sink drop=true max-buffers=1 sync=false"
        )
    }
}

pub fn run_capture(source: Source, queue: Arc<FrameQueue>) -> Result<()> {
    gst::init().context("gstreamer init failed")?;

    let desc = pipeline_for(&source);
    eprintln!("capture: {desc}");
    
    let pipeline = gst::parse::launch(&desc)
        .context("failed to parse gstreamer pipeline")?
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow!("parsed element was not a pipeline"))?;
    
    let appsink = pipeline
        .by_name("sink")
        .context("no element named 'sink' in pipeline")?
        .downcast::<AppSink>()
        .map_err(|_| anyhow!("'sink' element is not an appsink"))?;
    appsink.set_property("async", false);
    
    pipeline.set_state(gst::State::Playing).context("failed to start pipeline (is the device present?)")?;

    let started = std::time::Instant::now();
    let mut seq: u64 = 0;
    let mut waited_ms: u64 = 0;
    loop {
        // Allows the system to sleep until frame is available; Err or EOS or teardown
        let sample = match appsink.try_pull_sample(gst::ClockTime::from_mseconds(100)) {
            Some(sample) => {
                waited_ms = 0;
                sample
            }
            None => {
                if appsink.is_eos() {
                    break;
                }
                waited_ms += 100;
                if waited_ms >= 15_000 {
                    return Err(anyhow!("no frames from source within 15s (check dmesg / reseat CAM1 ribbon)"));
                }
                continue;
            }
        };

        let caps = sample.caps().context("sample had no captures")?;
        let s = caps.structure(0).context("captures had no structure")?;
        let width: i32 = s.get("width").context("captures missing width")?;
        let height: i32 = s.get("height").context("captures missing height")?;
        let (width, height) = (width as usize, height as usize);
        let buffer = sample.buffer().context("sample had no buffer")?;
        let captured_at = match buffer.pts() {
            Some(pts) => std::time::Duration::from_nanos(pts.nseconds()),
            None => started.elapsed()
        };
        let map = buffer.map_readable().context("failed to map buffer")?;
        
        let rgba = map.as_slice();
        let px = width * height;
        if rgba.len() < px * 4 {
            eprintln!("capture: short buffer at seq {seq}, skipping");
            continue;
        }

        // RGBA -> RGB
        let mut rgb = Vec::with_capacity(px * 3);
        for i in 0..px {
            let o = i * 4;
            rgb.push(rgba[o]);
            rgb.push(rgba[o + 1]);
            rgb.push(rgba[o + 2]);
        }
        queue.put(Frame { width, height, data: rgb, seq, captured_at });
        seq += 1;
    }

    let _ = pipeline.set_state(gst::State::Null);
    queue.close();
    Ok(())
}
