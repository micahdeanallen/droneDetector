use std::sync::{Condvar, Mutex};
use std::time::Duration;

// One captured frame; tightly packed RGB, row-major, no stride padding
#[derive(Clone)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    pub seq: u64, // index of frame in video; ex. frame 75 of the video will have seq == 75.
    pub captured_at: Duration
}

// Single object detected by the model, mapped to its pixel coordinates in the frame
#[derive(Clone, Debug)]
pub struct Detection {
    pub class_id: usize,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub is_drone: bool
}
impl Detection {
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
}

// A detection with identity maintained across frames for the dashboard to draw
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub class_id: usize,
    pub is_drone: bool,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub vx: f32,
    pub vy: f32,
    pub hits: u32,
    pub misses: u32,
    pub confirmed: bool,
    pub recent_classes: Vec<usize>
}
impl Track {
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    // Constant velocity extrapolation 'secs' into the future.
    pub fn predict(&self, secs: f32) -> (f32, f32) {
        let (cx, cy) = self.center();
        (cx + self.vx * secs, cy + self.vy * secs)
    }

    // Sampled predicted path for dashboard to draw as trailing line
    pub fn predicted_path(&self, horizon_secs: f32, steps: usize) -> Vec<(f32, f32)> {
        let mut path = Vec::with_capacity(steps);
        for i in 1..=steps {
            let t = horizon_secs * (i as f32 / steps as f32);
            path.push(self.predict(t));
        }
        path
    }
}

// Location given by GPS sensor
#[derive(Clone, Copy, Debug)]
pub struct GpsFix {
    pub lat: f64,
    pub lon: f64,
    pub altitude_m: Option<f64>,
    pub satellites: Option<u32>
}

// All necessary components for dashboard. Frame rides along so footage can be saved.
pub struct DetectionResult {
    pub seq: u64,
    pub captured_at: Duration,
    pub detections: Vec<Detection>,
    pub tracks: Vec<Track>,
    pub total_objects: usize,
    pub drone_count: usize,
    pub frame: Frame
}

// Single-slot frame handoff between capture and inference to minimize backlog of stale frames
pub struct FrameQueue {
    slot: Mutex<Option<Frame>>,
    closed: Mutex<bool>,
    cv_filled: Condvar,
    cv_drained: Condvar,
    drop_stale: bool
}
impl FrameQueue {
    pub fn new(drop_stale: bool) -> Self {
        FrameQueue {
            slot: Mutex::new(None),
            closed: Mutex::new(false),
            cv_filled: Condvar::new(),
            cv_drained: Condvar::new(),
            drop_stale
        }
    }

    pub fn put(&self, frame: Frame) {
        let mut guard = self.slot.lock().unwrap();
        if !self.drop_stale {
            // Backpressure: wait for the consumer to drain the slot
            while guard.is_some() {
                guard = self.cv_drained.wait(guard).unwrap();
            }
        }
        *guard = Some(frame);
        self.cv_filled.notify_one();
    }

    // Blocks until a frame is available. Returns 'None' once the source has ended and the slot is
    // empty, allowing for a clean exit
    pub fn take(&self) -> Option<Frame> {
        let mut guard = self.slot.lock().unwrap();
        loop {
            if let Some(frame) = guard.take() {
                self.cv_drained.notify_one();
                return Some(frame);
            }
            if *self.closed.lock().unwrap() { return None; }
            guard = self.cv_filled.wait(guard).unwrap();
        }
    }

    // Signal that no more frames are coming
    pub fn close(&self) {
        *self.closed.lock().unwrap() = true;
        self.cv_filled.notify_all();
        self.cv_drained.notify_all();
    }
}
