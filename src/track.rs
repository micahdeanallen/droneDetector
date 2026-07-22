use std::time::Duration;
use crate::types::{Detection, Track};

// Greedy association of current frame's detection to tracks from previous frames, which is used in
// a constant-velocity model to predict the detected object's path
const CLASS_VOTE_WINDOW: usize = 15;
const MAX_SPEED_FRAME_WIDTHS_PER_SEC: f32 = 3.0;
const COAST_REPORT_LIMIT: u32 = 7;
const OFF_FRAME_MARGIN: f32 = 0.25;

pub struct Tracker {
    tracks: Vec<Track>,
    last_update: Option<Duration>,
    next_id: u64,
    iou_thresh: f32,
    centroid_gate: f32,
    max_misses: u32,
    confirm_hits: u32,
    vel_alpha: f32,
    drone_class_index: usize
}
impl Tracker {
    pub fn new(iou_thresh: f32, max_misses: u32, confirm_hits: u32, drone_class_index: usize) -> Self {
        Tracker {
            tracks: Vec::new(),
            last_update: None,
            next_id: 1,
            iou_thresh,
            centroid_gate: 2.0,
            max_misses,
            confirm_hits,
            vel_alpha: 0.35,
            drone_class_index
        }
    }

    // Feed one processed frame. 'now' must be the frame's capture time, not the current instant,
    // so velocities stay correct when frames are dropped.
    pub fn update(&mut self, detections: &[Detection], now: Duration, frame_w: f32, frame_h: f32) -> Vec<Track> {
        let dt = match self.last_update {
            Some(prev) if now > prev => (now - prev).as_secs_f32(),
            _ => 0.0
        };
        self.last_update = Some(now);
        let max_speed = frame_w * MAX_SPEED_FRAME_WIDTHS_PER_SEC;
        let mut det_taken = vec![false; detections.len()];
        let mut track_matched = vec![false; self.tracks.len()];

        // Score every (track, detection) pair and teke them greedily best first.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in detections.iter().enumerate() {
                let score = self.match_score(track, det, dt);
                if score > 0.0 {
                    pairs.push((score, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (_, ti, di) in pairs {
            if track_matched[ti] || det_taken[di] { continue; }
            track_matched[ti] = true;
            det_taken[di] = true;
            let det = &detections[di];
            let drone_idx = self.drone_class_index;
            let track = &mut self.tracks[ti];
            if dt > 0.0 {
                let (old_cx, old_cy) = track.center();
                let (new_cx, new_cy) = det.center();
                let inst_vx = (new_cx - old_cx) / dt;
                let inst_vy = (new_cy - old_cy) / dt;
                if inst_vx.hypot(inst_vy) <= max_speed {
                    let a = self.vel_alpha;
                    track.vx = a * inst_vx + (1.0 - a) * track.vx;
                    track.vy = a * inst_vy + (1.0 - a) * track.vy;
                }
            }
            track.x = det.x;
            track.y = det.y;
            track.w = det.w;
            track.h = det.h;
            track.confidence = det.confidence;
            track.hits += 1;
            track.misses = 0;
            if track.hits >= self.confirm_hits {
                track.confirmed = true;
            }
            track.recent_classes.push(det.class_id);
            if track.recent_classes.len() > CLASS_VOTE_WINDOW {
                track.recent_classes.remove(0);
            }
            track.class_id = majority_class(&track.recent_classes);
            track.is_drone = track.class_id == drone_idx;
        }

        // Unmatched tracks coast forward on their last known velocity, filling potential detection
        // dropouts between frames. This saves the path rather than the detection/path jittering
        // and increasing prediction overhead.
        for (ti, track) in self.tracks.iter_mut().enumerate() {
            if track_matched[ti] { continue; }
            track.misses += 1;
            if dt > 0.0 {
                track.x += track.vx * dt;
                track.y += track.vy * dt;
            }
        }
        self.tracks.retain(|t| t.misses <= self.max_misses);
        let mx = frame_w * OFF_FRAME_MARGIN;
        let my = frame_h * OFF_FRAME_MARGIN;
        self.tracks.retain(|t| {
            let (cx, cy) = t.center();
            cx > -mx && cx < frame_w + mx && cy > -my && cy < frame_h + my
        });
        
        // Unmatch detections become new unconfirmed tracks.
        for (di, det) in detections.iter().enumerate() {
            if det_taken[di] { continue; }
            self.tracks.push(Track {
                id: self.next_id,
                class_id: det.class_id,
                is_drone: det.class_id == self.drone_class_index,
                confidence: det.confidence,
                x: det.x,
                y: det.y,
                w: det.w,
                h: det.h,
                vx: 0.0,
                vy: 0.0,
                hits: 1,
                misses: 0,
                confirmed: self.confirm_hits <= 1,
                recent_classes: vec![det.class_id]
            });
            self.next_id += 1;
        }
        self.tracks.iter().filter(|t| t.confirmed && t.misses < COAST_REPORT_LIMIT).cloned().collect()
    }

    // Returns 0.0 for "no match". IoU when boxes overlap, otherwise a decaying score based on
    // distance from predicted position.
    fn match_score(&self, track: &Track, det: &Detection, dt: f32) -> f32 {
        let i = iou_box((track.x, track.y, track.w, track.h), (det.x, det.y, det.w, det.h));
        if i >= self.iou_thresh { return 1.0 + i; }
        let (px, py) = track.predict(dt);
        let (dx, dy) = det.center();
        let dist = ((px - dx).powi(2) + (py - dy).powi(2)).sqrt();
        let diag = (track.w.powi(2) + track.h.powi(2)).sqrt().max(1.0);
        let gate = diag * self.centroid_gate * (1.0 + track.misses as f32 * 0.4);
        if dist <= gate { 1.0 - dist / gate } else { 0.0 }
    }
}

fn iou_box(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (ax, ay, aw, ah) = a;
    let (bx, by, bw, bh) = b;
    let ix1 = ax.max(bx);
    let iy1 = ay.max(by);
    let ix2 = (ax + aw).min(bx + bw);
    let iy2 = (ay + ah).min(by + bh);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = aw * ah + bw * bh - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn majority_class(recent: &[usize]) -> usize {
    let mut best = recent[0];
    let mut best_count = 0;
    for &candidate in recent {
        let count = recent.iter().filter(|&&c| c == candidate).count();
        if count > best_count {
            best_count = count;
            best = candidate;
        }
    }
    best
}
