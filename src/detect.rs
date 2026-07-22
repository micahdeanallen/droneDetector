use std::time::Instant;
use anyhow::{Context, Result, anyhow};
use image::{imageops::FilterType, RgbImage};
use ndarray::{Array4, Ix3};
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, TensorRTExecutionProvider};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use crate::types::{Detection, Frame};

// Detection pipeline that owns ONNX runtime session and runs one frame end to end:
// letterbox -> NCHW f32 tensor -> ort inference -> decode -> NMS -> Detections
const INPUT_SIZE: usize = 960;
const PAD_VALUE: u8 = 114;
const INPUT_NAME: &str = "images";
const OUTPUT_NAME: &str = "output0";

pub struct Detector {
    session: Session,
    conf_thresh: f32,
    iou_thresh: f32,
    drone_class_index: usize,
    pub profile: bool
}
impl Detector {
    pub fn new(
        model_path: &str, 
        conf_thresh: f32, 
        iou_thresh: f32, 
        drone_class_index: usize, 
        use_cuda: bool, 
        use_trt: bool,
        profile: bool
    ) -> Result<Self> {
        let mut providers = Vec::new();
        if use_trt {
            providers.push(TensorRTExecutionProvider::default().build());
        }
        if use_cuda {
            providers.push(CUDAExecutionProvider::default().build());
        }
        providers.push(CPUExecutionProvider::default().build());
        let builder = Session::builder()
            .map_err(|e| anyhow!("failed to create session builder: {e}"))?;
        let builder = builder
            .with_execution_providers(providers)
            .map_err(|e| anyhow!("failed to register execution providers: {e}"))?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("failed to set optimization level: {e}"))?;
        let mut builder = builder
            .with_intra_threads(4)
            .map_err(|e| anyhow!("failed to set intra threads: {e}"))?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("failed to load model at {model_path}: {e}"))?;
        Ok(Detector { session, conf_thresh, iou_thresh, drone_class_index, profile }) 
    }
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let t0 = Instant::now();
        // letterbox to INPUT_SIZE/build NCHW tensor
        let src = RgbImage::from_raw(frame.width as u32, frame.height as u32, frame.data.clone())
            .context("frame data did not match width*height*3")?;
        let scale = (INPUT_SIZE as f32 / frame.width as f32).min(INPUT_SIZE as f32 / frame.height as f32);
        let new_w = (frame.width as f32 * scale).round() as u32;
        let new_h = (frame.height as f32 * scale).round() as u32;
        let pad_x = (INPUT_SIZE as u32 - new_w) / 2;
        let pad_y = (INPUT_SIZE as u32 - new_h) / 2;
        let resized = image::imageops::resize(&src, new_w, new_h, FilterType::Triangle);
        let mut input = Array4::<f32>::from_elem((1, 3, INPUT_SIZE, INPUT_SIZE), PAD_VALUE as f32 / 255.0);
        for y in 0..new_h {
            for x in 0..new_w {
                let p = resized.get_pixel(x, y);
                let (dx, dy) = ((x + pad_x) as usize, (y + pad_y) as usize);
                input[[0, 0, dy, dx]] = p[0] as f32 / 255.0;
                input[[0, 1, dy, dx]] = p[1] as f32 / 255.0;
                input[[0, 2, dy, dx]] = p[2] as f32 / 255.0;
            }
        }
        let t_pre = t0.elapsed();

        // ort inference
        let t1 = Instant::now();
        let input_ref = TensorRef::from_array_view(&input)?;
        let outputs = self.session.run(ort::inputs![INPUT_NAME => input_ref]).context("inference failed")?;
        let t_infer = t1.elapsed();
        let t2 = Instant::now();
        let out = outputs[OUTPUT_NAME].try_extract_array::<f32>().context("failed to extract output tensor")?;
        let out = out.view().into_dimensionality::<Ix3>().context("unexpected output rank")?;
        let num_classes = out.shape()[1] - 4;
        let num_anchors = out.shape()[2];

        // decode & back-project to original frame coordinates
        let mut dets: Vec<Detection> = Vec::new();
        for a in 0..num_anchors {
            let mut best_id = 0usize;
            let mut best_score = 0.0f32;
            for k in 0..num_classes {
                let score = out[[0, 4 + k, a]];
                if score > best_score {
                    best_score = score;
                    best_id = k;
                }
            }
            if best_score < self.conf_thresh { continue; }
            let cx = out[[0, 0, a]];
            let cy = out[[0, 1, a]];
            let bw = out[[0, 2, a]];
            let bh = out[[0, 3, a]];
            let x = (cx - bw / 2.0 - pad_x as f32) / scale;
            let y = (cy - bh / 2.0 - pad_y as f32) / scale;
            let w = bw / scale;
            let h = bh / scale;
            dets.push(Detection { 
                class_id: best_id, 
                confidence: best_score, 
                x, y, w, h, 
                is_drone: best_id == self.drone_class_index
            });
        }
        nms(&mut dets, self.iou_thresh);
        let t_post = t2.elapsed();
        if self.profile {
            eprintln!(
                "profile: pre {:>7.2}ms infer {:>7.2}ms post {:>7.2}ms",
                t_pre.as_secs_f32() * 1000.0,
                t_infer.as_secs_f32() * 1000.0,
                t_post.as_secs_f32() * 1000.0
            );
        }
        Ok(dets)
    }
}

// Greedy per-class non-max suppression. Runs per-class so a real second object in a different
// class that is near the first object isn't wrongfully dropped
fn nms(dets: &mut Vec<Detection>, iou_thresh: f32) {
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep = vec![true; dets.len()];
    for i in 0..dets.len() {
        if !keep[i] { continue; }
        for j in (i + 1)..dets.len() {
            if !keep[j] { continue; }
            if iou(&dets[i], &dets[j]) > iou_thresh {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    dets.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}
fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = (a.x + a.w).min(b.x + b.w);
    let iy2 = (a.y + a.h).min(b.y + b.h);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}
