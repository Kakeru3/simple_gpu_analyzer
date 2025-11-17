use std::sync::Arc;
use egui_wgpu::wgpu;
use parking_lot::Mutex;
use once_cell::sync::OnceCell;

use nih_plug::editor::Editor;
use nih_plug::prelude::ParamSetter;
use nih_plug_egui::{create_egui_editor, egui};

use egui_wgpu::Callback; // your Callback::new_paint_callback API
use egui_wgpu::RenderState; // provided via types TypeMap in paint callback
use egui_wgpu::wgpu::util::DeviceExt;

use bytemuck::{Pod, Zeroable};

/// Vertex (x,y) in NDC
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Cached pipeline & resources per device/format
struct PipelineResources {
    pipeline: wgpu::RenderPipeline,
}

/// create(editor) expected by your plugin code
pub fn create(fft_data: Arc<Mutex<Vec<f32>>>) -> Option<Box<dyn Editor>> {
    // UI state carries shared FFT data
        struct UiState {
        fft: Arc<Mutex<Vec<f32>>>,
        prev_fft: Vec<f32>,
        min_db: f32,
        max_db: f32,
        // Full-scale reference tracked internally via EMA (no GUI control)
        full_scale_ref: f32,
            // Smoothing for the spectral display (number of bins in moving average).
            // 1 => no smoothing, 3 => small smoothing, etc.
            smoothing_bins: usize,
            // If true, choose the nearest FFT bin per pixel instead of linear
            // interpolation between bins. This makes the plot step-like.
            // remove nearest bin interpolation to avoid boxy 90-degree verticals
    }

    impl UiState {
        fn new(fft: Arc<Mutex<Vec<f32>>>) -> Self {
            // initialize prev_fft from current buffer (or empty)
            let initial = fft.lock().clone();
            // default: max_db fixed at +5 dB, min_db default -60 dB
            // initialize full_scale_ref from the current buffer peak (fallback to 1e-6)
            let init_peak = initial.iter().map(|v| v.abs()).fold(1e-6_f32, f32::max);
            Self { fft, prev_fft: initial, min_db: -60.0, max_db: 5.0, full_scale_ref: init_peak, smoothing_bins: 1 }
        }
    }

    let egui_state = nih_plug_egui::EguiState::from_size(900, 600);

    // build callback (run once)
    let build = |_ctx: &egui::Context, _state: &mut UiState| {};

    // update: per-frame UI
    let update = move |ctx: &egui::Context, _param_setter: &ParamSetter, state: &mut UiState| {
        use egui::{CentralPanel, Layout, Align};

        CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::top_down(Align::Min), |ui| {
                ui.heading("GPU FFT Spectrum (wgpu via PaintCallback)");

                // dB range sliders + full-scale calibration
                ui.horizontal(|ui| {
                    ui.label("dB range:");
                    // max_db is fixed to +5 dB; allow the user to move only min_db
                    ui.add(egui::Slider::new(&mut state.min_db, -120.0..=5.0).text("min dB"));
                    // ui.label(format!("max dB: {} dB", state.max_db));
                    ui.separator();
                    // Full-scale reference UI removed. Full-scale is provided
                    // / normalized on the audio side; GUI assumes 1.0 = 0 dBFS.
                });

                // Controls for display smoothing (spectral smoothing only)
                ui.horizontal(|ui| {
                    ui.label("Spectral smoothing (bins):");
                    // use an Int slider for the window size; min 1, max 51
                    let mut sb = state.smoothing_bins as i32;
                    ui.add(egui::Slider::new(&mut sb, 1..=51).text("bins"));
                    // disable the 'nearest bin' interpolation option to avoid
                    // boxy vertical edges; we always use linear interpolation
                    // between bins for frequency mapping.
                    state.smoothing_bins = sb.max(1) as usize;
                });

                // allocate area
                let desired = egui::vec2(ui.available_width(), ui.available_height());
                let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());

                // background
                ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb(12, 14, 18));

                // snapshot fft
                let fft_snapshot = state.fft.lock().clone();

                // FRAME-TO-FRAME ATTACK/RELEASE SMOOTHING
                // Immediate attack: if current > prev then take current.
                // Smooth release: otherwise interpolate towards current (alpha * prev + (1-alpha) * cur).
                let alpha = 0.85_f32; // release smoothing factor (0..1). Larger => slower decay.
                // Ensure prev buffer matches length
                if state.prev_fft.len() != fft_snapshot.len() {
                    state.prev_fft.resize(fft_snapshot.len(), 0.0);
                }
                let mut frame_smoothed = vec![0.0_f32; fft_snapshot.len()];
                for i in 0..fft_snapshot.len() {
                    let cur = fft_snapshot[i];
                    let prev = state.prev_fft[i];
                    let val = if cur > prev { cur } else { alpha * prev + (1.0 - alpha) * cur };
                    frame_smoothed[i] = val;
                    // update prev for next frame
                    state.prev_fft[i] = val;
                }

                // Make a clone for painter (smoothed)
                let fft_for_painter = frame_smoothed.clone();

                // Create the PaintCallback via egui_wgpu::Callback::new_paint_callback
                // SpectrumPainter now gets data + parameters for log mapping
                // NOTE: sample_rate/fft_size are heuristics here; ideally pass real values from audio side
                let sample_rate = 44100.0_f32;
                let fft_size = fft_for_painter.len().max(2) * 2; // heuristic: half-size bins -> fft_size
                let min_freq = 20.0_f32;

                let cb = Callback::new_paint_callback(
                    rect,
                    SpectrumPainter {
                        data: fft_for_painter,
                        sample_rate,
                        fft_size,
                        min_freq,
                        min_db: state.min_db,
                        max_db: state.max_db,
                        full_scale_ref: state.full_scale_ref,
                    },
                );

                ui.painter().add(cb);

                // Spectral smoothing: moving average of `smoothing_bins` centered
                // on each bin. `smoothing_bins == 1` means no extra smoothing.
                // This is far cheaper than octave-band averaging and good enough
                // for a responsive visualization.
                let smoothed_fft = {
                    let n = frame_smoothed.len();
                    if n == 0 {
                        Vec::new()
                    } else {
                        let mut out = vec![0.0f32; n];
                        let window = state.smoothing_bins.max(1);
                        let half = window / 2;
                        for i in 0..n {
                            if window <= 1 {
                                out[i] = frame_smoothed[i];
                                continue;
                            }
                            let start = i.saturating_sub(half);
                            let end = (i + half).min(n - 1);
                            let mut sum = 0.0f32;
                            let mut cnt = 0usize;
                            for j in start..=end {
                                sum += frame_smoothed[j];
                                cnt += 1;
                            }
                            out[i] = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
                        }
                        out
                    }
                };

                // GUI-side calibration removed. Instead, track a long-term
                // full-scale reference internally via an exponential moving
                // average (EMA). This keeps the GUI immutable while allowing
                // the display to follow slowly-changing overall levels.
                if !smoothed_fft.is_empty() {
                    let observed_peak = smoothed_fft.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                    // EMA alpha close to 1 for very slow tracking (long-term)
                    let ema_alpha = 0.995_f32;
                    // floor observed to avoid collapse to zero
                    let observed = observed_peak.max(1e-12_f32);
                    state.full_scale_ref = (ema_alpha * state.full_scale_ref + (1.0 - ema_alpha) * observed).max(1e-12_f32);
                }
                // CPU-side log-frequency grid (fallback / visible reference)
                let max_f = (sample_rate / 2.0).max(min_freq + 1.0);
                let log_min = min_freq.max(1.0).log10();
                let log_max = max_f.log10();
                let log_lines_freqs = [20.0_f32, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0];
                for &f in &log_lines_freqs {
                    let f_clamped = f.max(min_freq).min(max_f);
                    let t = if (log_max - log_min).abs() < std::f32::EPSILON {
                        0.0
                    } else {
                        (f_clamped.log10() - log_min) / (log_max - log_min)
                    };
                    let px_x = rect.left() + t * rect.width();
                    let p1 = egui::pos2(px_x, rect.top());
                    let p2 = egui::pos2(px_x, rect.bottom());
                    ui.painter().line_segment(
                        [p1, p2],
                        egui::Stroke::new(1.0, egui::Color32::from_gray(110)),
                    );
                }

                // Horizontal dB reference lines at absolute dBFS values.
                // Draw fixed 20 dB steps across the configured display range
                // [state.min_db, state.max_db]. This makes the grid independent
                // of screen size or the current waveform: 0 dB corresponds to
                // full-scale (0 dBFS) and will be placed accordingly.
                let db_min = state.min_db;
                let db_max = state.max_db;
                let db_range = (db_max - db_min).max(1.0);
                let step_db = 5.0_f32; // スペース
                let start_mul = (db_min / step_db).floor() as i32;
                let end_mul = (db_max / step_db).ceil() as i32;
                for m in start_mul..=end_mul {
                    let db = (m as f32) * step_db;
                    let normalized = ((db - db_min) / db_range).clamp(0.0, 1.0);
                    let y = rect.bottom() - normalized * rect.height();
                    let p1 = egui::pos2(rect.left(), y);
                    let p2 = egui::pos2(rect.right(), y);
                    let color = if (db - 0.0).abs() < std::f32::EPSILON {
                        egui::Color32::from_rgb(255, 200, 50)
                    } else {
                        egui::Color32::from_gray(100)
                    };
                    ui.painter().line_segment([p1, p2], egui::Stroke::new(1.0, color));
                }

                // CPU fallback line (log-frequency mapping with per-pixel interpolation)
                if fft_snapshot.len() >= 2 {
                    let w = rect.width();
                    let h = rect.height();

                    // parameters for mapping
                    let sr = sample_rate; // sample rate used above
                    let fft_n = fft_size; // fft_size set above
                    let min_f = min_freq.max(1.0);
                    let max_f = (sr / 2.0).max(min_f + 1.0);

                    let log_min = min_f.log10();
                    let log_max = max_f.log10();

                    let n_pixels = w.max(2.0) as usize;

                    // Simple thresholding: use a small normalized threshold so low
                    // levels are not accidentally filtered out by complex logic.
                    // This is intentionally simple and easy to remove later.
                    let display_threshold = 0.0005_f32; // normalized (0..1)

                    // Build per-pixel Option points
                    let mut pixel_points: Vec<Option<egui::Pos2>> = vec![None; n_pixels];

                    for px in 0..n_pixels {
                        let t = px as f32 / (n_pixels - 1) as f32;
                        let log_f = log_min + t * (log_max - log_min);
                        let freq = 10f32.powf(log_f);

                        // map freq to bin (centered)
                        let bin_pos = freq / (sr / fft_n as f32) - 0.5;
                        // Use linear interpolation between bins to avoid sudden
                        // vertical drops caused by nearest-bin sampling.
                        // This keeps the line continuous without 90-degree bends.
                        let val = {
                            let bin_idx0 = bin_pos.floor() as isize;
                            let bin_idx1 = bin_pos.ceil() as isize;
                            let frac = bin_pos - bin_idx0 as f32;

                            let v0 = if bin_idx0 >= 0 && (bin_idx0 as usize) < smoothed_fft.len() {
                                smoothed_fft[bin_idx0 as usize]
                            } else {
                                0.0
                            };
                            let v1 = if bin_idx1 >= 0 && (bin_idx1 as usize) < smoothed_fft.len() {
                                smoothed_fft[bin_idx1 as usize]
                            } else {
                                0.0
                            };

                            v0 * (1.0 - frac) + v1 * frac
                        };

                        // Use absolute magnitude for dBFS calculation (handle negative values)
                        let mag_db = 20.0 * (val.abs() / state.full_scale_ref).max(1e-12).log10();
                        let denom = (state.max_db - state.min_db).max(1e-6);
                        let mut norm = ((mag_db - state.min_db) / denom).clamp(0.0, 1.0);
                        let x = rect.left() + t * w;
                        let mut y = rect.bottom() - norm * h;
                        // snap near-zero to exact bottom for visual continuity
                        if norm <= 1e-6 {
                            norm = 0.0;
                            y = rect.bottom();
                        }

                        if norm > display_threshold {
                            pixel_points[px] = Some(egui::pos2(x, y));
                        }
                    }
                    

                    // find contiguous segments of Some points and draw each segment
                    let stroke = egui::Stroke::new(1.6, egui::Color32::from_rgb(198, 198, 198));
                    let mut idx = 0usize;
                    while idx < n_pixels {
                        // skip None
                        while idx < n_pixels && pixel_points[idx].is_none() {
                            idx += 1;
                        }
                        if idx >= n_pixels {
                            break;
                        }
                        let start = idx;
                        while idx < n_pixels && pixel_points[idx].is_some() {
                            idx += 1;
                        }
                        let end = idx - 1; // inclusive

                        let seg_len = end - start + 1;
                        if seg_len < 2 {
                            // skip isolated single points to avoid artifacts
                            continue;
                        }

                        // collect visible points
                        let mut seg_points: Vec<egui::Pos2> = Vec::with_capacity(seg_len);
                        for p in start..=end {
                            seg_points.push(pixel_points[p].unwrap());
                        }

                        // 線グラフの中の塗りつぶし
                        let fill_color = egui::Color32::from_rgba_unmultiplied(226, 226, 226, 50);
                        for p in &seg_points {
                            let bottom = egui::pos2(p.x, rect.bottom());
                            ui.painter().line_segment([*p, bottom], egui::Stroke::new(1.0, fill_color));
                        }

                        // draw the visible polyline on top
                        ui.painter().add(egui::Shape::line(seg_points.clone(), stroke));

                        // Do not draw vertical drops inside the visible rect for
                        // off-screen frequencies — leaving the visible polyline
                        // truncated at its last visible point gives the impression
                        // that the rest falls off-screen.
                    }
                }
            });
        });
    };

    create_egui_editor(egui_state, UiState::new(fft_data), build, update)
}

/// ----------------------------------------
/// GPU CALLBACK IMPLEMENTATION
/// ----------------------------------------
struct SpectrumPainter {
    data: Vec<f32>,       // normalized magnitudes 0..1 (from process())
    sample_rate: f32,     // e.g. 44100.0
    fft_size: usize,      // e.g. 4096 etc. (we heuristically use data.len()*2 if not provided)
    min_freq: f32,        // lower bound for log mapping 
    min_db: f32,
    max_db: f32,
    full_scale_ref: f32,
}

// Pipeline cache
static PIPE_RES: OnceCell<PipelineResources> = OnceCell::new();
// Persistent vertex buffer store (replaceable)
static VERTEX_BUFFER: OnceCell<Mutex<Option<wgpu::Buffer>>> = OnceCell::new();
// store capacity (number of vertices) so we can recreate when needed
static VERTEX_CAPACITY: OnceCell<Mutex<usize>> = OnceCell::new();

impl egui_wgpu::CallbackTrait for SpectrumPainter {
    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        rp: &mut wgpu::RenderPass<'_>,
        types: &type_map::concurrent::TypeMap,
    ) {
        // 1) obtain RenderState from TypeMap (as in your working code)
        let render_state = types
            .get::<RenderState>()
            .expect("RenderState missing from TypeMap in paint callback");

        let device = &render_state.device;
        let queue = &render_state.queue;

        // 2) create pipeline lazily (we choose a default target format)
        // If your RenderState/TM includes the actual target format, prefer that
        let target_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let res = PIPE_RES.get_or_init(|| {
            let shader_src = r#"
                @vertex
                fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
                    return vec4<f32>(pos, 0.0, 1.0);
                }

                fn palette(v: f32) -> vec3<f32> {
                    return mix(vec3<f32>(0.0, 0.2, 0.7), vec3<f32>(0.0, 0.9, 0.3), v);
                }

                @fragment
                fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
                    // scale y from clip (-1..1) -> 0..1
                    let y = (frag_coord.y + 1.0) * 0.5;
                    let col = palette(y);
                    let alpha = 0.7 * y + 0.3;
                    return vec4<f32>(col, alpha);
                }
            "#;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("spectrum-shader"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("spectrum-pipeline-layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("spectrum-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                cache: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineStrip,
                    strip_index_format: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            PipelineResources { pipeline }
        });

        // 3) Build vertices using log-frequency mapping + dB normalization
        // Parameters
        let n_bins = self.data.len().max(2);
        let sr = self.sample_rate;
        let fft_n = self.fft_size.max(2);
        let min_f = self.min_freq.max(1.0);
        let max_f = (sr / 2.0).max(min_f + 1.0);

        // clip rect in physical pixels (top-left origin)
        let clip_min = info.clip_rect.min;
        let clip_max = info.clip_rect.max;
        let px_w = clip_max.x - clip_min.x;
        let px_h = clip_max.y - clip_min.y;

        // We'll map freq->t in [0,1] using log10
        let log_min = min_f.log10();
        let log_max = max_f.log10();

        // Create vertices (NDC) — we compute x from log frequency of each bin
        let mut verts: Vec<Vertex> = Vec::with_capacity(n_bins);
        for (i, &mag) in self.data.iter().enumerate() {
            // compute frequency of bin i (center)
            let bin = i;
            let freq = (bin as f32 + 0.5) * (sr / fft_n as f32); // center freq of bin

            // map to t using log scale
            let freq_clamped = freq.max(min_f).min(max_f);
            let t = if log_max - log_min == 0.0 {
                0.0
            } else {
                (freq_clamped.log10() - log_min) / (log_max - log_min)
            };

            // x in pixels within clip
            let px_x = clip_min.x + t * px_w;

            // convert magnitude (assumed linear amplitude) to dB relative to FS ref
            // use absolute value in case mag can be negative
            let mag_db = 20.0 * (mag.abs() / self.full_scale_ref).max(1e-12).log10();
            let norm = ((mag_db - self.min_db) / (self.max_db - self.min_db)).clamp(0.0, 1.0);

            // y in pixels (top=clip_min.y)
            let px_y = clip_min.y + (1.0 - norm) * px_h;

            // map px coords to NDC relative to clip rect (so shader sees -1..1 inside this rect)
            let ndc_x = ((px_x - clip_min.x) / px_w) * 2.0 - 1.0;
            let ndc_y = ((px_y - clip_min.y) / px_h) * 2.0 - 1.0;

            verts.push(Vertex { pos: [ndc_x, ndc_y] });
        }

        // 4) ensure vertex buffer exists & has capacity, then upload using queue.write_buffer
        // Init VERTEX_BUFFER and CAPACITY storages
        let vb_mutex = VERTEX_BUFFER.get_or_init(|| Mutex::new(None));
        let cap_mutex = VERTEX_CAPACITY.get_or_init(|| Mutex::new(0usize));

        // bytes to upload
        let verts_bytes = bytemuck::cast_slice(&verts);
        let bytes_len = verts_bytes.len() as u64;

        let mut cap_guard = cap_mutex.lock();
        let mut vb_guard = vb_mutex.lock();

        // If buffer doesn't exist or capacity too small, recreate
        let vb_ref = if vb_guard.is_none() || *cap_guard < verts.len() {
            // create buffer with some slack (e.g. 2x) to reduce reallocs
            let new_cap = (verts.len().max(256)) * 2;
            let new_size = (new_cap * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress;

            let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("spectrum-vertex-buffer"),
                size: new_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // store
            *vb_guard = Some(new_buf);
            *cap_guard = new_cap;
            vb_guard.as_ref().unwrap()
        } else {
            vb_guard.as_ref().unwrap()
        };

        // upload actual vertex bytes to offset 0
        queue.write_buffer(vb_ref, 0, verts_bytes);

        // 5) draw main spectrum
        rp.set_pipeline(&res.pipeline);
        rp.set_vertex_buffer(0, vb_ref.slice(0..bytes_len));
        rp.draw(0..(verts.len() as u32), 0..1);

        // 6) draw log-frequency vertical grid lines
        let log_lines_freqs = [20.0_f32, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0];
        for &f in &log_lines_freqs {
            let f_clamped = f.max(min_f).min(max_f);
            let t = if (log_max - log_min).abs() < std::f32::EPSILON {
                0.0
            } else {
                (f_clamped.log10() - log_min) / (log_max - log_min)
            };
            let px_x = clip_min.x + t * px_w;
            // NDC x
            let ndc_x = ((px_x - clip_min.x) / px_w) * 2.0 - 1.0;

            let line_verts = [
                Vertex { pos: [ndc_x, -1.0] },
                Vertex { pos: [ndc_x,  1.0] },
            ];

            // create a small temporary buffer and draw it
            let tmp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("logline-buffer"),
                contents: bytemuck::cast_slice(&line_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });

            rp.set_pipeline(&res.pipeline);
            rp.set_vertex_buffer(0, tmp_buf.slice(..));
            rp.draw(0..2, 0..1);
        }
    }
}
