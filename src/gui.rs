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
    }

    impl UiState {
        fn new(fft: Arc<Mutex<Vec<f32>>>) -> Self {
            Self { fft }
        }
    }

    let egui_state = nih_plug_egui::EguiState::from_size(900, 360);

    // build callback (run once)
    let build = |_ctx: &egui::Context, _state: &mut UiState| {};

    // update: per-frame UI
    let update = move |ctx: &egui::Context, _param_setter: &ParamSetter, state: &mut UiState| {
        use egui::{CentralPanel, Layout, Align};

        CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::top_down(Align::Min), |ui| {
                ui.heading("GPU FFT Spectrum (wgpu via PaintCallback)");

                // allocate area
                let desired = egui::vec2(ui.available_width(), 260.0);
                let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());

                // background
                ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb(12, 14, 18));

                // snapshot fft
                let fft_snapshot = state.fft.lock().clone();
                // Make a clone for painter
                let fft_for_painter = fft_snapshot.clone();

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
                    },
                );

                ui.painter().add(cb);

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

                // CPU fallback line (log-frequency mapping)
                if fft_snapshot.len() >= 2 {
                    let points: Vec<egui::Pos2> = {
                        let n = fft_snapshot.len();
                        let w = rect.width();
                        let h = rect.height();

                        // parameters for mapping
                        let sr = sample_rate; // sample rate used above
                        let fft_n = fft_size; // fft_size set above
                        let min_f = min_freq.max(1.0);
                        let max_f = (sr / 2.0).max(min_f + 1.0);

                        let log_min = min_f.log10();
                        let log_max = max_f.log10();

                        (0..n)
                            .map(|i| {
                                let freq = (i as f32 + 0.5) * (sr / fft_n as f32); // bin center freq
                                let freq_clamped = freq.max(min_f).min(max_f);
                                let t = if (log_max - log_min).abs() < std::f32::EPSILON {
                                    0.0
                                } else {
                                    (freq_clamped.log10() - log_min) / (log_max - log_min) // 0..1
                                };

                                let x = rect.left() + t * w;
                                let y = rect.bottom() - fft_snapshot[i].clamp(0.0, 1.0) * h;
                                egui::pos2(x, y)
                            })
                            .collect()
                    };
                    ui.painter().add(egui::Shape::line(
                        points,
                        egui::Stroke::new(1.6, egui::Color32::from_rgb(0, 200, 255)),
                    ));
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
    min_freq: f32,        // lower bound for log mapping (e.g. 20.0)
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

            // convert magnitude (assumed linear magnitude) to dB then normalize to 0..1
            let mag_db = 20.0 * mag.max(1e-12).log10();
            let norm = ((mag_db + 80.0) / 80.0).clamp(0.0, 1.0);

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
