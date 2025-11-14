use std::sync::Arc;
use egui_wgpu::wgpu;
use parking_lot::Mutex;
use once_cell::sync::OnceCell;

use nih_plug::editor::Editor;
use nih_plug::prelude::ParamSetter;
use nih_plug_egui::{create_egui_editor, egui};

use egui_wgpu::Callback; // your Callback::new_paint_callback API
use egui_wgpu::RenderState; // provides access to device/queue (may differ by version)

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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
    vertex_buffer_capacity: usize,
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

                // Clone snapshot for the GPU painter so we don't move the original, which we
                // still use for the CPU fallback drawing below.
                let fft_for_painter = fft_snapshot.clone();

                // Create the PaintCallback via egui_wgpu::Callback::new_paint_callback
                // The second arg must implement your CallbackTrait; here we use a small struct.
                let cb = Callback::new_paint_callback(
                    rect,
                    SpectrumPainter { data: fft_for_painter },
                );

                ui.painter().add(cb);

                // CPU fallback line (so you can visually compare)
                if fft_snapshot.len() >= 2 {
                    let points: Vec<egui::Pos2> = {
                        let n = fft_snapshot.len();
                        let w = rect.width();
                        let h = rect.height();
                        (0..n)
                            .map(|i| {
                                let x = rect.left() + i as f32 / (n - 1) as f32 * w;
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
    data: Vec<f32>, // normalized magnitudes 0..1
}

// Cache pipeline per process (OnceCell inside paint closure uses global static)
static PIPE_RES: OnceCell<PipelineResources> = OnceCell::new();

impl egui_wgpu::CallbackTrait for SpectrumPainter {
    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        rp: &mut wgpu::RenderPass<'static>,
        types: &type_map::concurrent::TypeMap,
    ) {
        // 1) obtain device & queue from RenderState stored in the provided TypeMap
        let render_state = types
            .get::<RenderState>()
            .expect("RenderState missing from TypeMap in paint callback");

        // Access device and queue from RenderState fields (some egui_wgpu versions expose these as fields)
        let device = &render_state.device;
        let queue = &render_state.queue;

        // 2) Prepare pipeline resources (lazy init)
        // We need a pipeline compatible with the current render target format.
        // PaintCallbackInfo in this egui_wgpu version doesn't expose target format;
        // use a sensible default (BGRA sRGB) instead.
        let target_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        // NOTE: if RenderState provides the actual format via the TypeMap, you can read it from there.

        // Initialize pipeline lazily
        let res = PIPE_RES.get_or_init(|| {
            // create shader
            let shader_src = r#"
                @vertex
                fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
                    return vec4<f32>(pos, 0.0, 1.0);
                }

                @fragment
                fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
                    // color gradient top->bottom based on y in clip space (-1..1)
                    let y = (frag_coord.y + 1.0) * 0.5;
                    return vec4<f32>(0.0, y * 0.9 + 0.1, 1.0, 1.0);
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
                cache: None,
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
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineStrip,
                    strip_index_format: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            PipelineResources {
                pipeline,
                vertex_buffer_capacity: 0,
            }
        });

        // 3) build vertices (NDC) from data and info.clip_rect / info.screen_size
        let n = self.data.len().max(2);
        // clip rect is in physical pixels
        let clip_min = info.clip_rect.min; // egui::emath::Pos2 (physical px)
        let clip_max = info.clip_rect.max;
        let px_w = clip_max.x - clip_min.x;
        let px_h = clip_max.y - clip_min.y;

        // screen physical size: PaintCallbackInfo may not expose a screen_size field across versions,
        // so use the clip rect size as a fallback (physical pixels).
        let screen_size = [px_w, px_h];

        // create vertices: map pixel coords -> NDC (-1..1)
        let mut verts: Vec<Vertex> = Vec::with_capacity(n);
        for (i, &mag) in self.data.iter().enumerate() {
            let t = if n <= 1 { 0.0 } else { i as f32 / (n - 1) as f32 };
            let px_x = clip_min.x + t * px_w;
            let px_y = clip_max.y - mag.clamp(0.0, 1.0) * px_h;

            // map to NDC using screen_size
            let ndc_x = (px_x / screen_size[0]) * 2.0 - 1.0;
            let ndc_y = (px_y / screen_size[1]) * 2.0 - 1.0;
            verts.push(Vertex { pos: [ndc_x, ndc_y] });
        }

        // 4) create or resize vertex buffer and upload with queue.write_buffer (preferred)
        let vb_usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        // For simplicity create a temp buffer each frame:
        // Better: reuse a persistent buffer and queue.write_buffer to update contents.
        // Here we create ephemeral buffer and set it
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spectrum-vertex-buffer"),
            contents: bytemuck::cast_slice(&verts),
            usage: vb_usage,
        });

        // 5) draw: set pipeline, set vertex buffer, draw
        rp.set_pipeline(&res.pipeline);
        rp.set_vertex_buffer(0, vb.slice(..));
        rp.draw(0..(verts.len() as u32), 0..1);
    }
}
