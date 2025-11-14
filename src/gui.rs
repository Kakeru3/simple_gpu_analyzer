use std::sync::Arc;
use parking_lot::Mutex;

use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui::{self, epaint}};

use egui_wgpu::{Callback, CallbackTrait};

pub fn create(fft_data: Arc<Mutex<Vec<f32>>>) -> Option<Box<dyn Editor>> {
    struct UiState {
        fft: Arc<Mutex<Vec<f32>>>,
    }

    impl UiState {
        fn new(fft: Arc<Mutex<Vec<f32>>>) -> Self {
            Self { fft }
        }
    }

    let egui_state = nih_plug_egui::EguiState::from_size(800, 300);

    let build = |_ctx: &egui::Context, _state: &mut UiState| {};

    let update = move |ctx: &egui::Context, _setter: &ParamSetter, state: &mut UiState| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 200.0),
                egui::Sense::hover(),
            );

            // CPU 背景
            ui.painter()
                .rect_filled(rect, 0.0, egui::Color32::from_rgb(20, 22, 30));

            let fft_copy = state.fft.lock().clone();

            //
            // ==========================
            // GPU CALLBACK
            // ==========================
            //
            let gpu_cb = Callback::new_paint_callback(
                rect,
                SpectrumPainter { data: fft_copy },
            );

            ui.painter().add(gpu_cb);

            // CPU fallback line (optional)
        });
    };

    create_egui_editor(
        egui_state,
        UiState::new(fft_data),
        build,
        update,
    )
}

/// ----------------------------------------
/// GPU CALLBACK IMPLEMENTATION
/// ----------------------------------------
struct SpectrumPainter {
    data: Vec<f32>,
}

impl CallbackTrait for SpectrumPainter {
    fn paint(
        &self,
        info: epaint::PaintCallbackInfo,
        render_pass: &mut egui_wgpu::wgpu::RenderPass<'static>,
        resources: &type_map::concurrent::TypeMap,
    ) {
        // ここに wgpu の描画コードを書く
        // 例：
        // - self.data → GPU に Upload
        // - Pipeline bind
        // - render_pass.draw(...)
        //
        // 現状は No-op（クラッシュしない最小構成）
        //
        // Note: `info`, `render_pass` and `resources` are available for real rendering code.
        // Currently left unused to keep this a no-op implementation.
    }
}