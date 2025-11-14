use nih_plug::prelude::*;
use std::sync::Arc;
use parking_lot::Mutex;

mod gui;
mod gpui;

#[derive(Params)]
struct EmptyParams {}

pub struct SimpleGpuAnalyzer {
    /// 共有 FFT データ（0..1 に正規化）
    fft_data: Arc<Mutex<Vec<f32>>>,
}

impl Default for SimpleGpuAnalyzer {
    fn default() -> Self {
        Self {
            fft_data: Arc::new(Mutex::new(vec![0.0f32; 512])),
        }
    }
}

impl Plugin for SimpleGpuAnalyzer {
    const NAME: &'static str = "Simple GPU Analyzer";
    const VENDOR: &'static str = "Kakeru3";
    const URL: &'static str = "https://example/";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = "0.1.0";
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        Arc::new(EmptyParams {})
    }

    // process signature can differ by nih-plug version. Adjust if compiler complains.
    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Take first channel only for simplicity
        let samples: Vec<f32> = buffer
            .as_slice()
            .iter()
            .flat_map(|chunk| chunk.iter().copied())
            .collect();

        if samples.len() >= 128 {
            // compute FFT magnitudes (simple)
            let spec = compute_magnitude_spectrum(&samples);

            // normalise to 0..1 (log dB mapping helps visualization)
            let mapped: Vec<f32> = spec
                .into_iter()
                .map(|mag| {
                    // convert magnitude to dB (avoid log(0))
                    let amp = mag.max(1e-10);
                    let db = 20.0 * amp.log10();
                    // map db range (-80..0) to 0..1
                    ((db + 80.0) / 80.0).clamp(0.0, 1.0)
                })
                .collect();

            // store (truncate/pad into shared vec)
            let mut guard = self.fft_data.lock();
            if guard.len() != mapped.len() {
                *guard = mapped;
            } else {
                guard.copy_from_slice(&mapped);
            }
        }

        ProcessStatus::Normal
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        // pass the Arc shared buffer to GUI
        gui::create(self.fft_data.clone())
    }
}

/// compute magnitude spectrum using rustfft
fn compute_magnitude_spectrum(samples: &[f32]) -> Vec<f32> {
    use rustfft::{FftPlanner, num_complex::Complex};

    // pick power-of-two length (use samples.len() rounded down to pow2)
    // Zero-pad to a larger minimum FFT size to improve low-frequency resolution
    let n = samples.len().next_power_of_two();
    let n = if n > samples.len() { n / 2 } else { n };
    // Increase minimum FFT size so low frequencies (eg. <100 Hz) are represented.
    // This uses zero-padding when the captured sample window is smaller than `MIN_FFT`.
    const MIN_FFT: usize = 2048;
    let n = n.max(MIN_FFT);

    // prepare buffer (windowed)
    let mut input: Vec<Complex<f32>> = (0..n)
        .map(|i| {
            let s = samples.get(i).copied().unwrap_or(0.0);
            // Hann window
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
            Complex::new(s * w as f32, 0.0)
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut input);

    // magnitude of first n/2 bins
    let half = n / 2;
    let mut mags = Vec::with_capacity(half);
    for k in 0..half {
        mags.push(input[k].norm());
    }
    mags
}

impl ClapPlugin for SimpleGpuAnalyzer {
    const CLAP_ID: &'static str = "com.moist-plugins-gmbh-egui.gain-gui";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A smoothed gain parameter example plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for SimpleGpuAnalyzer {
    const VST3_CLASS_ID: [u8; 16] = *b"SimpleGpuAnalyzr";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

nih_export_clap!(SimpleGpuAnalyzer);
nih_export_vst3!(SimpleGpuAnalyzer);