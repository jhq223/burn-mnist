use crate::inference::inference::Mnist;
use eframe::egui;
use egui::*;
use egui_plot::{Bar, BarChart, Legend, Plot};
use image::{GrayImage, Rgba, RgbaImage};

use super::image::rgb2gray;

pub fn ui() -> Result<(), eframe::Error> {
    // env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 450.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Inference GUI",
        options,
        Box::new(|cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::<MyApp>::default()
        }),
    )
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
struct MyApp {
    /// in 0-1 normalized coordinates
    lines: Vec<Vec<Pos2>>,
    stroke: Stroke,
    result: Vec<f32>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            lines: Default::default(),
            stroke: Stroke::new(5.0, Color32::from_rgb(0, 0, 0)),
            result: Default::default(),
        }
    }
}

impl MyApp {
    pub fn ui_control(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            egui::stroke_ui(ui, &mut self.stroke, "Stroke");
            ui.separator();
            if ui.button("Clear Painting").clicked() {
                self.lines.clear();
            }

            if ui.button("run mnist").clicked() {
                self.run_mnist();
            }
        })
        .response
    }

    pub fn ui_content(&mut self, ui: &mut Ui) -> egui::Response {
        let (mut response, painter) =
            ui.allocate_painter(ui.available_size_before_wrap(), Sense::drag());

        let to_screen = emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, response.rect.square_proportions()),
            response.rect,
        );
        let from_screen = to_screen.inverse();

        if self.lines.is_empty() {
            self.lines.push(vec![]);
        }

        let current_line = self.lines.last_mut().unwrap();

        if let Some(pointer_pos) = response.interact_pointer_pos() {
            let canvas_pos = from_screen * pointer_pos;
            if current_line.last() != Some(&canvas_pos) {
                current_line.push(canvas_pos);
                response.mark_changed();
            }
        } else if !current_line.is_empty() {
            self.lines.push(vec![]);
            response.mark_changed();
        }

        let shapes = self
            .lines
            .iter()
            .filter(|line| line.len() >= 2)
            .map(|line| {
                let points: Vec<Pos2> = line.iter().map(|p| to_screen * *p).collect();
                egui::Shape::line(points, self.stroke)
            });

        painter.extend(shapes);

        response
    }

    fn bar_gauss(&self, ui: &mut Ui) -> Response {
        let mut chart = BarChart::new(vec![]);
        if !self.result.is_empty() {
            chart = BarChart::new(
                (0..=9)
                    .step_by(1)
                    .map(|x| (x as f64, self.result[x as usize] as f64))
                    // The 10 factor here is purely for a nice 1:1 aspect ratio
                    .map(|(x, f)| Bar::new(x, f).width(0.5))
                    .collect(),
            )
            .color(Color32::LIGHT_BLUE)
            .name("Rate");
        }

        Plot::new("rate chart")
            .legend(Legend::default())
            .clamp_grid(true)
            .y_axis_width(3)
            .allow_zoom(false)
            .allow_drag(false)
            .show(ui, |plot_ui| plot_ui.bar_chart(chart))
            .response
    }

    pub fn image_input(&self) -> Vec<f32> {
        // Set up image dimensions
        let width = 28;
        let height = 28;

        // Create a new RGB image
        let mut img = RgbaImage::from_fn(width, height, |_, _| Rgba([0, 0, 0, 255]));

        // Draw points on the image
        for row in &self.lines {
            for pos in row {
                // Convert Pos2 coordinates to pixel coordinates
                let pixel_x = (pos.x * width as f32) as u32;
                let pixel_y = (pos.y * height as f32) as u32;

                // Set the pixel color to red
                img.put_pixel(pixel_x, pixel_y, Rgba([255, 255, 255, 255]));
            }
        }

        // // Crop and resize the image
        // let cropped_img = crop_and_resize_image(&mut img.into());

        // Convert cropped image to gray scale
        let gray_img = img
            .pixels()
            .map(|rgba| rgb2gray(*rgba).0[0])
            .collect::<Vec<u8>>();

        let gray_img = GrayImage::from_raw(width, height, gray_img).expect("to image error");

        // gray_img.save("cut.png").unwrap();
        // Convert GrayImage to a 2D array of f32
        let input: Vec<f32> = gray_img
            .pixels()
            .flat_map(|rgba| vec![rgba.0[0] as f32])
            .collect();

        input
    }

    pub fn run_mnist(&mut self) {
        let input = self.image_input();
        let mut mnist = Mnist::new();
        let res = mnist.inference(&input).unwrap();
        self.result = res;
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("left_panel")
            .resizable(false)
            .exact_width(400.0)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Draw a digit here");
                    ui.group(|ui| {
                        self.ui_control(ui);
                        self.ui_content(ui);
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Probability result");
                ui.group(|ui| {
                    self.bar_gauss(ui);
                });
            });
        });
    }
}
