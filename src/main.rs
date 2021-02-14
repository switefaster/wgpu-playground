// #![windows_subsystem = "windows"]

use std::iter;

use futures::{executor::block_on, task::SpawnExt};
use log::info;
use model::obj::ModelLoader;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

mod camera;
mod model;
mod pipeline;
mod render;
mod texture;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swapchain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    depth_texture: texture::Texture,
    size: winit::dpi::PhysicalSize<u32>,
    camera_controller: camera::CameraController,
    mouse_pressed: bool,
    glyph_brush: wgpu_glyph::GlyphBrush<()>,
    staging_belt: wgpu::util::StagingBelt,
    local_pool: futures::executor::LocalPool,
    local_spawner: futures::executor::LocalSpawner,
    scene: render::Scene,
}

const FONT_DATA: &[u8] = include_bytes!("font/arial.ttf");

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits {
                        max_bind_groups: 5,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .unwrap();

        let swapchain_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            format: adapter.get_swap_chain_preferred_format(&surface),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
        };
        let swap_chain = device.create_swap_chain(&surface, &swapchain_desc);

        let depth_texture = texture::Texture::create_depth_texture(
            &device,
            swapchain_desc.width,
            swapchain_desc.height,
            "depth_texture",
        );

        let camera_controller = camera::CameraController::new(4.0, 0.4);

        let model_loader = ModelLoader::new(&device);

        let mut scene = render::Scene::new(&device, &swapchain_desc);

        let res_dir = std::env::current_dir().unwrap().join("res");
        let obj_model = model_loader
            .load(&device, &queue, &mut scene, res_dir.join("cube.obj"))
            .unwrap();

        let arial = wgpu_glyph::ab_glyph::FontArc::try_from_slice(FONT_DATA).unwrap();
        let glyph_brush =
            wgpu_glyph::GlyphBrushBuilder::using_font(arial).build(&device, swapchain_desc.format);

        let staging_belt = wgpu::util::StagingBelt::new(1024);
        let local_pool = futures::executor::LocalPool::new();
        let local_spawner = local_pool.spawner();

        scene.spawn_component(&device, obj_model);

        Self {
            surface,
            device,
            queue,
            swapchain_desc,
            swap_chain,
            size,
            depth_texture,
            camera_controller,
            mouse_pressed: false,
            glyph_brush,
            staging_belt,
            local_pool,
            local_spawner,
            scene,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.swapchain_desc.width = new_size.width;
        self.swapchain_desc.height = new_size.height;
        self.swap_chain = self
            .device
            .create_swap_chain(&self.surface, &self.swapchain_desc);
        self.scene
            .resize(self.swapchain_desc.width, self.swapchain_desc.height);
        self.depth_texture = texture::Texture::create_depth_texture(
            &self.device,
            self.swapchain_desc.width,
            self.swapchain_desc.height,
            "depth_texture",
        );
    }

    fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(key),
                state,
                ..
            }) => self.camera_controller.process_keyboard(*key, *state),
            DeviceEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            DeviceEvent::Button { button: 1, state } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            DeviceEvent::MouseMotion { delta } => {
                if self.mouse_pressed {
                    self.camera_controller.process_mouse(delta.0, delta.1);
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, _dt: std::time::Duration) {
        self.scene.update_light_pos();
    }

    fn render(&mut self, dt: std::time::Duration) -> Result<(), wgpu::SwapChainError> {
        let frame = self.swap_chain.get_current_frame()?.output;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        self.scene.render_queue_items(
            &self.queue,
            &mut encoder,
            &frame.view,
            &self.depth_texture.view,
            &mut self.camera_controller,
            dt,
        );
        {
            self.glyph_brush.queue(wgpu_glyph::Section {
                screen_position: (30.0, 30.0),
                bounds: (
                    self.swapchain_desc.width as f32,
                    self.swapchain_desc.height as f32,
                ),
                text: vec![
                    wgpu_glyph::Text::new(&format!("FPS: {0:.1}", 1.0 / dt.as_secs_f64()))
                        .with_color([0.0, 0.0, 0.0, 1.0])
                        .with_scale(40.0),
                ],
                ..wgpu_glyph::Section::default()
            });
            self.glyph_brush
                .draw_queued(
                    &self.device,
                    &mut self.staging_belt,
                    &mut encoder,
                    &frame.view,
                    self.swapchain_desc.width,
                    self.swapchain_desc.height,
                )
                .expect("Draw queued!");
        }
        self.staging_belt.finish();
        self.queue.submit(iter::once(encoder.finish()));
        self.local_spawner
            .spawn(self.staging_belt.recall())
            .expect("Recall staging belt");
        self.local_pool.run_until_stalled();

        Ok(())
    }
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    info!("Starting up...");
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let state = block_on(State::new(&window));

    game_loop::game_loop(
        event_loop,
        window,
        state,
        60,
        0.016,
        |g| {
            g.game
                .update(std::time::Duration::from_secs_f64(g.fixed_time_step()));
        },
        |g| match g
            .game
            .render(std::time::Duration::from_secs_f64(g.last_frame_time()))
        {
            Ok(_) => (),
            Err(wgpu::SwapChainError::Lost) => g.game.resize(g.game.size),
            Err(wgpu::SwapChainError::OutOfMemory) => g.exit(),
            Err(e) => eprintln!("{:?}", e),
        },
        |g, event| match event {
            Event::DeviceEvent { ref event, .. } => {
                g.game.input(event);
            }
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == g.window.id() => match event {
                WindowEvent::CloseRequested => g.exit(),
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    } => g.exit(),
                    _ => {}
                },
                WindowEvent::Resized(physical_size) => {
                    g.game.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    g.game.resize(**new_inner_size);
                }
                _ => {}
            },
            _ => {}
        },
    );
}
