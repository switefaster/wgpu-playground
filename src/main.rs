//#![windows_subsystem = "windows"]

use std::iter;

use futures::{executor::block_on, task::SpawnExt};
use log::info;
use model::obj::ModelLoader;
use specs::WorldExt;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod camera;
mod command;
mod gui;
mod model;
mod pipeline;
mod render;
mod texture;

struct PhysicsState {
    pipeline: rapier3d::pipeline::PhysicsPipeline,
    gravity: rapier3d::na::Vector3<f32>,
    integration_parameter: rapier3d::dynamics::IntegrationParameters,
    broad_phase: rapier3d::geometry::BroadPhase,
    narrow_phase: rapier3d::geometry::NarrowPhase,
    bodies: rapier3d::dynamics::RigidBodySet,
    colliders: rapier3d::geometry::ColliderSet,
    joints: rapier3d::dynamics::JointSet,
    ccd_solver: rapier3d::dynamics::CCDSolver,
}

impl PhysicsState {
    pub fn new() -> Self {
        let pipeline = rapier3d::pipeline::PhysicsPipeline::new();
        let gravity = rapier3d::na::Vector3::new(0.0, -9.8, 0.0);
        let integration_parameter = rapier3d::dynamics::IntegrationParameters::default();
        let broad_phase = rapier3d::geometry::BroadPhase::new();
        let narrow_phase = rapier3d::geometry::NarrowPhase::new();
        let bodies = rapier3d::dynamics::RigidBodySet::new();
        let colliders = rapier3d::geometry::ColliderSet::new();
        let joints = rapier3d::dynamics::JointSet::new();
        let ccd_solver = rapier3d::dynamics::CCDSolver::new();

        Self {
            pipeline,
            gravity,
            integration_parameter,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            joints,
            ccd_solver,
        }
    }
}

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
    physics: PhysicsState,
    box_handle: rapier3d::dynamics::RigidBodyHandle,
    box_render_handle: render::RenderComponentHandle,
}

const FONT_DATA: &[u8] = include_bytes!("font/arial.ttf");

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
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
            format: adapter.get_swap_chain_preferred_format(&surface).unwrap(),
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

        let grid_vertices = [
            render::ColorVertex {
                position: [-1.0, 0.0, -1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
            },
            render::ColorVertex {
                position: [1.0, 0.0, -1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
            },
            render::ColorVertex {
                position: [1.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
            },
            render::ColorVertex {
                position: [-1.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
            },
        ];

        let grid_indices: [u32; 6] = [0, 3, 1, 1, 3, 2];

        let v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Buffer"),
            contents: bytemuck::cast_slice(&[grid_vertices]),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let i_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Buffer"),
            contents: bytemuck::cast_slice(&[grid_indices]),
            usage: wgpu::BufferUsage::INDEX,
        });

        let mesh = scene.register_mesh(render::Mesh::new(v_buffer, i_buffer, 6));
        let material = scene.register_material(
            &device,
            render::RenderType::Color,
            render::MaterialType::BlinnPhong {
                ambient_color: [1.0, 1.0, 1.0],
                diffuse_color: [0.8, 0.8, 0.8],
                specular_color: [0.2, 0.2, 0.2],
                shininess: 32.0,
            },
        );

        scene.spawn_component(
            &device,
            render::RenderComponentBuilder::new()
                .add_item(render::RenderItemBuilder::new(mesh, material))
                .with_lit_type(render::LitType::Shadow)
                .with_position([0.0, -2.0, 0.0])
                .with_scale(10.0),
        );

        let box_render_handle = scene.spawn_component(
            &device,
            obj_model.with_rotation((1.0, 0.0, 0.0), cgmath::Rad(0.5)),
        );

        let mut physics = PhysicsState::new();

        let floor_body = rapier3d::dynamics::RigidBodyBuilder::new_static()
            .translation(0.0, -2.0, 0.0)
            .build();

        let box_body = rapier3d::dynamics::RigidBodyBuilder::new_dynamic()
            .rotation(rapier3d::na::Vector3::x() * 0.5)
            .build();

        let floor_handle = physics.bodies.insert(floor_body);
        let box_handle = physics.bodies.insert(box_body);

        let floor_collider = rapier3d::geometry::ColliderBuilder::trimesh(
            grid_vertices
                .iter()
                .map(|c| {
                    rapier3d::na::Point3::new(
                        c.position[0] * 10.0,
                        c.position[1] * 10.0,
                        c.position[2] * 10.0,
                    )
                })
                .collect(),
            vec![[0, 3, 1], [1, 3, 2]],
        )
        .build();
        let box_collider = rapier3d::geometry::ColliderBuilder::cuboid(1.0, 1.0, 1.0).build();

        physics
            .colliders
            .insert(floor_collider, floor_handle, &mut physics.bodies);
        physics
            .colliders
            .insert(box_collider, box_handle.clone(), &mut physics.bodies);

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
            physics,
            box_handle,
            box_render_handle,
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
            }) => {
                match key {
                    VirtualKeyCode::K => {
                        let box_body = self.physics.bodies.get_mut(self.box_handle).unwrap();
                        box_body.apply_impulse_at_point(
                            rapier3d::na::Vector3::new(0.0, 10.0, 0.0),
                            rapier3d::na::Point3::new(1.0, -1.0, 1.0),
                            true,
                        );
                    }
                    VirtualKeyCode::R => {
                        let box_body = self.physics.bodies.get_mut(self.box_handle).unwrap();
                        box_body.set_position(
                            rapier3d::na::Isometry3::new(
                                rapier3d::na::Vector3::new(0.0, 2.0, 0.0),
                                rapier3d::na::Vector3::y(),
                            ),
                            true,
                        );
                    }
                    VirtualKeyCode::S => {}
                    _ => {}
                }
                self.camera_controller.process_keyboard(*key, *state)
            }
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
        let physics_hooks = ();
        let event_handle = ();
        self.physics.pipeline.step(
            &self.physics.gravity,
            &self.physics.integration_parameter,
            &mut self.physics.broad_phase,
            &mut self.physics.narrow_phase,
            &mut self.physics.bodies,
            &mut self.physics.colliders,
            &mut self.physics.joints,
            &mut self.physics.ccd_solver,
            &physics_hooks,
            &event_handle,
        );
        let box_body = self.physics.bodies.get(self.box_handle).unwrap();
        let box_pose = box_body.position();
        {
            let box_comp = self
                .scene
                .get_component_ref_mut(self.box_render_handle)
                .unwrap();
            box_comp.set_position((
                box_pose.translation.x,
                box_pose.translation.y,
                box_pose.translation.z,
            ));
            if let Some((axis, angle)) = box_pose.rotation.axis_angle() {
                box_comp.set_rotation((axis.x, axis.y, axis.z), cgmath::Rad(angle));
            }
        }
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

    let mut state = block_on(State::new(&window));

    let mut world = specs::World::new();
    let mut dispatcher = specs::DispatcherBuilder::new().build();

    let mut now = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::DeviceEvent { ref event, .. } => {
            state.input(event);
        }
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => match input {
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::Escape),
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size);
            }
            _ => {}
        },
        Event::RedrawRequested(_) => {
            let dt = now.elapsed();
            now = Instant::now();
            state.update(dt);
            match state.render(dt) {
                Ok(_) => {}
                Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
                Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}
