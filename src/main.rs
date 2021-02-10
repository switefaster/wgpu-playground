// #![windows_subsystem = "windows"]

use std::iter;

use cgmath::{InnerSpace, Rotation3, SquareMatrix, Zero};
use futures::executor::block_on;
use model::{
    obj::{DrawObjLight, DrawObjModel, ModelLoader, ObjModel, ObjModelVertex},
    Vertex,
};
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

mod camera;
mod model;
mod pipeline;
mod texture;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into()
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Light {
    position: [f32; 3],
    _padding: u32,
    color: [f32; 3],
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

const INSTANCE_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
    step_mode: wgpu::InputStepMode::Instance,
    attributes: &wgpu::vertex_attr_array![
        5=>Float4,
        6=>Float4,
        7=>Float4,
        8=>Float4
    ],
};

impl Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        INSTANCE_VERTEX_LAYOUT
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swapchain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    pipeline: wgpu::RenderPipeline,
    depth_texture: texture::Texture,
    size: winit::dpi::PhysicalSize<u32>,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    camera: camera::Camera,
    projection: camera::Projection,
    camera_controller: camera::CameraController,
    mouse_pressed: bool,
    obj_model: ObjModel,
    light_pipeline: wgpu::RenderPipeline,
    light: Light,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
}

const NUM_INSTANCES_PER_ROW: u32 = 10;

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
                    limits: wgpu::Limits::default(),
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
            present_mode: wgpu::PresentMode::Mailbox,
        };
        let swap_chain = device.create_swap_chain(&surface, &swapchain_desc);

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &swapchain_desc, "depth_texture");

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Object Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Object Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_layout,
                    &uniform_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            pipeline::create_render_pipeline(
                &device,
                &layout,
                swapchain_desc.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ObjModelVertex::desc(), InstanceRaw::desc()],
                wgpu::include_spirv!("shaders/shader.vert.spv"),
                wgpu::include_spirv!("shaders/shader.frag.spv"),
                Some("Object Render Pipeline"),
            )
        };

        let light_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            pipeline::create_render_pipeline(
                &device,
                &layout,
                swapchain_desc.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ObjModelVertex::desc()],
                wgpu::include_spirv!("shaders/light.vert.spv"),
                wgpu::include_spirv!("shaders/light.frag.spv"),
                Some("Light Render Pipeline"),
            )
        };

        let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection = camera::Projection::new(
            swapchain_desc.width,
            swapchain_desc.height,
            cgmath::Deg(45.0),
            0.1,
            100.0,
        );
        let camera_controller = camera::CameraController::new(4.0, 0.4);

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera, &projection);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                },
            }],
            label: Some("uniform_bind_group"),
        });

        let light = Light {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
        };

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &light_buffer,
                    offset: 0,
                    size: None,
                },
            }],
            label: Some("light_bind_group"),
        });

        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                    let position = cgmath::Vector3 { x, y: 0.0, z };

                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(
                            position.clone().normalize(),
                            cgmath::Deg(45.0),
                        )
                    };

                    Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let model_loader = ModelLoader::new(&device);

        let res_dir = std::env::current_dir().unwrap().join("res");
        let obj_model = model_loader
            .load(&device, &queue, &texture_layout, res_dir.join("cube.obj"))
            .unwrap();

        Self {
            surface,
            device,
            queue,
            swapchain_desc,
            swap_chain,
            size,
            depth_texture,
            pipeline,
            camera,
            projection,
            camera_controller,
            mouse_pressed: false,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            obj_model,
            light_pipeline,
            light,
            light_buffer,
            light_bind_group,
            instances,
            instance_buffer,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.swapchain_desc.width = new_size.width;
        self.swapchain_desc.height = new_size.height;
        self.swap_chain = self
            .device
            .create_swap_chain(&self.surface, &self.swapchain_desc);
        self.projection.resize(new_size.width, new_size.height);
        self.depth_texture = texture::Texture::create_depth_texture(
            &self.device,
            &self.swapchain_desc,
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

    fn update(&mut self, dt: std::time::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.uniforms
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        let old_position: cgmath::Vector3<_> = self.light.position.into();
        self.light.position =
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
                * old_position)
                .into();
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light]));
    }

    fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
        let frame = self.swap_chain.get_current_frame()?.output;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
                label: Some("Render Pass"),
            });
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.draw_model_instanced(
                &self.obj_model,
                &self.uniform_bind_group,
                &self.light_bind_group,
                0..self.instances.len() as _,
            );
            render_pass.set_pipeline(&self.light_pipeline);
            render_pass.draw_light_model(
                &self.obj_model,
                &self.uniform_bind_group,
                &self.light_bind_group,
            );
        }
        self.queue.submit(iter::once(encoder.finish()));

        Ok(())
    }
}

fn main() {
    env_logger::init();
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
        |g| match g.game.render() {
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
