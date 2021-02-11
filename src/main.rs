// #![windows_subsystem = "windows"]

use std::{iter, ops::Range};

use cgmath::{Rotation3, SquareMatrix};
use futures::executor::block_on;
use log::info;
use model::{
    obj::{DrawObjLight, DrawObjModel, DrawObjShadow, ModelLoader, ObjModel, ObjModelVertex},
    Vertex,
};
use text_render::{DrawText, FontRenderer};
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

mod camera;
mod model;
mod pipeline;
mod text_render;
mod texture;

#[rustfmt::skip]
const NDC_TO_TEXCOORD: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    0.5, 0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0,
);

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

struct Light {
    depth: Range<f32>,
    fov: cgmath::Rad<f32>,
    position: [f32; 3],
    color: [f32; 3],
    buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    shadow_bind_group: wgpu::BindGroup,
    shadow_map: texture::Texture,
}

impl Light {
    fn new<F: Into<cgmath::Rad<f32>>>(
        device: &wgpu::Device,
        light_layout: &wgpu::BindGroupLayout,
        shadow_layout: &wgpu::BindGroupLayout,
        depth: Range<f32>,
        fov: F,
        position: [f32; 3],
        color: [f32; 3],
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Buffer"),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            size: std::mem::size_of::<RawLight>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let shadow_map =
            texture::Texture::create_depth_texture(device, 1024, 1024, "Shadow Depth Texture");

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: light_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("light_bind_group"),
        });
        let shadow_bind_group = {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: shadow_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&shadow_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&shadow_map.sampler),
                    },
                ],
                label: Some("shadow_bind_group"),
            })
        };
        Self {
            depth,
            fov: fov.into(),
            position,
            color,
            buffer,
            light_bind_group,
            shadow_bind_group,
            shadow_map,
        }
    }

    fn set_position(&mut self, point: [f32; 3]) {
        self.position = point;
    }

    fn update_buffer(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.into_raw()]))
    }

    fn into_raw(&self) -> RawLight {
        let proj = cgmath::perspective(self.fov, 1.0, self.depth.start, self.depth.end);
        let view = cgmath::Matrix4::look_at_rh(
            self.position.into(),
            cgmath::point3(0.0, 0.0, 0.0),
            cgmath::Vector3::unit_y(),
        );
        RawLight {
            view_proj_tex: (NDC_TO_TEXCOORD * proj * view).into(),
            view_proj: (proj * view).into(),
            position: self.position.into(),
            color: self.color,
            _padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RawLight {
    view_proj_tex: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    _padding: u32,
    color: [f32; 3],
}

struct State<'a> {
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
    cube_model: ObjModel,
    light_pipeline: wgpu::RenderPipeline,
    light: Light,
    font_renderer: FontRenderer<'a>,
    shadow_pipeline: wgpu::RenderPipeline,
}

const FONT_DATA: &[u8] = include_bytes!("font/arial.ttf");

impl<'a> State<'a> {
    async fn new(window: &Window) -> State<'a> {
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
            present_mode: wgpu::PresentMode::Immediate,
        };
        let swap_chain = device.create_swap_chain(&surface, &swapchain_desc);

        let depth_texture = texture::Texture::create_depth_texture(
            &device,
            swapchain_desc.width,
            swapchain_desc.height,
            "depth_texture",
        );

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

        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: true,
                        },
                        count: None,
                    },
                ],
                label: None,
            });

        let pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Object Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_layout,
                    &uniform_bind_group_layout,
                    &light_bind_group_layout,
                    &shadow_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            pipeline::create_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ObjModelVertex::desc()],
                wgpu::include_spirv!("shaders/shader.vert.spv"),
                Some((
                    swapchain_desc.format,
                    wgpu::include_spirv!("shaders/shader.frag.spv"),
                )),
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
                Some(texture::Texture::DEPTH_FORMAT),
                &[ObjModelVertex::desc()],
                wgpu::include_spirv!("shaders/light.vert.spv"),
                Some((
                    swapchain_desc.format,
                    wgpu::include_spirv!("shaders/light.frag.spv"),
                )),
                Some("Light Render Pipeline"),
            )
        };

        let shadow_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Mapping Pipeline Layout"),
                bind_group_layouts: &[&light_bind_group_layout],
                push_constant_ranges: &[],
            });
            pipeline::create_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ObjModelVertex::desc()],
                wgpu::include_spirv!("shaders/shadow.vert.spv"),
                None,
                Some("Shadow Mapping Pipeline"),
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
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let model_loader = ModelLoader::new(&device);

        let res_dir = std::env::current_dir().unwrap().join("res");
        let obj_model = model_loader
            .load(&device, &queue, &texture_layout, res_dir.join("cube.obj"))
            .unwrap();
        let cube_model = model_loader
            .load(&device, &queue, &texture_layout, res_dir.join("cube.obj"))
            .unwrap();

        let font_renderer = FontRenderer::new(
            &device,
            &swapchain_desc,
            FONT_DATA,
            window.scale_factor() as _,
        )
        .unwrap();

        let light = Light::new(
            &device,
            &light_bind_group_layout,
            &shadow_bind_group_layout,
            0.1..100.0,
            cgmath::Deg(45.0),
            [3.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
        );

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
            cube_model,
            light_pipeline,
            light,
            font_renderer,
            shadow_pipeline,
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
        self.light.set_position(
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
                * old_position)
                .into(),
        );
        self.light.update_buffer(&self.queue);
    }

    fn render(&mut self, window: &Window, frame_time: f64) -> Result<(), wgpu::SwapChainError> {
        let frame = self.swap_chain.get_current_frame()?.output;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.light.shadow_map.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
                label: Some("Shadow Render Pass"),
            });
            render_pass.set_pipeline(&self.shadow_pipeline);
            render_pass.draw_model_shadow(&self.obj_model, &self.light.light_bind_group);
        }
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
            render_pass.set_bind_group(3, &self.light.shadow_bind_group, &[]);
            render_pass.draw_model(
                &self.obj_model,
                &self.uniform_bind_group,
                &self.light.light_bind_group,
            );
            render_pass.set_pipeline(&self.light_pipeline);
            render_pass.draw_light_model(
                &self.cube_model,
                &self.uniform_bind_group,
                &self.light.light_bind_group,
            );
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
                label: Some("UI Render Pass"),
            });
            render_pass.draw_text_full_paragraph(
                &mut self.font_renderer,
                &self.device,
                &self.queue,
                &self.swapchain_desc,
                window.scale_factor() as _,
                &format!("FPS: {0:.1}", 1.0 / frame_time),
            );
        }
        self.queue.submit(iter::once(encoder.finish()));

        Ok(())
    }
}

fn main() {
    env_logger::init();
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
        |g| match g.game.render(&g.window, g.last_frame_time()) {
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
