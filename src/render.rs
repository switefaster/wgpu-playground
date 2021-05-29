use cgmath::{EuclideanSpace, Matrix, Rotation3, SquareMatrix};
use log::debug;
use std::{borrow::Cow, collections::HashMap, ops::Range};
use wgpu::util::DeviceExt;

use crate::{camera, pipeline, texture};

#[rustfmt::skip]
const NDC_TO_TEXCOORD: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    0.5, 0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0,
);

#[allow(dead_code)]
pub enum RenderType {
    Color,
    DiffuseTexture {
        diffuse: texture::Texture,
    },
    DiffuseNormalTexture {
        diffuse: texture::Texture,
        normal: texture::Texture,
    },
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum ColorType {
    Solid,
    Transparent,
}

///- General -> WILL NOT cast shadow
///- Shadow -> WILL cast shadow
#[allow(dead_code)]
pub enum LitType {
    General,
    Shadow,
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum MaterialType {
    BlinnPhong {
        ambient_color: [f32; 3],
        diffuse_color: [f32; 3],
        specular_color: [f32; 3],
        shininess: f32,
    },
    PBR,
}

#[allow(dead_code)]
struct MeshMaterial {
    render_type: RenderType,
    material_type: MaterialType,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    ambient_color: [f32; 3],
    _padding: u32,
    diffuse_color: [f32; 3],
    _padding_1: u32,
    specular_color: [f32; 3],
    shininess: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelTransformUniform {
    model_matrix: [[f32; 4]; 4],
    model_matrix_inv_transpose: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub normal: [f32; 3],
}

const COLOR_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<ColorVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::InputStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x4,
        2 => Float32x3,
    ],
};

impl ColorVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        COLOR_VERTEX_LAYOUT
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DiffuseVertex {
    pub position: [f32; 3],
    pub tex_coord: [f32; 2],
    pub normal: [f32; 3],
}

const DIFFUSE_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<AlmightVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::InputStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Float32x3,
    ],
};

impl DiffuseVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        DIFFUSE_VERTEX_LAYOUT
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AlmightVertex {
    pub position: [f32; 3],
    pub tex_coord: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

const ALMIGHT_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<AlmightVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::InputStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Float32x3,
        3 => Float32x3,
        4 => Float32x3,
    ],
};

impl AlmightVertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        ALMIGHT_VERTEX_LAYOUT
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl ViewUniform {
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
    shadow_map_width: u32,
    shadow_map_height: u32,
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
        (shadow_map_width, shadow_map_height): (u32, u32),
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
            shadow_map_width,
            shadow_map_height,
        }
    }

    fn set_position(&mut self, point: [f32; 3]) {
        self.position = point.into();
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

struct RenderItem {
    position: cgmath::Point3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: f32,
    color_type: ColorType,
    material_id: u32,
    transform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    mesh_id: u32,
}

impl RenderItem {
    fn update_transform(&self, queue: &wgpu::Queue, parent_transform: cgmath::Matrix4<f32>) {
        let model_matrix = parent_transform
            * (cgmath::Matrix4::from_translation(self.position.to_vec())
                * cgmath::Matrix4::from(self.rotation)
                * cgmath::Matrix4::from_scale(self.scale));
        let inv_transpose = model_matrix.transpose().invert().unwrap();
        let uniform = ModelTransformUniform {
            model_matrix: model_matrix.into(),
            model_matrix_inv_transpose: inv_transpose.into(),
        };
        queue.write_buffer(&self.transform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }
}

pub struct RenderItemBuilder {
    position: cgmath::Point3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: f32,
    color_type: ColorType,
    material_id: u32,
    mesh_id: u32,
}

impl RenderItemBuilder {
    pub fn new(mesh_id: MeshHandle, material_id: MaterialHandle) -> Self {
        Self {
            position: cgmath::point3(0.0, 0.0, 0.0),
            rotation: cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(0.0)),
            scale: 1.0,
            color_type: ColorType::Solid,
            material_id: material_id.id,
            mesh_id: mesh_id.id,
        }
    }

    pub fn with_position<P: Into<cgmath::Point3<f32>>>(mut self, new_pos: P) -> Self {
        self.position = new_pos.into();
        self
    }

    pub fn with_rotation<V: Into<cgmath::Vector3<f32>>, R: Into<cgmath::Rad<f32>>>(
        mut self,
        axis: V,
        angle: R,
    ) -> Self {
        self.rotation = cgmath::Quaternion::from_axis_angle(axis.into(), angle);
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_color_type(mut self, color_type: ColorType) -> Self {
        self.color_type = color_type;
        self
    }
}

pub struct RenderComponent {
    items: Vec<RenderItem>,
    position: cgmath::Point3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: f32,
    lit_type: LitType,
}

impl RenderComponent {
    pub fn set_position<P: Into<cgmath::Point3<f32>>>(&mut self, point: P) {
        self.position = point.into();
    }

    pub fn set_rotation<V: Into<cgmath::Vector3<f32>>, R: Into<cgmath::Rad<f32>>>(
        &mut self,
        axis: V,
        angle: R,
    ) {
        self.rotation = cgmath::Quaternion::from_axis_angle(axis.into(), angle);
    }

    fn update_transform(&self, queue: &wgpu::Queue) {
        let model_matrix = cgmath::Matrix4::from_translation(self.position.to_vec())
            * cgmath::Matrix4::from(self.rotation)
            * cgmath::Matrix4::from_scale(self.scale);
        for item in &self.items {
            item.update_transform(queue, model_matrix);
        }
    }
}

pub struct RenderComponentBuilder {
    items: Vec<RenderItemBuilder>,
    position: cgmath::Point3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: f32,
    lit_type: LitType,
}

impl RenderComponentBuilder {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            position: cgmath::point3(0.0, 0.0, 0.0),
            rotation: cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(0.0)),
            scale: 1.0,
            lit_type: LitType::Shadow,
        }
    }

    pub fn with_position<P: Into<cgmath::Point3<f32>>>(mut self, new_pos: P) -> Self {
        self.position = new_pos.into();
        self
    }

    pub fn with_rotation<V: Into<cgmath::Vector3<f32>>, R: Into<cgmath::Rad<f32>>>(
        mut self,
        axis: V,
        angle: R,
    ) -> Self {
        self.rotation = cgmath::Quaternion::from_axis_angle(axis.into(), angle);
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_lit_type(mut self, lit_type: LitType) -> Self {
        self.lit_type = lit_type;
        self
    }

    pub fn add_item(mut self, item: RenderItemBuilder) -> Self {
        self.items.push(item);
        self
    }

    pub fn add_items(mut self, mut items: Vec<RenderItemBuilder>) -> Self {
        self.items.append(&mut items);
        self
    }
}

#[derive(Clone, Copy)]
pub struct RenderComponentHandle {
    id: u32,
}

#[derive(Clone, Copy)]
pub struct MeshHandle {
    id: u32,
}

#[derive(Clone, Copy)]
pub struct MaterialHandle {
    id: u32,
}

pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_element: u32,
}

impl Mesh {
    pub fn new(vertex_buffer: wgpu::Buffer, index_buffer: wgpu::Buffer, num_element: u32) -> Self {
        Self {
            vertex_buffer,
            index_buffer,
            num_element,
        }
    }
}

pub struct Scene {
    mesh_counter: u32,
    mesh_registry: Vec<Mesh>,
    material_counter: u32,
    material_registry: Vec<MeshMaterial>,
    component_counter: u32,
    render_queue: HashMap<u32, RenderComponent>,
    camera: camera::Camera,
    projection: camera::Projection,
    view_uniform: ViewUniform,
    view_uniform_buffer: wgpu::Buffer,
    view_uniform_bind_group: wgpu::BindGroup,
    light: Light,
    solid_color_pipeline: wgpu::RenderPipeline,
    diffuse_only_pipeline: wgpu::RenderPipeline,
    almight_pipeline: wgpu::RenderPipeline,
    solid_shadow_pipeline: wgpu::RenderPipeline,
    diffuse_shadow_pipeline: wgpu::RenderPipeline,
    almight_shadow_pipeline: wgpu::RenderPipeline,
    solid_item_layout: wgpu::BindGroupLayout,
    diffuse_item_layout: wgpu::BindGroupLayout,
    almight_item_layout: wgpu::BindGroupLayout,
    transform_layout: wgpu::BindGroupLayout,
}

impl Scene {
    pub fn new(device: &wgpu::Device, swapchain_desc: &wgpu::SwapChainDescriptor) -> Self {
        let solid_item_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Solid Color Bind Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let diffuse_item_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Diffuse Only Bind Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: false,
                        },
                        count: None,
                    },
                ],
            });
        let almight_item_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Almight Bind Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: false,
                        },
                        count: None,
                    },
                ],
            });

        let view_uniform_bind_group_layout =
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
                label: Some("view_uniform_bind_group_layout"),
            });

        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("transform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
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
                            sample_type: wgpu::TextureSampleType::Depth,
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

        let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection = camera::Projection::new(
            swapchain_desc.width,
            swapchain_desc.height,
            cgmath::Deg(45.0),
            0.1,
            1000.0,
        );

        let solid_color_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Solid Color Render Pipeline Layout"),
                bind_group_layouts: &[
                    &transform_bind_group_layout,
                    &solid_item_layout,
                    &view_uniform_bind_group_layout,
                    &light_bind_group_layout,
                    &shadow_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            pipeline::create_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ColorVertex::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Solid Color Vertex"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/solid_color.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                },
                Some((
                    swapchain_desc.format,
                    wgpu::ShaderModuleDescriptor {
                        label: Some("Solid Color Fragment"),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                            "shaders/solid_color.wgsl"
                        ))),
                        flags: wgpu::ShaderFlags::VALIDATION,
                    },
                )),
                Some("Solid Color Render Pipeline"),
            )
        };

        let diffuse_only_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Diffuse Only Render Pipeline Layout"),
                bind_group_layouts: &[
                    &transform_bind_group_layout,
                    &diffuse_item_layout,
                    &view_uniform_bind_group_layout,
                    &light_bind_group_layout,
                    &shadow_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            pipeline::create_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[DiffuseVertex::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Diffuse Only Vertex"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/diffuse_only.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                },
                Some((
                    swapchain_desc.format,
                    wgpu::ShaderModuleDescriptor {
                        label: Some("Diffuse Only Fragment"),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                            "shaders/diffuse_only.wgsl"
                        ))),
                        flags: wgpu::ShaderFlags::VALIDATION,
                    },
                )),
                Some("Diffuse Only Render Pipeline"),
            )
        };

        let almight_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Almight Render Pipeline Layout"),
                bind_group_layouts: &[
                    &transform_bind_group_layout,
                    &almight_item_layout,
                    &view_uniform_bind_group_layout,
                    &light_bind_group_layout,
                    &shadow_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            pipeline::create_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[AlmightVertex::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Almight Vertex"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/almight.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                },
                Some((
                    swapchain_desc.format,
                    wgpu::ShaderModuleDescriptor {
                        label: Some("Almight Fragment"),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                            "shaders/almight.wgsl"
                        ))),
                        flags: wgpu::ShaderFlags::VALIDATION,
                    },
                )),
                Some("Almight Render Pipeline"),
            )
        };

        let solid_shadow_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Mapping Pipeline Layout"),
                bind_group_layouts: &[&light_bind_group_layout, &transform_bind_group_layout],
                push_constant_ranges: &[],
            });
            pipeline::create_shadow_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ColorVertex::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Color Shadow Mapping Vertex"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/color_shadow.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                },
                Some(wgpu::ShaderModuleDescriptor {
                    label: Some("Color Shadow Mapping Fragment"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/color_shadow.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                }),
                Some("Solid Shadow Mapping Pipeline"),
            )
        };

        let diffuse_shadow_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Mapping Pipeline Layout"),
                bind_group_layouts: &[
                    &light_bind_group_layout,
                    &transform_bind_group_layout,
                    &diffuse_item_layout,
                ],
                push_constant_ranges: &[],
            });
            pipeline::create_shadow_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[AlmightVertex::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Diffuse Shadow Mapping Vertex"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/almight_diffuse_shadow.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                },
                Some(wgpu::ShaderModuleDescriptor {
                    label: Some("Diffuse Shadow Mapping Fragment"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/almight_diffuse_shadow.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                }),
                Some("Diffuse Shadow Mapping Pipeline"),
            )
        };

        let almight_shadow_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Mapping Pipeline Layout"),
                bind_group_layouts: &[
                    &light_bind_group_layout,
                    &transform_bind_group_layout,
                    &almight_item_layout,
                ],
                push_constant_ranges: &[],
            });
            pipeline::create_shadow_render_pipeline(
                &device,
                &layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[AlmightVertex::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Almight Shadow Mapping Vertex"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/almight_diffuse_shadow.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                },
                Some(wgpu::ShaderModuleDescriptor {
                    label: Some("Almight Shadow Mapping Fragment"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "shaders/almight_diffuse_shadow.wgsl"
                    ))),
                    flags: wgpu::ShaderFlags::VALIDATION,
                }),
                Some("Almight Shadow Mapping Pipeline"),
            )
        };

        let mut view_uniform = ViewUniform::new();
        view_uniform.update_view_proj(&camera, &projection);

        let view_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Uniform Buffer"),
            contents: bytemuck::cast_slice(&[view_uniform]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let view_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &view_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_uniform_buffer.as_entire_binding(),
            }],
            label: Some("view_uniform_bind_group"),
        });

        let light = Light::new(
            &device,
            &light_bind_group_layout,
            &shadow_bind_group_layout,
            0.1..100.0,
            cgmath::Deg(60.0),
            [9.0, 6.0, 9.0],
            [1.0, 1.0, 1.0],
            (1024, 1024),
        );

        Self {
            mesh_counter: 0,
            mesh_registry: Vec::new(),
            material_counter: 0,
            material_registry: Vec::new(),
            component_counter: 0,
            render_queue: HashMap::new(),
            camera,
            projection,
            view_uniform,
            view_uniform_buffer,
            view_uniform_bind_group,
            light,
            solid_color_pipeline,
            diffuse_only_pipeline,
            almight_pipeline,
            solid_shadow_pipeline,
            diffuse_shadow_pipeline,
            almight_shadow_pipeline,
            solid_item_layout,
            diffuse_item_layout,
            almight_item_layout,
            transform_layout: transform_bind_group_layout,
        }
    }

    pub fn get_component_ref(&self, handle: RenderComponentHandle) -> Option<&RenderComponent> {
        self.render_queue.get(&handle.id)
    }

    pub fn get_component_ref_mut(
        &mut self,
        handle: RenderComponentHandle,
    ) -> Option<&mut RenderComponent> {
        self.render_queue.get_mut(&handle.id)
    }

    pub fn spawn_component(
        &mut self,
        device: &wgpu::Device,
        builder: RenderComponentBuilder,
    ) -> RenderComponentHandle {
        let root_matrix = cgmath::Matrix4::from_translation(builder.position.to_vec())
            * cgmath::Matrix4::from(builder.rotation)
            * cgmath::Matrix4::from_scale(builder.scale);
        let items = builder
            .items
            .into_iter()
            .map(|b| {
                let model_matrix = root_matrix
                    * (cgmath::Matrix4::from_translation(b.position.to_vec())
                        * cgmath::Matrix4::from(b.rotation)
                        * cgmath::Matrix4::from_scale(b.scale));
                let inv_transpose = model_matrix.transpose().invert().unwrap();
                let transform_uniform = ModelTransformUniform {
                    model_matrix: model_matrix.into(),
                    model_matrix_inv_transpose: inv_transpose.into(),
                };

                let transform_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Transform Buffer"),
                        contents: bytemuck::cast_slice(&[transform_uniform]),
                        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
                    });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Transform Bind Group"),
                    layout: &self.transform_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &transform_buffer,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

                let RenderItemBuilder {
                    position,
                    rotation,
                    scale,
                    color_type,
                    material_id,
                    mesh_id,
                } = b;

                RenderItem {
                    position,
                    rotation,
                    scale,
                    color_type,
                    material_id,
                    transform_buffer,
                    bind_group,
                    mesh_id,
                }
            })
            .collect();
        let component = RenderComponent {
            items,
            position: builder.position,
            rotation: builder.rotation,
            scale: builder.scale,
            lit_type: builder.lit_type,
        };
        self.component_counter = self.component_counter + 1;
        self.render_queue.insert(self.component_counter, component);
        RenderComponentHandle {
            id: self.component_counter,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.projection.resize(width, height);
    }

    pub fn register_mesh(&mut self, mesh: Mesh) -> MeshHandle {
        self.mesh_registry.push(mesh);
        self.mesh_counter = self.mesh_counter + 1;
        MeshHandle {
            id: self.mesh_counter - 1,
        }
    }

    pub fn register_meshes(&mut self, mut meshes: Vec<Mesh>) -> Vec<MeshHandle> {
        let nums = meshes.len() as u32;
        self.mesh_registry.append(&mut meshes);
        let result = (self.mesh_counter..self.mesh_counter + nums)
            .map(|id| MeshHandle { id })
            .collect();
        self.mesh_counter = self.mesh_counter + nums;
        result
    }

    pub fn register_material(
        &mut self,
        device: &wgpu::Device,
        render_type: RenderType,
        material_type: MaterialType,
    ) -> MaterialHandle {
        if let MaterialType::BlinnPhong {
            ambient_color,
            diffuse_color,
            specular_color,
            shininess,
        } = material_type
        {
            let material_uniform = MaterialUniform {
                ambient_color: ambient_color,
                diffuse_color: diffuse_color,
                specular_color: specular_color,
                shininess: shininess,
                _padding: 0,
                _padding_1: 0,
            };
            debug!("Material Uploaded: {:?}", material_uniform);
            let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material Buffer"),
                contents: bytemuck::cast_slice(&[material_uniform]),
                usage: wgpu::BufferUsage::UNIFORM,
            });
            let bind_group = match &render_type {
                RenderType::Color => device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Solid Color Bind Group"),
                    layout: &self.solid_item_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: material_buffer.as_entire_binding(),
                    }],
                }),
                RenderType::DiffuseTexture { diffuse } => {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Diffuse Only Bind Group"),
                        layout: &self.diffuse_item_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: material_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&diffuse.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&diffuse.sampler),
                            },
                        ],
                    })
                }
                RenderType::DiffuseNormalTexture { diffuse, normal } => {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Almight Bind Group"),
                        layout: &self.almight_item_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: material_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&diffuse.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&diffuse.sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&normal.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&normal.sampler),
                            },
                        ],
                    })
                }
            };
            self.material_registry.push(MeshMaterial {
                render_type,
                material_type,
                bind_group,
            });
            self.material_counter = self.material_counter + 1;
            MaterialHandle {
                id: self.material_counter - 1,
            }
        } else {
            unimplemented!("PBR Unsupportted")
        }
    }

    pub fn register_materials(
        &mut self,
        device: &wgpu::Device,
        materials: Vec<(RenderType, MaterialType)>,
    ) -> Vec<MaterialHandle> {
        let mut result = Vec::new();
        for (render_type, material_type) in materials {
            result.push(self.register_material(device, render_type, material_type));
        }
        result
    }

    pub fn update_light_pos(&mut self) {
        let rotate: cgmath::Quaternion<f32> =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Deg(1.0));
        let new_pos = rotate * cgmath::Vector3::from(self.light.position);
        self.light.position = new_pos.into();
    }

    pub fn render_queue_items(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_target: &wgpu::TextureView,
        depth_target: &wgpu::TextureView,
        camera_controller: &mut camera::CameraController,
        dt: std::time::Duration,
    ) {
        camera_controller.update_camera(&mut self.camera, dt);
        self.view_uniform
            .update_view_proj(&self.camera, &self.projection);
        queue.write_buffer(
            &self.view_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.view_uniform]),
        );
        self.light.update_buffer(queue);
        for (_, component) in &self.render_queue {
            component.update_transform(queue);
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.light.shadow_map.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            render_pass.set_bind_group(0, &self.light.light_bind_group, &[]);
            for (_, component) in &self.render_queue {
                if let LitType::Shadow = component.lit_type {
                    for item in &component.items {
                        render_pass.set_bind_group(1, &item.bind_group, &[]);
                        let material = &self.material_registry[item.material_id as usize];
                        match &self.material_registry[item.material_id as usize].render_type {
                            RenderType::Color => {
                                render_pass.set_pipeline(&self.solid_shadow_pipeline);
                            }
                            RenderType::DiffuseTexture { .. } => {
                                render_pass.set_pipeline(&self.diffuse_shadow_pipeline);
                                render_pass.set_bind_group(2, &material.bind_group, &[]);
                            }
                            RenderType::DiffuseNormalTexture { .. } => {
                                render_pass.set_pipeline(&self.almight_shadow_pipeline);
                                render_pass.set_bind_group(2, &material.bind_group, &[]);
                            }
                        }
                        let mesh = &self.mesh_registry[item.mesh_id as usize];
                        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..mesh.num_element, 0, 0..1);
                    }
                }
            }
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Forward Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: color_target,
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_target,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            render_pass.set_bind_group(2, &self.view_uniform_bind_group, &[]);
            render_pass.set_bind_group(3, &self.light.light_bind_group, &[]);
            render_pass.set_bind_group(4, &self.light.shadow_bind_group, &[]);
            for (_, component) in &self.render_queue {
                for item in &component.items {
                    match &self.material_registry[item.material_id as usize].render_type {
                        RenderType::Color => render_pass.set_pipeline(&self.solid_color_pipeline),
                        RenderType::DiffuseTexture { .. } => {
                            render_pass.set_pipeline(&self.diffuse_only_pipeline)
                        }
                        RenderType::DiffuseNormalTexture { .. } => {
                            render_pass.set_pipeline(&self.almight_pipeline)
                        }
                    }
                    let mesh = &self.mesh_registry[item.mesh_id as usize];
                    let material = &self.material_registry[item.material_id as usize];
                    render_pass.set_bind_group(0, &item.bind_group, &[]);
                    render_pass.set_bind_group(1, &material.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.num_element, 0, 0..1);
                }
            }
        }
    }
}
