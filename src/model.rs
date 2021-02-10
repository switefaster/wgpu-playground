pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

pub mod obj {
    use std::{ops::Range, path::Path};

    use anyhow::{Context, Result};
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
    use wgpu::util::DeviceExt;

    use crate::pipeline;

    use super::{super::texture, Vertex};

    const OBJ_BUFFER_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ObjModelVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::InputStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float3,
            1 => Float2,
            2 => Float3,
            3 => Float3,
            4 => Float3,
        ],
    };

    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct ObjModelVertex {
        position: [f32; 3],
        tex_coords: [f32; 2],
        normal: [f32; 3],
        tangent: [f32; 3],
        bitangent: [f32; 3],
    }

    impl Vertex for ObjModelVertex {
        fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
            OBJ_BUFFER_LAYOUT
        }
    }

    pub struct Material {
        pub name: String,
        pub diffuse_texture: texture::Texture,
        pub normal_texture: texture::Texture,
        pub material_param: MaterialParameter,
        pub material_buffer: wgpu::Buffer,
        pub bind_group: wgpu::BindGroup,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct MaterialParameter {
        ambient_color: [f32; 3],
        _padding: f32,
        diffuse_color: [f32; 3],
        _padding_1: f32,
        specular_color: [f32; 3],
        shininess: f32,
    }

    impl Material {
        pub fn new(
            device: &wgpu::Device,
            name: &str,
            diffuse_texture: texture::Texture,
            normal_texture: texture::Texture,
            material_param: MaterialParameter,
            layout: &wgpu::BindGroupLayout,
        ) -> Self {
            let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Material Buffer", name)),
                contents: bytemuck::cast_slice(&[material_param]),
                usage: wgpu::BufferUsage::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &material_buffer,
                            offset: 0,
                            size: None,
                        },
                    },
                ],
                label: Some(name),
            });

            Self {
                name: String::from(name),
                diffuse_texture,
                normal_texture,
                bind_group,
                material_param,
                material_buffer,
            }
        }
    }

    pub struct Mesh {
        pub name: String,
        pub vertex_buffer: wgpu::Buffer,
        pub index_buffer: wgpu::Buffer,
        pub num_elements: u32,
        pub material: usize,
    }

    pub struct ObjModel {
        pub meshes: Vec<Mesh>,
        pub materials: Vec<Material>,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct ComputeInfo {
        num_vertices: u32,
        num_indices: u32,
    }

    struct BitangentComputeBinding {
        src_vertex_buffer: wgpu::Buffer,
        dst_vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        info_buffer: wgpu::Buffer,
        compute_info: ComputeInfo,
    }

    impl pipeline::Bindable for BitangentComputeBinding {
        fn layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        }

        fn bind_group_entries(&self) -> Vec<wgpu::BindGroupEntry> {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &self.src_vertex_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &self.dst_vertex_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &self.index_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &self.info_buffer,
                        offset: 0,
                        size: None,
                    },
                },
            ]
        }
    }

    pub struct ModelLoader {
        binder: pipeline::Binder<BitangentComputeBinding>,
        pipeline: wgpu::ComputePipeline,
    }

    impl ModelLoader {
        pub fn new(device: &wgpu::Device) -> Self {
            let binder = pipeline::Binder::new(device, Some("ModelLoader Compute Binder"));
            let pipeline = pipeline::create_compute_pipeline(
                device,
                &[&binder.layout],
                wgpu::include_spirv!("shaders/tangent.comp.spv"),
                Some("ModelLoader ComputePipeline"),
            );
            Self { binder, pipeline }
        }

        pub fn load<P: AsRef<Path>>(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            layout: &wgpu::BindGroupLayout,
            path: P,
        ) -> Result<ObjModel> {
            let (obj_models, obj_materials) = tobj::load_obj(path.as_ref(), true)?;

            let containing_folder = path.as_ref().parent().context("Directory has no parent")?;

            let materials = obj_materials
                .par_iter()
                .map(|mat| {
                    let mut textures = [
                        (containing_folder.join(&mat.diffuse_texture), false),
                        (containing_folder.join(&mat.normal_texture), true),
                    ]
                    .par_iter()
                    .map(|(texture_path, is_normal_map)| {
                        texture::Texture::load(device, queue, texture_path, *is_normal_map)
                    })
                    .collect::<Result<Vec<_>>>()?;

                    let normal_texture = textures.pop().unwrap();
                    let diffuse_texture = textures.pop().unwrap();

                    Ok(Material::new(
                        device,
                        &mat.name,
                        diffuse_texture,
                        normal_texture,
                        MaterialParameter {
                            ambient_color: mat.ambient,
                            diffuse_color: mat.diffuse,
                            specular_color: mat.specular,
                            shininess: mat.shininess,
                            _padding: 0.0,
                            _padding_1: 0.0,
                        },
                        layout,
                    ))
                })
                .collect::<Result<Vec<Material>>>()?;

            let meshes = obj_models
                .par_iter()
                .map(|m| {
                    let vertices = (0..m.mesh.positions.len() / 3)
                        .into_par_iter()
                        .map(|i| ObjModelVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ],
                            tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                            normal: [
                                m.mesh.normals[i * 3],
                                m.mesh.normals[i * 3 + 1],
                                m.mesh.normals[i * 3 + 2],
                            ],
                            tangent: [0.0; 3],
                            bitangent: [0.0; 3],
                        })
                        .collect::<Vec<_>>();

                    let src_vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Source Vertex Buffer", m.name)),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsage::STORAGE,
                        });

                    let dst_vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Destination Vertex Buffer", m.name)),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::STORAGE,
                        });
                    let index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Index Buffer", m.name)),
                            contents: bytemuck::cast_slice(&m.mesh.indices),
                            usage: wgpu::BufferUsage::INDEX | wgpu::BufferUsage::STORAGE,
                        });
                    let compute_info = ComputeInfo {
                        num_vertices: vertices.len() as _,
                        num_indices: m.mesh.indices.len() as _,
                    };
                    let info_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Compute Info Buffer", m.name)),
                            contents: bytemuck::cast_slice(&[compute_info]),
                            usage: wgpu::BufferUsage::UNIFORM,
                        });

                    let binding = BitangentComputeBinding {
                        src_vertex_buffer,
                        dst_vertex_buffer,
                        index_buffer,
                        info_buffer,
                        compute_info,
                    };

                    let calc_bind_group =
                        self.binder
                            .create_bind_group(&binding, device, Some("Mesh BindGroup"));
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Bitangent Calc Encoder"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Bitangent Calc Pass"),
                        });
                        pass.set_pipeline(&self.pipeline);
                        pass.set_bind_group(0, &calc_bind_group, &[]);
                        pass.dispatch(binding.compute_info.num_vertices as u32, 1, 1);
                    }

                    queue.submit(std::iter::once(encoder.finish()));
                    device.poll(wgpu::Maintain::Wait);

                    Mesh {
                        name: m.name.clone(),
                        vertex_buffer: binding.dst_vertex_buffer,
                        index_buffer: binding.index_buffer,
                        num_elements: binding.compute_info.num_indices,
                        material: m.mesh.material_id.unwrap_or(0),
                    }
                })
                .collect();
            Ok(ObjModel { meshes, materials })
        }
    }

    pub trait DrawObjModel<'a, 'b>
    where
        'b: 'a,
    {
        fn draw_mesh(
            &mut self,
            mesh: &'b Mesh,
            material: &'b Material,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        );
        fn draw_mesh_instanced(
            &mut self,
            mesh: &'b Mesh,
            material: &'b Material,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        );

        fn draw_model(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        );
        fn draw_model_instanced(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        );
    }

    impl<'a, 'b> DrawObjModel<'a, 'b> for wgpu::RenderPass<'a>
    where
        'b: 'a,
    {
        fn draw_mesh(
            &mut self,
            mesh: &'b Mesh,
            material: &'b Material,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        ) {
            self.draw_mesh_instanced(mesh, material, uniforms, light, 0..1);
        }

        fn draw_mesh_instanced(
            &mut self,
            mesh: &'b Mesh,
            material: &'b Material,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        ) {
            self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            self.set_bind_group(0, &material.bind_group, &[]);
            self.set_bind_group(1, uniforms, &[]);
            self.set_bind_group(2, light, &[]);
            self.draw_indexed(0..mesh.num_elements, 0, instances);
        }

        fn draw_model(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        ) {
            self.draw_model_instanced(model, uniforms, light, 0..1);
        }

        fn draw_model_instanced(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        ) {
            for mesh in &model.meshes {
                let material = &model.materials[mesh.material];
                self.draw_mesh_instanced(mesh, material, uniforms, light, instances.clone());
            }
        }
    }
    pub trait DrawObjLight<'a, 'b>
    where
        'b: 'a,
    {
        fn draw_light_mesh(
            &mut self,
            mesh: &'b Mesh,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        );
        fn draw_light_mesh_instanced(
            &mut self,
            mesh: &'b Mesh,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        ) where
            'b: 'a;

        fn draw_light_model(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        );
        fn draw_light_model_instanced(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        );
    }

    impl<'a, 'b> DrawObjLight<'a, 'b> for wgpu::RenderPass<'a>
    where
        'b: 'a,
    {
        fn draw_light_mesh(
            &mut self,
            mesh: &'b Mesh,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        ) {
            self.draw_light_mesh_instanced(mesh, uniforms, light, 0..1);
        }

        fn draw_light_mesh_instanced(
            &mut self,
            mesh: &'b Mesh,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        ) where
            'b: 'a,
        {
            self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            self.set_bind_group(0, uniforms, &[]);
            self.set_bind_group(1, light, &[]);
            self.draw_indexed(0..mesh.num_elements, 0, instances);
        }

        fn draw_light_model(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
        ) {
            self.draw_light_model_instanced(model, uniforms, light, 0..1);
        }

        fn draw_light_model_instanced(
            &mut self,
            model: &'b ObjModel,
            uniforms: &'b wgpu::BindGroup,
            light: &'b wgpu::BindGroup,
            instances: Range<u32>,
        ) {
            for mesh in &model.meshes {
                self.draw_light_mesh_instanced(mesh, uniforms, light, instances.clone());
            }
        }
    }
}
