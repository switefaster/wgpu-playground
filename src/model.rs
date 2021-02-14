pub mod obj {
    use std::path::Path;

    use anyhow::{Context, Result};
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
    use render::{MaterialType, RenderType};
    use wgpu::util::DeviceExt;

    use crate::{
        pipeline,
        render::{self, AlmightVertex, LitType, RenderComponentBuilder, RenderItemBuilder},
    };

    use super::super::texture;

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

    impl pipeline::Bindable<()> for BitangentComputeBinding {
        fn layout_entries(_filter: ()) -> Vec<wgpu::BindGroupLayoutEntry> {
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
        binder: pipeline::Binder<(), BitangentComputeBinding>,
        pipeline: wgpu::ComputePipeline,
    }

    impl ModelLoader {
        pub fn new(device: &wgpu::Device) -> Self {
            let binder = pipeline::Binder::new(device, (), Some("ModelLoader Compute Binder"));
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
            scene: &mut render::Scene,
            path: P,
        ) -> Result<RenderComponentBuilder> {
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

                    Ok((
                        RenderType::DiffuseNormalTexture {
                            diffuse: diffuse_texture,
                            normal: normal_texture,
                        },
                        MaterialType::BlinnPhong {
                            ambient_color: mat.ambient,
                            diffuse_color: mat.diffuse,
                            specular_color: mat.specular,
                            shininess: mat.shininess,
                        },
                    ))
                })
                .collect::<Result<Vec<(RenderType, MaterialType)>>>()?;

            let materials = scene.register_materials(device, materials);

            let meshes: Vec<(usize, render::Mesh)> = obj_models
                .par_iter()
                .map(|m| {
                    let vertices = (0..m.mesh.positions.len() / 3)
                        .into_par_iter()
                        .map(|i| AlmightVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ],
                            tex_coord: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
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

                    (
                        m.mesh.material_id.unwrap_or_default(),
                        render::Mesh::new(
                            binding.dst_vertex_buffer,
                            binding.index_buffer,
                            binding.compute_info.num_indices,
                        ),
                    )
                })
                .collect();
            let mut builder = RenderComponentBuilder::new().with_lit_type(LitType::Shadow);
            for (mat, mesh) in meshes {
                let mesh_handle = scene.register_mesh(mesh);
                builder =
                    builder.add_item(RenderItemBuilder::new(mesh_handle, materials[mat].clone()));
            }
            Ok(builder)
        }
    }
}
