pub trait Bindable<F> {
    fn layout_entries(filter: F) -> Vec<wgpu::BindGroupLayoutEntry>;
    fn bind_group_entries(&self) -> Vec<wgpu::BindGroupEntry>;
}

pub struct Binder<F, T: Bindable<F>> {
    pub layout: wgpu::BindGroupLayout,
    _marker: std::marker::PhantomData<T>,
    _marker_1: std::marker::PhantomData<F>,
}

impl<F, T: Bindable<F>> Binder<F, T> {
    pub fn new(device: &wgpu::Device, filter: F, label: Option<&str>) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label,
            entries: &T::layout_entries(filter),
        });
        Self {
            layout,
            _marker: std::marker::PhantomData,
            _marker_1: std::marker::PhantomData,
        }
    }

    pub fn create_bind_group(
        &self,
        data: &T,
        device: &wgpu::Device,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &self.layout,
            entries: &data.bind_group_entries(),
        })
    }
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    shader: wgpu::ShaderModuleDescriptor,
    label: Option<&str>,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label,
        bind_group_layouts,
        push_constant_ranges: &[],
    });
    let cs = device.create_shader_module(&shader);
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: label,
        layout: Some(&layout),
        module: &cs,
        entry_point: "main",
    })
}

pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_desc: &[wgpu::VertexBufferLayout],
    vs: wgpu::ShaderModuleDescriptor,
    fs: Option<(wgpu::TextureFormat, wgpu::ShaderModuleDescriptor)>,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    let vs_module = device.create_shader_module(&vs);

    if let Some((color_format, f)) = fs {
        let fs_module = device.create_shader_module(&f);
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: "vs_main",
                buffers: vertex_desc,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format: format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
        })
    } else {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: "vs_main",
                buffers: vertex_desc,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format: format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: None,
        })
    }
}

pub fn create_shadow_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_desc: &[wgpu::VertexBufferLayout],
    vs: wgpu::ShaderModuleDescriptor,
    fs: Option<wgpu::ShaderModuleDescriptor>,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    let vs_module = device.create_shader_module(&vs);

    if let Some(f) = fs {
        let fs_module = device.create_shader_module(&f);
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: "vs_main",
                buffers: vertex_desc,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format: format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: "fs_main",
                targets: &[],
            }),
        })
    } else {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: "vs_main",
                buffers: vertex_desc,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format: format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: None,
        })
    }
}

#[allow(dead_code)]
pub fn create_transparent_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_desc: &[wgpu::VertexBufferLayout],
    vs: wgpu::ShaderModuleDescriptor,
    fs: wgpu::ShaderModuleDescriptor,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    let vs_module = device.create_shader_module(&vs);
    let fs_module = device.create_shader_module(&fs);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label,
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "vs_main",
            buffers: vertex_desc,
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format: format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 0,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::Zero,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrite::ALL,
            }],
        }),
    })
}
