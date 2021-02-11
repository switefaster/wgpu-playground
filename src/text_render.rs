use anyhow::Result;
use wgpu::util::DeviceExt;

use crate::{model::Vertex, pipeline, texture};

fn layout_paragraph<'a>(
    font: &rusttype::Font<'a>,
    scale: rusttype::Scale,
    width: u32,
    text: &str,
) -> Vec<rusttype::PositionedGlyph<'a>> {
    let mut result = Vec::new();
    let v_metrics = font.v_metrics(scale);
    let advance_height = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;
    let mut caret = rusttype::point(0.0, v_metrics.ascent);
    let mut last_glyph_id = None;
    for c in text.chars() {
        if c.is_control() {
            match c {
                '\r' => {
                    caret = rusttype::point(0.0, caret.y + advance_height);
                }
                '\n' => {}
                _ => {}
            }
            continue;
        }
        let base_glyph = font.glyph(c);
        if let Some(id) = last_glyph_id.take() {
            caret.x += font.pair_kerning(scale, id, base_glyph.id());
        }
        last_glyph_id = Some(base_glyph.id());
        let mut glyph = base_glyph.scaled(scale).positioned(caret);
        if let Some(bb) = glyph.pixel_bounding_box() {
            if bb.max.x > width as i32 {
                caret = rusttype::point(0.0, caret.y + advance_height);
                glyph.set_position(caret);
                last_glyph_id = None;
            }
        }
        caret.x += glyph.unpositioned().h_metrics().advance_width;
        result.push(glyph);
    }
    result
}

pub struct FontRenderer<'a> {
    cache: rusttype::gpu_cache::Cache<'a>,
    font_data: rusttype::Font<'a>,
    texture: texture::Texture,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: Option<wgpu::Buffer>,
}

const TEXT_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::InputStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float2,
        1 => Float2,
        2 => Float4,
    ],
};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
    color: [f32; 4],
}

impl Vertex for TextVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        TEXT_VERTEX_LAYOUT
    }
}

impl<'a> FontRenderer<'a> {
    pub fn new(
        device: &wgpu::Device,
        swapchain_desc: &wgpu::SwapChainDescriptor,
        font_bytes: &'a [u8],
        scale_factor: f32,
    ) -> Result<Self> {
        let font_data = rusttype::Font::try_from_bytes(font_bytes).ok_or(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Font construction failed!",
        ))?;

        let (cache_width, cache_height) =
            ((512.0 * scale_factor) as _, (512.0 * scale_factor) as _);

        let cache = rusttype::gpu_cache::Cache::builder()
            .dimensions(cache_width, cache_height)
            .build();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Font Render Bind Group Layout"),
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
            ],
        });

        let pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Font Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            pipeline::create_transparent_render_pipeline(
                device,
                &layout,
                swapchain_desc.format,
                None,
                &[TextVertex::desc()],
                wgpu::include_spirv!("shaders/text.vert.spv"),
                wgpu::include_spirv!("shaders/text.frag.spv"),
                Some("Text Render Pipeline"),
            )
        };

        let texture = texture::Texture::create_text_texture(
            device,
            cache_width,
            cache_height,
            "Font Render Texture",
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Font Render Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });

        Ok(Self {
            cache,
            font_data,
            texture,
            pipeline,
            bind_group,
            vertex_buffer: None,
        })
    }
}

pub trait DrawText<'a, 'b>
where
    'b: 'a,
{
    fn draw_text_full_paragraph(
        &mut self,
        font_renderer: &'b mut FontRenderer,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        swapchain_desc: &'a wgpu::SwapChainDescriptor,
        scale_factor: f32,
        text: &str,
    );
}

impl<'a, 'b> DrawText<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_text_full_paragraph(
        &mut self,
        font_renderer: &'b mut FontRenderer,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        swapchain_desc: &'a wgpu::SwapChainDescriptor,
        scale_factor: f32,
        text: &str,
    ) {
        let glyphs = layout_paragraph(
            &font_renderer.font_data,
            rusttype::Scale::uniform(24.0 * scale_factor),
            swapchain_desc.width,
            text,
        );
        for glyph in &glyphs {
            font_renderer.cache.queue_glyph(0, glyph.clone());
        }

        let texture_ref = &font_renderer.texture.texture;

        font_renderer
            .cache
            .cache_queued(|rect, data| {
                queue.write_texture(
                    wgpu::TextureCopyView {
                        texture: texture_ref,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: rect.min.x,
                            y: rect.min.y,
                            z: 0,
                        },
                    },
                    data,
                    wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row: rect.width(),
                        rows_per_image: rect.height(),
                    },
                    wgpu::Extent3d {
                        width: rect.width(),
                        height: rect.height(),
                        depth: 1,
                    },
                )
            })
            .unwrap();

        let (vertex_buffer, vertices) = {
            let color = [0.0, 0.0, 0.0, 1.0];

            let (screen_width, screen_height) =
                (swapchain_desc.width as f32, swapchain_desc.height as f32);
            let origin = rusttype::point(0.0, 0.0);
            let vertices: Vec<TextVertex> = glyphs
                .iter()
                .filter_map(|g| font_renderer.cache.rect_for(0, g).ok().flatten())
                .flat_map(|(uv_rect, screen_rect)| {
                    let gl_rect = rusttype::Rect {
                        min: origin
                            + (rusttype::vector(
                                screen_rect.min.x as f32 / screen_width - 0.5,
                                1.0 - screen_rect.min.y as f32 / screen_height - 0.5,
                            )) * 2.0,
                        max: origin
                            + (rusttype::vector(
                                screen_rect.max.x as f32 / screen_width - 0.5,
                                1.0 - screen_rect.max.y as f32 / screen_height - 0.5,
                            )) * 2.0,
                    };
                    vec![
                        TextVertex {
                            position: [gl_rect.min.x, gl_rect.max.y],
                            tex_coord: [uv_rect.min.x, uv_rect.max.y],
                            color,
                        },
                        TextVertex {
                            position: [gl_rect.min.x, gl_rect.min.y],
                            tex_coord: [uv_rect.min.x, uv_rect.min.y],
                            color,
                        },
                        TextVertex {
                            position: [gl_rect.max.x, gl_rect.min.y],
                            tex_coord: [uv_rect.max.x, uv_rect.min.y],
                            color,
                        },
                        TextVertex {
                            position: [gl_rect.max.x, gl_rect.min.y],
                            tex_coord: [uv_rect.max.x, uv_rect.min.y],
                            color,
                        },
                        TextVertex {
                            position: [gl_rect.max.x, gl_rect.max.y],
                            tex_coord: [uv_rect.max.x, uv_rect.max.y],
                            color,
                        },
                        TextVertex {
                            position: [gl_rect.min.x, gl_rect.max.y],
                            tex_coord: [uv_rect.min.x, uv_rect.max.y],
                            color,
                        },
                    ]
                })
                .collect();

            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Text Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsage::VERTEX,
                }),
                vertices,
            )
        };
        font_renderer.vertex_buffer = Some(vertex_buffer);

        if let Some(ref vertex_buffer) = font_renderer.vertex_buffer {
            self.set_pipeline(&font_renderer.pipeline);
            self.set_bind_group(0, &font_renderer.bind_group, &[]);
            self.set_vertex_buffer(0, vertex_buffer.slice(..));
            self.draw(0..vertices.len() as _, 0..1);
        }
    }
}
