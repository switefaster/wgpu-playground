[[block]]
struct LightViewProj {
    view_proj_tex: mat4x4<f32>;
    view_proj: mat4x4<f32>;
    light_position: vec3<f32>;
    light_color: vec3<f32>;
};

[[group(0), binding(0)]]
var<uniform> u_light_view_proj: LightViewProj;

[[block]]
struct Transforms {
    model: mat4x4<f32>;
    model_inv_transpose: mat4x4<f32>;
};

[[group(1), binding(0)]]
var<uniform> u_transform: Transforms;

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] color: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec3<f32>,
    [[location(1)]] color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = color;
    out.position = u_light_view_proj.view_proj * u_transform.model * vec4<f32>(position, 1.0);
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) {
    if (in.color.a < 0.1) {
        discard;
    }
}
