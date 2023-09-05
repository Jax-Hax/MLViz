use eval::{Expr, to_value, Error, Value};
use wgpu::{Device, util::DeviceExt};

use crate::{MathMesh};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}
pub fn build_mesh(eval_exp: &str, x: u32, y: u32, device: &Device) -> Result<MathMesh,Error>{
    let mut vertices: Vec<Vertex> = vec![];
    let mut indices: Vec<u32> = vec![];
    let mut losses: Vec<Vec<[f32; 3]>> = vec![];
    for i in 0..x+1{
        let mut vec2: Vec<[f32; 3]> = vec![];
        for j in 0..y+1{
            let z = Expr::new(eval_exp)
            .value("x", i)
            .value("y", j)
            .exec();
            match z {
                Ok(val) => {
                    match val {
                        Value::Number(string) => vec2.push([i as f32,j as f32,string.as_f64().unwrap() as f32]),
                        _ => return Err(Error::Custom("Not a numeric answer".to_string()))
                    }
                    
                },
                Err(err) => return Err(err)
            }
        }
        losses.push(vec2);
    }
    for i in losses{
        for j in i {
            Vertex {}
        }
    }
    let vertex_buffer = device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        Ok(MathMesh {
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
        })

}
