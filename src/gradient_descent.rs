use eval::{Error, Expr, Value};
use wgpu::{util::DeviceExt, Device};

use crate::MathMesh;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
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
pub fn build_mesh(eval_exp: &str, x: u32, y: u32, device: &Device) -> Result<MathMesh, Error> {
    let mut vertices: Vec<Vertex> = vec![];
    let mut indices: Vec<u32> = vec![];
    let mut losses: Vec<Vec<Vertex>> = vec![];
    let max = Expr::new(eval_exp).value("x", x).value("y", y).exec();
    let max_num = match max {
                Ok(val) => match val {
                    Value::Number(string) => string.as_f64().unwrap() as f32,
                    _ => return Err(Error::Custom("Not a numeric answer".to_string())),
                },
                Err(err) => return Err(err),
            };
    for i in 0..x + 1 {
        let mut vec2 = vec![];
        for j in 0..y + 1 {
            let z = Expr::new(eval_exp).value("x", i).value("y", j).exec();
            match z {
                Ok(val) => match val {
                    Value::Number(string) => vec2.push(Vertex {
                        position: [i as f32 / x as f32, string.as_f64().unwrap() as f32 / max_num, j as f32 / y as f32],
                        tex_coords: [i as f32 / x as f32, j as f32 / y as f32],
                    }),
                    _ => return Err(Error::Custom("Not a numeric answer".to_string())),
                },
                Err(err) => return Err(err),
            }
        }
        losses.push(vec2);
    }
    let loss_len = losses.len();
    let losses_2 = losses.clone();
    for (i, i_vec) in losses.into_iter().enumerate() { //just do 0..losses.len()
        let len = i_vec.len();
        {
            let i_vec_2 = i_vec.clone();
            if i < loss_len - 1 {
                for (j, j_vertex) in i_vec.into_iter().enumerate() {
                    if j < len - 1 {
                        let base_index = vertices.len() as u32;
                        vertices.push(j_vertex);
                        vertices.push(i_vec_2[j + 1]);
                        vertices.push(losses_2[i + 1][j]);
                        vertices.push(losses_2[i + 1][j + 1]);
                        indices.push(base_index);
                        indices.push(base_index + 1);
                        indices.push(base_index + 2);
                        indices.push(base_index + 3);
                        indices.push(base_index + 2);
                        indices.push(base_index + 1);
                    }
                }
            }
        }
    }
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

pub fn gradient_descent() {
    let mut x = rustygrad::Value::from(100.0);
    let mut y = rustygrad::Value::from(100.0);
    for _idx in 1..50 {
        let x1 = x.clone();
        let y1 = y.clone();
        let loss = &x1 * &x1 + &y1 * &y1;
        loss.backward();
        x = x + x1.borrow().grad * -0.01;
        y = y + y1.borrow().grad * -0.01;
        //let loss = Expr::new("x*x + y*y").value("x", &x).value("y", &y).exec().unwrap().as_f64().unwrap();
        println!("loss: {:#?}", loss);
    }
}