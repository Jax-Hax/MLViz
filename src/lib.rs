use crate::{engine::{GameObject, Instance, InstanceContainer, State}, gradient_descent::build_mesh,gradient_descent::gradient_descent};
use cgmath::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

mod camera;
mod engine;
mod model;
mod resources;
mod texture;
mod gradient_descent;
pub struct MathMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    // State::new uses async code, so we're going to wait for it to finish
    let (mut state, event_loop) = State::new(false).await;

    //add models
    const SPACE_BETWEEN: f32 = 3.0;
    const NUM_INSTANCES_PER_ROW: usize = 10;
    let instances = (0..NUM_INSTANCES_PER_ROW)
        .flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance { position, rotation }
            })
        })
        .collect::<Vec<_>>();
    let mut entities = vec![];
    let instances = state.create_dynamic_instances("cube.obj", instances).await;
    entities.push(instances);


    let result = build_mesh("x*x + y*y",100,100,&state.device);
    match result {
        Ok(mesh) => entities.push(GameObject::CustomMesh(mesh)),
        Err(err) => {println!("{}",err); return;}
    }
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
    }
    //render loop
    let mut last_render_time = instant::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => state.window().request_redraw(),
            // NEW!
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                .. // We're not using device_id currently
            } => if state.mouse_pressed || state.mouse_locked {
                state.camera_controller.process_mouse(delta.0, delta.1)
            }
            // UPDATED!
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() && !state.input(event) => {
                match event {
                    #[cfg(not(target_arch="wasm32"))]
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                let now = instant::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                /*for instance in &mut entities[0].instances {
                    instance.position[0] += 0.01;
                }*/
                state.update(dt);


                let x1 = x.clone();
                let y1 = y.clone();
                let x2 = x.clone();
                let y2 = y.clone();
                let loss = &x1 * &x1 + &y1 * &y1;
                loss.backward();
                x = x2 + x1.borrow().grad * -0.0001;
                y = y2 + y1.borrow().grad * -0.0001;
                //let loss = Expr::new("x*x + y*y").value("x", &x).value("y", &y).exec().unwrap().as_f64().unwrap();


                //state.update_instances(&entities[0]);

                match state.render(&entities) {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            _ => {}
        }
    });
}
