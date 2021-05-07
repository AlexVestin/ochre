use std::ffi::{CStr, CString};
use gl::types::{GLuint, GLint, GLchar, GLenum, GLvoid, GLsizei};

use glutin::event::{Event, WindowEvent};
use glutin::event_loop::{ControlFlow, EventLoop};
use glutin::window::{Fullscreen, WindowBuilder};
use glutin::ContextBuilder;
use glutin::monitor::{MonitorHandle, VideoMode};


use std::io::{stdin, stdout, Write};

use ochre::{Mat2x2, PathCmd, Rasterizer, TileBuilder, Transform, Vec2, TILE_SIZE};

macro_rules! offset {
    ($type:ty, $field:ident) => { &(*(0 as *const $type)).$field as *const _ as usize }
}

const SCREEN_WIDTH: u32 = 900;
const SCREEN_HEIGHT: u32 = 900;

const ATLAS_SIZE: usize = 4096;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub pos: [i16; 2],
    pub uv: [u16; 2],
    pub col: [u8; 4],
}

struct Builder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    atlas: Vec<u8>,
    next_row: u16,
    next_col: u16,
    color: [u8; 4],
}

impl Builder {
    fn new() -> Builder {
        let mut atlas = vec![0; ATLAS_SIZE * ATLAS_SIZE];
        for row in 0..TILE_SIZE {
            for col in 0..TILE_SIZE {
                atlas[row * ATLAS_SIZE + col] = 255;
            }
        }

        Builder {
            vertices: Vec::new(),
            indices: Vec::new(),
            atlas,
            next_row: 0,
            next_col: 1,
            color: [255; 4],
        }
    }
}

impl TileBuilder for Builder {
    fn tile(&mut self, x: i16, y: i16, data: [u8; TILE_SIZE * TILE_SIZE]) {
        let base = self.vertices.len() as u32;
       
        let u1 = self.next_col * TILE_SIZE as u16;
        let u2 = (self.next_col + 1) * TILE_SIZE as u16;
        let v1 = self.next_row * TILE_SIZE as u16;
        let v2 = (self.next_row + 1) * TILE_SIZE as u16;
          
        self.vertices.push(Vertex { pos: [x, y], col: self.color, uv: [u1, v1] });
        self.vertices.push(Vertex { pos: [x + TILE_SIZE as i16, y], col: self.color, uv: [u2, v1] });
        self.vertices.push(Vertex { pos: [x + TILE_SIZE as i16, y + TILE_SIZE as i16], col: self.color, uv: [u2, v2] });
        self.vertices.push(Vertex { pos: [x, y + TILE_SIZE as i16], col: self.color, uv: [u1, v2] });
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        let nr : usize = self.next_row as usize * TILE_SIZE * ATLAS_SIZE;
        let nc : usize = self.next_col as usize * TILE_SIZE;

        for row in 0..TILE_SIZE {
            let ra: usize = row * ATLAS_SIZE;
            let rt: usize = row * TILE_SIZE;

            for col in 0..TILE_SIZE {
                self.atlas[nr + ra + nc + col] = data[rt + col];
            }
        }

        self.next_col += 1;
        if self.next_col as usize == ATLAS_SIZE / TILE_SIZE {
            self.next_col = 0;
            self.next_row += 1;
        }
    }

    fn span(&mut self, x: i16, y: i16, width: u16) {
        let base = self.vertices.len() as u32;

        self.vertices.push(Vertex { pos: [x, y], col: self.color, uv: [0, 0] });
        self.vertices.push(Vertex { pos: [x + (width as i16), y], col: self.color, uv: [0, 0] });
        self.vertices.push(Vertex { pos: [x + (width as i16), y + TILE_SIZE as i16], col: self.color, uv: [0, 0] });
        self.vertices.push(Vertex { pos: [x, y + TILE_SIZE as i16], col: self.color, uv: [0, 0] });
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

// Enumerate monitors and prompt user to choose one
fn prompt_for_monitor(el: &EventLoop<()>) -> MonitorHandle {
    for (num, monitor) in el.available_monitors().enumerate() {
        println!("Monitor #{}: {:?}", num, monitor.name());
    }

    print!("Please write the number of the monitor to use: ");
    stdout().flush().unwrap();

    let mut num = String::new();
    stdin().read_line(&mut num).unwrap();
    let num = num.trim().parse().ok().expect("Please enter a number");
    let monitor = el.available_monitors().nth(num).expect("Please enter a valid ID");

    println!("Using {:?}", monitor.name());
    monitor
}


fn main() {
    let el = EventLoop::new();

    let num: i16 = 2;
    let fullscreen = Some(match num {
        2 => Fullscreen::Borderless(Some(prompt_for_monitor(&el))),
        _ => panic!("Please enter a valid number"),
    });

    let wb = WindowBuilder::new()
        //.with_inner_size(glutin::dpi::LogicalSize::new(SCREEN_WIDTH as f64, SCREEN_HEIGHT as f64))
        .with_fullscreen(fullscreen.clone())
        .with_title("Ochre");


    let windowed_context = ContextBuilder::new().with_vsync(false).build_windowed(wb, &el).unwrap();
    let windowed_context = unsafe { windowed_context.make_current().unwrap() };

    gl::load_with(|symbol| windowed_context.get_proc_address(symbol) as *const _);

    let path = std::env::args().nth(1).expect("provide an svg file");
    let tree = usvg::Tree::from_file(path, &usvg::Options::default()).unwrap();

    let mut builder = Builder::new();

    fn render(node: &usvg::Node, builder: &mut Builder) {
        use usvg::NodeExt;

        let s: f32 = 1.0;
        match *node.borrow() {
            usvg::NodeKind::Path(ref p) => {
                let t = node.transform();
                let transform = Transform::new(Mat2x2::new(
                    t.a as f32 * s, t.c as f32,
                    t.b as f32, t.d as f32 * s,
                ), Vec2::new(t.e as f32, t.f as f32));

                let mut path = Vec::new();

                let ws: f32 = 1.0;//&SCREEN_WIDTH as f32 / 512.;
                let hs: f32 = 1.0;//&SCREEN_HEIGHT as f32 / 512.;

                for segment in p.data.0.iter() {
                    match *segment {
                        usvg::PathSegment::MoveTo { x, y } => {
                            path.push(PathCmd::Move(Vec2::new(x as f32 * ws, y as f32 * hs)));
                        }
                        usvg::PathSegment::LineTo { x, y } => {
                            path.push(PathCmd::Line(Vec2::new(x as f32 * ws, y as f32 * hs)));
                        }
                        usvg::PathSegment::CurveTo { x1, y1, x2, y2, x, y } => {
                            path.push(PathCmd::Cubic(
                                Vec2::new(x1 as f32 * ws, y1 as f32 * hs),
                                Vec2::new(x2 as f32 * ws, y2 as f32 * hs),
                                Vec2::new(x as f32 * ws, y as f32 * hs),
                            ));
                        }
                        usvg::PathSegment::ClosePath => {
                            path.push(PathCmd::Close);
                        }
                    }
                }

                if let Some(ref f) = p.fill {
                    if let usvg::Paint::Color(color) = f.paint {
                        builder.color = [color.red, color.green, color.blue, f.opacity.to_u8()];
                        let mut rasterizer = Rasterizer::new();
                        rasterizer.fill(&path, transform);
                        rasterizer.finish(builder);
                    }
                }

                if let Some(ref s) = p.stroke {
                    if let usvg::Paint::Color(color) = s.paint {
                         builder.color = [color.red, color.green, color.blue, s.opacity.to_u8()];
                         let mut rasterizer = Rasterizer::new();
                         rasterizer.stroke(&path, s.width.value() as f32, transform);
                         rasterizer.finish(builder);
                    }
                }
            }
            _ => {}
        }

        for child in node.children() {
            render(&child, builder);
        }
    }

    render(&tree.root(), &mut builder);

    let prog = Program::new(
        &CStr::from_bytes_with_nul(VERT).unwrap(),
        &CStr::from_bytes_with_nul(FRAG).unwrap()).unwrap();

    unsafe {
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        gl::Enable(gl::BLEND);
    }

    let mut tex: GLuint = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
        gl::TexImage2D(gl::TEXTURE_2D, 0, gl::R8 as GLint, ATLAS_SIZE as GLsizei, ATLAS_SIZE as GLsizei, 0, gl::RED, gl::UNSIGNED_BYTE, builder.atlas.as_ptr() as *const std::ffi::c_void);
    }

    let mut vbo: GLuint = 0;
    let mut ibo: GLuint = 0;
    let mut vao: GLuint = 0;
    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);

        gl::GenBuffers(1, &mut vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (builder.vertices.len() * std::mem::size_of::<Vertex>()) as isize, builder.vertices.as_ptr() as *const std::ffi::c_void, gl::STREAM_DRAW);

        gl::GenBuffers(1, &mut ibo);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo);
        gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, (builder.indices.len() * std::mem::size_of::<u32>()) as isize, builder.indices.as_ptr() as *const std::ffi::c_void, gl::STREAM_DRAW);

        let pos = gl::GetAttribLocation(prog.id, b"pos\0" as *const u8 as *const i8) as GLuint;
        gl::EnableVertexAttribArray(pos);
        gl::VertexAttribPointer(pos, 2, gl::SHORT, gl::FALSE, std::mem::size_of::<Vertex>() as GLint, offset!(Vertex, pos) as *const GLvoid);

        let uv = gl::GetAttribLocation(prog.id, b"uv\0" as *const u8 as *const i8) as GLuint;
        gl::EnableVertexAttribArray(uv);
        gl::VertexAttribPointer(uv, 2, gl::UNSIGNED_SHORT, gl::FALSE, std::mem::size_of::<Vertex>() as GLint, offset!(Vertex, uv) as *const GLvoid);

        let col = gl::GetAttribLocation(prog.id, b"col\0" as *const u8 as *const i8) as GLuint;
        gl::EnableVertexAttribArray(col);
        gl::VertexAttribPointer(col, 4, gl::UNSIGNED_BYTE, gl::TRUE, std::mem::size_of::<Vertex>() as GLint, offset!(Vertex, col) as *const GLvoid);

        gl::UseProgram(prog.id);

        let res = gl::GetUniformLocation(prog.id, b"res\0" as *const u8 as *const i8);
        gl::Uniform2ui(res, SCREEN_WIDTH, SCREEN_HEIGHT);

        let atlas_size = gl::GetUniformLocation(prog.id, b"atlas_size\0" as *const u8 as *const i8);
        gl::Uniform2ui(atlas_size, ATLAS_SIZE as u32, ATLAS_SIZE as u32);

        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        let tex_uniform = gl::GetUniformLocation(prog.id, b"tex\0" as *const u8 as *const i8);
        gl::Uniform1i(tex_uniform, 0);
    }

    el.run(move |event, _, control_flow| {
        
        *control_flow = ControlFlow::Wait;

        match event {
            Event::LoopDestroyed => return,
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(physical_size) => {
                 
                    println!("{:?}", physical_size);
                    windowed_context.resize(physical_size);
                    unsafe {
                        let res = gl::GetUniformLocation(prog.id, b"res\0" as *const u8 as *const i8);
                        gl::Uniform2ui(res, physical_size.width, physical_size.height);
                        gl::Viewport(0,0,physical_size.width as i32, physical_size.height as i32);
                    }
                    render(&tree.root(), &mut builder);               
                    windowed_context.swap_buffers().unwrap();
                    
                    println!("Rendered");
                },
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                _ => (),
            },
            Event::RedrawRequested(_) => {
                unsafe {
                    gl::ClearColor(1.0, 1.0, 1.0, 1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT);
                    gl::DrawElements(gl::TRIANGLES, builder.indices.len() as GLint, gl::UNSIGNED_INT, 0 as *const std::ffi::c_void);
                }
                render(&tree.root(), &mut builder);               
                windowed_context.swap_buffers().unwrap();
            }
            _ => (),
        }
    });


    
}

struct Program {
    id: GLuint,
}

impl Program {
    fn new(vert_src: &CStr, frag_src: &CStr) -> Result<Program, String> {
        unsafe {
            let vert = shader(vert_src, gl::VERTEX_SHADER).unwrap();
            let frag = shader(frag_src, gl::FRAGMENT_SHADER).unwrap();
            let prog = gl::CreateProgram();
            gl::AttachShader(prog, vert);
            gl::AttachShader(prog, frag);
            gl::LinkProgram(prog);

            let mut valid: GLint = 1;
            gl::GetProgramiv(prog, gl::COMPILE_STATUS, &mut valid);
            if valid == 0 {
                let mut len: GLint = 0;
                gl::GetProgramiv(prog, gl::INFO_LOG_LENGTH, &mut len);
                let error = CString::new(vec![b' '; len as usize]).unwrap();
                gl::GetProgramInfoLog(prog, len, std::ptr::null_mut(), error.as_ptr() as *mut GLchar);
                return Err(error.into_string().unwrap());
            }

            gl::DetachShader(prog, vert);
            gl::DetachShader(prog, frag);

            gl::DeleteShader(vert);
            gl::DeleteShader(frag);

            Ok(Program { id: prog })
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.id); }
    }
}

fn shader(shader_src: &CStr, shader_type: GLenum) -> Result<GLuint, String> {
    unsafe {
        let shader: GLuint = gl::CreateShader(shader_type);
        gl::ShaderSource(shader, 1, &shader_src.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);

        let mut valid: GLint = 1;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut valid);
        if valid == 0 {
            let mut len: GLint = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let error = CString::new(vec![b' '; len as usize]).unwrap();
            gl::GetShaderInfoLog(shader, len, std::ptr::null_mut(), error.as_ptr() as *mut GLchar);
            return Err(error.into_string().unwrap());
        }

        Ok(shader)
    }
}

const VERT: &[u8] = b"
#version 330

uniform uvec2 res;
uniform uvec2 atlas_size;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 col;

out vec2 v_uv;
out vec4 v_col;

void main() {
    vec2 scaled = 2.0 * pos / vec2(res);
    gl_Position = vec4(scaled.x - 1.0, 1.0 - scaled.y, 0.0, 1.0);
    v_uv = uv / vec2(atlas_size);
    v_col = col;
}
\0";
const FRAG: &[u8] = b"
#version 330

uniform sampler2D tex;

in vec2 v_uv;
in vec4 v_col;

out vec4 f_col;

void main() {
    f_col = v_col * vec4(1.0, 1.0, 1.0, texture(tex, v_uv).r);
}
\0";
