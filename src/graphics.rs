use crate::geom::*;
use crate::render::*;

const TOLERANCE: f32 = 0.1;

pub struct Graphics {
    renderer: Renderer,
    width: f32,
    height: f32,
    color: Color,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    atlas_texture: TextureId,
    tex: TextureId,
}

impl Graphics {
    pub fn new(width: f32, height: f32) -> Graphics {
        let mut renderer = Renderer::new(width as u32, height as u32);
        let atlas_texture = renderer.create_texture(TextureFormat::A, 400, 300, &vec![0; 400 * 300]);
        let tex = renderer.create_texture(TextureFormat::RGBA, 1024, 1024, &vec![0; 1024 * 1024 * 4]);
        Graphics {
            renderer,
            width,
            height,
            color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            vertices: Vec::new(),
            indices: Vec::new(),
            atlas_texture,
            tex,
        }
    }

    pub fn set_size(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;

        self.renderer.resize(width as u32, height as u32);
    }

    pub fn clear(&mut self, color: Color) {
        self.renderer.clear(color.to_linear_premul(), &RenderOptions::default());
    }

    pub fn begin_frame(&mut self) {
        self.vertices = Vec::new();
        self.indices = Vec::new();
    }

    pub fn end_frame(&mut self) {
        self.renderer.draw(&self.vertices, &self.indices, &RenderOptions::default());
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color;
    }

    pub fn draw_mesh(&mut self, mesh: &Mesh) {
        let base = self.vertices.len() as u16;
        for point in &mesh.vertices[0..mesh.fringe_vertices] {
            let ndc = point.pixel_to_ndc(self.width, self.height);
            self.vertices.push(Vertex { pos: [ndc.x, ndc.y, 0.0], col: self.color.to_linear_premul() });
        }
        for point in &mesh.vertices[mesh.fringe_vertices..] {
            let ndc = point.pixel_to_ndc(self.width, self.height);
            self.vertices.push(Vertex { pos: [ndc.x, ndc.y, 0.0], col: [0.0, 0.0, 0.0, 0.0] });
        }
        for index in &mesh.indices {
            self.indices.push(base + index);
        }
    }

    pub fn draw_texture_test(&mut self) {
        let tex = self.renderer.create_texture(TextureFormat::A, 4, 4, &[
            127, 0, 127, 0,
            0, 127, 0, 127,
            127, 0, 127, 0,
            0, 127, 0, 127,
        ]);
        self.renderer.draw_textured(&[
            TexturedVertex { pos: [0.0, 0.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [0.0, 0.0] },
            TexturedVertex { pos: [1.0, 0.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 0.0] },
            TexturedVertex { pos: [1.0, 1.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 1.0] },
            TexturedVertex { pos: [0.0, 1.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [0.0, 1.0] },
        ], &[0, 1, 2, 0, 2, 3], tex, &RenderOptions { target: Some(self.tex), ..RenderOptions::default() });
        self.renderer.draw_textured(&[
            TexturedVertex { pos: [0.0, 0.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [0.0, 0.0] },
            TexturedVertex { pos: [1.0, 0.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 0.0] },
            TexturedVertex { pos: [1.0, 1.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 1.0] },
            TexturedVertex { pos: [0.0, 1.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [0.0, 1.0] },
        ], &[0, 1, 2, 0, 2, 3], self.tex, &RenderOptions::default());
    }

    pub fn draw_trapezoids_test(&mut self) {
        self.renderer.clear([0.0, 0.0, 0.0, 0.0], &RenderOptions { target: Some(self.atlas_texture) });
        self.renderer.draw_trapezoids(&[
            TrapezoidVertex { pos: [-1.0, -1.0], from: [150.5, 100.5], to: [100.5, 130.5] },
            TrapezoidVertex { pos: [1.0, -1.0], from: [150.5, 100.5], to: [100.5, 130.5] },
            TrapezoidVertex { pos: [1.0, 1.0], from: [150.5, 100.5], to: [100.5, 130.5] },
            TrapezoidVertex { pos: [-1.0, 1.0], from: [150.5, 100.5], to: [100.5, 130.5] },
            TrapezoidVertex { pos: [-1.0, -1.0], from: [100.5, 130.5], to: [50.5, 100.5] },
            TrapezoidVertex { pos: [1.0, -1.0], from: [100.5, 130.5], to: [50.5, 100.5] },
            TrapezoidVertex { pos: [1.0, 1.0], from: [100.5, 130.5], to: [50.5, 100.5] },
            TrapezoidVertex { pos: [-1.0, 1.0], from: [100.5, 130.5], to: [50.5, 100.5] },
            TrapezoidVertex { pos: [-1.0, -1.0], from: [50.5, 100.5], to: [100.5, 50.5] },
            TrapezoidVertex { pos: [1.0, -1.0], from: [50.5, 100.5], to: [100.5, 50.5] },
            TrapezoidVertex { pos: [1.0, 1.0], from: [50.5, 100.5], to: [100.5, 50.5] },
            TrapezoidVertex { pos: [-1.0, 1.0], from: [50.5, 100.5], to: [100.5, 50.5] },
            TrapezoidVertex { pos: [-1.0, -1.0], from: [100.5, 50.5], to: [150.5, 100.5] },
            TrapezoidVertex { pos: [1.0, -1.0], from: [100.5, 50.5], to: [150.5, 100.5] },
            TrapezoidVertex { pos: [1.0, 1.0], from: [100.5, 50.5], to: [150.5, 100.5] },
            TrapezoidVertex { pos: [-1.0, 1.0], from: [100.5, 50.5], to: [150.5, 100.5] },
        ], &[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15], &RenderOptions { target: Some(self.atlas_texture) });
        self.renderer.draw_textured(&[
            TexturedVertex { pos: [-1.0, -1.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [0.0, 0.0] },
            TexturedVertex { pos: [0.0, -1.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 0.0] },
            TexturedVertex { pos: [0.0, 0.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [1.0, 1.0] },
            TexturedVertex { pos: [-1.0, 0.0, 0.0], col: [1.0, 1.0, 1.0, 1.0], uv: [0.0, 1.0] },
        ], &[0, 1, 2, 0, 2, 3], self.atlas_texture, &RenderOptions::default());
    }
}

pub struct Path {
    points: Vec<Vec2>,
    components: Vec<usize>,
}

impl Path {
    pub fn new() -> Path {
        Path {
            points: vec![Vec2::new(0.0, 0.0)],
            components: vec![0],
        }
    }

    pub fn move_to(&mut self, point: Vec2) -> &mut Self {
        if *self.components.last().unwrap() == self.points.len() - 1 {
            *self.points.last_mut().unwrap() = point;
        } else {
            self.components.push(self.points.len());
            self.points.push(point);
        }
        self
    }

    pub fn line_to(&mut self, point: Vec2) -> &mut Self {
        self.points.push(point);
        self
    }

    pub fn quadratic_to(&mut self, control: Vec2, point: Vec2) -> &mut Self {
        let current = *self.points.last().unwrap();
        let a_x = current.x - 2.0 * control.x + point.x;
        let a_y = current.y - 2.0 * control.y + point.y;
        let dt = ((8.0 * TOLERANCE * TOLERANCE) / (a_x * a_x + a_y * a_y)).sqrt().sqrt();
        let mut t = dt;
        while t < 1.0 {
            let p12 = Vec2::lerp(t, current, control);
            let p23 = Vec2::lerp(t, control, point);
            self.points.push(Vec2::lerp(t, p12, p23));
            t += dt;
        }
        self
    }

    pub fn cubic_to(&mut self, control1: Vec2, control2: Vec2, point: Vec2) -> &mut Self {
        let current = *self.points.last().unwrap();
        let a_x = -current.x + 3.0 * control1.x - 3.0 * control2.x + point.x;
        let b_x = 3.0 * (current.x - 2.0 * control1.x + control2.x);
        let a_y = -current.y + 3.0 * control1.y - 3.0 * control2.y + point.y;
        let b_y = 3.0 * (current.y - 2.0 * control1.y + control2.y);
        let conc = (b_x * b_x + b_y * b_y).max((a_x + b_x) * (a_x + b_x) + (a_y + b_y) * (a_y + b_y));
        let dt = ((8.0 * TOLERANCE * TOLERANCE) / conc).sqrt().sqrt();
        let mut t = dt;
        while t < 1.0 {
            let p12 = Vec2::lerp(t, current, control1);
            let p23 = Vec2::lerp(t, control1, control2);
            let p34 = Vec2::lerp(t, control2, point);
            let p123 = Vec2::lerp(t, p12, p23);
            let p234 = Vec2::lerp(t, p23, p34);
            self.points.push(Vec2::lerp(t, p123, p234));
            t += dt;
        }
        self
    }

    pub fn arc_to(&mut self, radius: f32, point: Vec2) -> &mut Self {
        let current = *self.points.last().unwrap();
        let winding = radius.signum();
        let to_midpoint = 0.5 * (point - current);
        let dist_to_midpoint = to_midpoint.length();
        let radius = radius.abs().max(to_midpoint.length());
        let dist_to_center = (radius * radius - dist_to_midpoint * dist_to_midpoint).sqrt();
        let to_center = winding * dist_to_center * if to_midpoint.length() == 0.0 {
            Vec2::new(-1.0, 0.0)
        } else {
            Vec2::new(to_midpoint.y, -to_midpoint.x).normalized()
        };
        let center = current + to_midpoint + to_center;
        let mut angle = current - center;
        let end_angle = point - center;
        let rotor_x = (1.0 - 2.0 * (TOLERANCE / radius)).max(0.0);
        let rotor_y = -winding * (1.0 - rotor_x * rotor_x).sqrt();
        loop {
            let prev_sign = winding * (angle.x * end_angle.y - angle.y * end_angle.x);
            angle = Vec2::new(rotor_x * angle.x - rotor_y * angle.y, rotor_x * angle.y + rotor_y * angle.x);
            let sign = winding * (angle.x * end_angle.y - angle.y * end_angle.x);
            if prev_sign <= 0.0 && sign >= 0.0 {
                break;
            }
            self.points.push(center + angle);
        }
        self.points.push(point);
        self
    }

    pub fn fill_convex(mut self) -> Mesh {
        if self.points.len() < 3 {
            return Mesh {
                vertices: Vec::new(),
                indices: Vec::new(),
                fringe_vertices: 0,
                fringe_indices: 0
            };
        }
        let num_points = self.points.len() as u16;
        for i in 0..self.points.len() {
            let prev = self.points[(i + self.points.len() - 1) % self.points.len()];
            let curr = self.points[i];
            let next = self.points[(i + 1) % self.points.len()];
            let prev_tangent = curr - prev;
            let next_tangent = next - curr;
            let tangent = prev_tangent + next_tangent;
            let normal = Vec2::new(-tangent.y, tangent.x).normalized();
            self.points[i] = curr - 0.5 * normal;
            self.points.push(curr + 0.5 * normal);
        }
        let mut indices = Vec::new();
        for i in 1..(num_points.saturating_sub(1) as u16) {
            indices.extend_from_slice(&[0, i, i + 1]);
        }
        let fringe_indices = indices.len();
        for i in 0..(num_points as u16) {
            indices.extend_from_slice(&[
                i, num_points + i, num_points + ((i + 1) % num_points),
                i, num_points + ((i + 1) % num_points), ((i + 1) % num_points),
            ]);
        }
        Mesh {
            vertices: self.points,
            indices,
            fringe_vertices: num_points as usize,
            fringe_indices,
        }
    }

    pub fn rect(pos: Vec2, size: Vec2) -> Path {
        let mut path = Path::new();
        path.move_to(pos)
            .line_to(Vec2::new(pos.x, pos.y + size.y))
            .line_to(Vec2::new(pos.x + size.x, pos.y + size.y))
            .line_to(Vec2::new(pos.x + size.x, pos.y));
        path
    }

    pub fn rect_fill(pos: Vec2, size: Vec2) -> Mesh {
        Path::rect(pos, size).fill_convex()
    }

    pub fn round_rect(pos: Vec2, size: Vec2, radius: f32) -> Path {
        let radius = radius.min(0.5 * size.x).min(0.5 * size.y);
        let mut path = Path::new();
        path.move_to(Vec2::new(pos.x, pos.y + radius))
            .line_to(Vec2::new(pos.x, pos.y + size.y - radius))
            .arc_to(radius, Vec2::new(pos.x + radius, pos.y + size.y))
            .line_to(Vec2::new(pos.x + size.x - radius, pos.y + size.y))
            .arc_to(radius, Vec2::new(pos.x + size.x, pos.y + size.y - radius))
            .line_to(Vec2::new(pos.x + size.x, pos.y + radius))
            .arc_to(radius, Vec2::new(pos.x + size.x - radius, pos.y))
            .line_to(Vec2::new(pos.x + radius, pos.y))
            .arc_to(radius, Vec2::new(pos.x, pos.y + radius));
        path
    }

    pub fn round_rect_fill(pos: Vec2, size: Vec2, radius: f32) -> Mesh {
        Path::round_rect(pos, size, radius).fill_convex()
    }
}

pub struct Mesh {
    vertices: Vec<Vec2>,
    indices: Vec<u16>,
    fringe_vertices: usize,
    fringe_indices: usize,
}

#[derive(Copy, Clone)]
pub struct Color {
    pub r: f32, pub g: f32, pub b: f32, pub a: f32,
}

impl Color {
    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    fn to_linear_premul(&self) -> [f32; 4] {
        [
            self.a * srgb_to_linear(self.r),
            self.a * srgb_to_linear(self.g),
            self.a * srgb_to_linear(self.b),
            self.a
        ]
    }
}

fn srgb_to_linear(x: f32) -> f32 {
    if x < 0.04045 { x / 12.92 } else { ((x + 0.055)/1.055).powf(2.4)  }
}
