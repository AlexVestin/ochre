use ochre::{PathCmd, Rasterizer, TileBuilder, Transform, Vec2, TILE_SIZE};

struct Builder;

impl TileBuilder for Builder {
    fn tile(&mut self, x: i16, y: i16, data: [u8; TILE_SIZE * TILE_SIZE]) {
        println!("tile at ({}, {}):", x, y);
        for row in 0..TILE_SIZE {
            print!("  ");
            for col in 0..TILE_SIZE {
                print!("{:3} ", data[row * TILE_SIZE + col]);
            }
            print!("\n");
        }
    }

    fn span(&mut self, x: i16, y: i16, width: u16) {
        //println!("span at ({}, {}), width {}", x, y, width);
    }
}

fn main() {
    let mut builder = Builder;

    let mut rasterizer = Rasterizer::new();

    // commands.push_back(static_cast<float>(PathVerb::kMove));
    // commands.push_back(static_cast<float>(100));
    // commands.push_back(static_cast<float>(200));

    // commands.push_back(static_cast<float>(PathVerb::kLine));
    // commands.push_back(static_cast<float>(300));
    // commands.push_back(static_cast<float>(200));

    // commands.push_back(static_cast<float>(PathVerb::kLine));
    // commands.push_back(static_cast<float>(300));
    // commands.push_back(static_cast<float>(400));

    // commands.push_back(static_cast<float>(PathVerb::kLine));
    // commands.push_back(static_cast<float>(100));
    // commands.push_back(static_cast<float>(400));


    rasterizer.fill(&[
        PathCmd::Move(Vec2::new(100.0, 200.0)),
        PathCmd::Line(Vec2::new(300.0, 200.0)),
        PathCmd::Line(Vec2::new(300.0, 400.0)),
        PathCmd::Line(Vec2::new(100.0, 300.0)),
        PathCmd::Close,
    ], Transform::id());
    rasterizer.finish(&mut builder);
}
