#include <vector>
#include "geom.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "jpeglib.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

struct Vertex {
    int16_t pos[2];
    uint16_t uv[2];
    uint8_t color[4];
};

constexpr int TILE_SIZE = 8;
constexpr int ATLAS_SIZE = 1024;

constexpr float TOLERANCE = 0.005;

#define i16 int32_t
#define f32 float

enum class PathVerb {
    kMove,   
    kLine,   
    kQuad,   
    kClose,  
    kConic,  
    kCubic   
};


void Join(std::vector<float>& path, std::vector<float>& commands, float width, VPoint prevNormal, VPoint nextNormal, VPoint point) {
    float offset = 1.0 / (1.0 + prevNormal.Dot(nextNormal));
    if (fabs(offset) > 2.0) {
        path.push_back(static_cast<float>(PathVerb::kLine));
        VPoint p0 = point + 0.5 * width * prevNormal;
        path.push_back(p0.fX);
        path.push_back(p0.fY);

        path.push_back(static_cast<float>(PathVerb::kLine));
        VPoint p1 = point + 0.5 * width * nextNormal;
        path.push_back(p1.fX);
        path.push_back(p1.fY);
    } else {
        path.push_back(static_cast<float>(PathVerb::kLine));
        VPoint p0 = point + 0.5 * width * offset * (prevNormal + nextNormal);
        path.push_back(p0.fX);
        path.push_back(p0.fY);
    }
}

void Offset(std::vector<float>& output, float width, std::vector<float>& commands, bool closed, bool reverse) {
    
    VPoint firstPoint, nextPoint, prevPoint;
    VPoint prevNormal = { 0.0, 0.0 };
    int size = commands.size();

    if (closed == reverse) {
        firstPoint = { commands[1], commands[2] };
    } else {
        firstPoint = { commands[size - 2], commands[size - 1] };
    }

    int i = 0;
    while (true) {
        if (i < commands.size()) {
            if (reverse) {
                nextPoint = { commands[ size - (i + 1) * 3 ], commands[size - (i + 1)  * 3] };
            } else {
                nextPoint = { commands[i * 3], commands[i * 3] };
            }
        } else {
            nextPoint.fX = firstPoint.fX;
            nextPoint.fY = firstPoint.fY;
        }

        if (nextPoint != prevPoint || i == commands.size()) {
            VPoint nextTangent = nextPoint - prevPoint;
            VPoint nextNormal = { -nextTangent.fY, nextTangent.fX };
            float nextNormalLen = nextNormal.Length();

            nextNormal = nextNormalLen == 0.0 ? VPoint::Make( 0.0, 0.0) : nextNormal * (1.0 / nextNormalLen);
            Join(output, commands, width, prevNormal, nextNormal, prevPoint);
            prevPoint = nextPoint;
            prevNormal = nextNormal;
        }
        i+=3;

        if (i > commands.size()) 
            break;
    }
}

std::vector<float> Stroke(std::vector<float>& commands, float width) {
    
    bool closed = false;
    std::vector<float> output;
    uint32_t contour_start = 0;
    uint32_t contour_end = 0;
    std::cout << "Stroke 1" << std::endl;

    int i = 0;
    while (true) {  
        PathVerb command = static_cast<PathVerb>(commands[i]);
        if (command == PathVerb::kClose)  {
            closed = true;
        }

        if (command == PathVerb::kMove || command == PathVerb::kClose) {
            if (contour_start != contour_end) {
                uint32_t base = output.size();

                Offset(output, width, commands, closed, false);
                output[base] = static_cast<float>(PathVerb::kMove);
                base = output.size();

                Offset(output, width, commands, closed, true);
                if (closed) {
                    output[base] = static_cast<float>(PathVerb::kMove);
                }

                output.push_back(static_cast<float>(PathVerb::kClose));
            }
        }

        switch(command) {
            case PathVerb::kMove: {
                contour_start = contour_end;
                contour_end = contour_start + 1;
                break;
            }
            case PathVerb::kLine: {
                contour_end += 1;
                break;
            }
            case PathVerb::kClose: {
                contour_start = contour_end + 1;
                contour_end = contour_start;
                closed = true;
                break;
            }
        }

        i += 3;
        if (i > commands.size()) {
            break;
        }
    }
    return std::move(output);
}

struct TileBuilder {
    virtual void Tile(uint32_t w, uint32_t h, std::vector<uint8_t>&tile) = 0;
    virtual void Span(int16_t x, int16_t y, int16_t width) = 0;
    virtual const uint8_t* GetAtlas() const = 0; 
    virtual const std::vector<Vertex> GetVertices() const = 0; 
    virtual const std::vector<uint32_t> GetIndices() const = 0; 
};

void write_to_img(std::string& name, const uint8_t* pdata, int screenWidth, int screenHeight) {
  const int num_components = 1;

  FILE* outfile;
  std::string pre = "texture.jpg";
  if ((outfile = fopen(pre.c_str(), "wb")) == NULL) {
    printf("can't open %s");
    exit(1);
  }
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);
  cinfo.image_width = screenWidth;
  cinfo.image_height = screenHeight;
  cinfo.input_components = num_components;
  cinfo.in_color_space = JCS_GRAYSCALE;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, true);
  jpeg_start_compress(&cinfo, true);
  JSAMPROW row_pointer;
  int row_stride = screenWidth * num_components;
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer = (JSAMPROW)&pdata[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }
  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
}

struct Increment {
    i16 x;
    i16 y;
    float area;
    float height;
};

struct TileIncrement {
    i16 tile_x;
    i16 tile_y;
    int8_t sign;
};

struct Bin {
    i16 tile_x;
    i16 tile_y;
    uint32_t start;
    uint32_t end;
};

class Rasterizer {
public:  
    Rasterizer() : first({0,0}) ,  last({0,0}), tile_y_prev(0) { }


    void Finish(TileBuilder& builder) {
        if (last != first) {
            LineTo(first);
        }

        std::vector<Bin> bins;
        Bin bin = { 0, 0, 0, 0 };
        if (increments.size() > 0) {
            bin.tile_x = increments[0].x / TILE_SIZE;
            bin.tile_y = increments[0].y / TILE_SIZE;
        }

        for(uint32_t i = 0; i < increments.size(); i++) {
            Increment increment = increments[i];
            i16 tile_x = increment.x / TILE_SIZE;
            i16 tile_y = increment.y / TILE_SIZE;
            if (tile_x != bin.tile_x || tile_y != bin.tile_y) {
                bins.push_back(bin);
                bin = { tile_x, tile_y, i, i };
            }
            bin.end += 1;
        }

        bins.push_back(bin);
        std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) -> bool {
            if(a.tile_y == b.tile_y) return a.tile_x < b.tile_x;
            return a.tile_y < b.tile_y;
        });

        std::vector<float> areas(TILE_SIZE*TILE_SIZE, 0.0f);
        std::vector<float> heights(TILE_SIZE*TILE_SIZE, 0.0f);
        std::vector<float> prev(TILE_SIZE, 0.0f);
        std::vector<float> next(TILE_SIZE, 0.0f);

        uint32_t tile_increments_i = 0;
        uint8_t winding = 0;

        //printf("Num bins: %d increments: %d\n", bins.size(), increments.size());
        for (int i = 0; i < bins.size(); i++) {
            Bin bin = bins[i];
            for (int j = bin.start; j < bin.end; j++) {
                Increment increment = increments[j];
                uint32_t x = increment.x % TILE_SIZE;
                uint32_t y = increment.y % TILE_SIZE;

                uint32_t p = y * TILE_SIZE + x;
                printf("%d %d\n", increment.x, increment.y);

                areas[p] += increment.area;
                heights[p] += increment.height;
            }


            if (i + 1 == bins.size() || bins[i + 1].tile_x != bin.tile_x || bins[i + 1].tile_y != bin.tile_y) {
                std::vector<uint8_t> tile(TILE_SIZE*TILE_SIZE);
                
                for(int y = 0; y < TILE_SIZE; y++) {
                    float accum = prev[y];
                    for(int x = 0; x < TILE_SIZE; x++) {
                        tile[y * TILE_SIZE + x] = fmin( fabs(accum + areas[y * TILE_SIZE + x]) * 256.0, 255);
                        accum += heights[y * TILE_SIZE + x];
                    }
                    next[y] = accum;
                }

                builder.Tile(bin.tile_x * TILE_SIZE, bin.tile_y * TILE_SIZE, tile);
                std::fill(areas.begin(), areas.end(), 0);
                std::fill(heights.begin(), heights.end(), 0);

                if(i + 1 < bins.size() && bins[i + 1].tile_y == bin.tile_y) {
                    std::copy(next.begin(), next.end(), prev.begin());
                } else {
                    std::fill(prev.begin(), prev.end(), 0);
                }

                std::fill(next.begin(), next.end(), 0);

                if (i + 1 < bins.size() && bins[i + 1].tile_y == bin.tile_y && bins[i + 1].tile_x > bin.tile_x + 1) {
                    while (tile_increments_i < tile_increments.size()) {
                        TileIncrement tile_increment = tile_increments[tile_increments_i];
                        if(tile_increment.tile_y > bin.tile_y) break;

                        if (tile_increment.tile_y == bin.tile_y && tile_increment.tile_x > bin.tile_x) {
                            break;
                        }
                        winding += tile_increment.sign;
                        tile_increments_i += 1;
                    }

                    if (winding != 0) {
                        auto width = bins[i+1].tile_x - bin.tile_x - 1;
                        builder.Span((bin.tile_x + 1) * TILE_SIZE, bin.tile_y * TILE_SIZE, width * TILE_SIZE );
                    }
                }
            }
        }
    }

    void LineTo(VPoint point) {
        if (point != last) {
            i16 x_dir = (point.fX - last.fX) < 0 ? -1 : 1;
            i16 y_dir = (point.fY - last.fY) < 0 ? -1 : 1;

            float dtdx = 1.0f / (point.fX - last.fX);
            float dtdy = 1.0f / (point.fY - last.fY);

            i16 x = floor(last.fX);
            i16 y = floor(last.fY);

            float row_t0 = 0.0f;
            float col_t0 = 0.0f;
    
            float row_t1, col_t1;
            if (last.fY == point.fY) {
                row_t1 = std::numeric_limits<float>::infinity();
            } else {
                float next_y = point.fY > last.fY ? y + 1.f :  y;
                row_t1 = fmin(dtdy * (next_y - last.fY), 1.0f);
            }

            if (last.fX == point.fX) {
                col_t1 = std::numeric_limits<float>::infinity();
            } else {
                float next_x = point.fX > last.fX ? x + 1.f :  x;
                col_t1 = fmin(dtdx * (next_x  - last.fX), 1.0);
            }   

            float x_step = fabs(dtdx);
            float y_step = fabs(dtdy);


            while (true) {
                float t0 = fmax(row_t0, col_t0);
                float t1 = fmin(row_t1, col_t1);
                VPoint p0 = (1.0f - t0) * last + t0 * point;
                VPoint p1 = (1.0f - t1) * last + t1 * point;

                float height = p1.fY - p0.fY;
                float right = x + 1.f;
                float area = 0.5f * height * ((right - p0.fX) + (right - p1.fX));
                increments.push_back({ x, y, area, height });

                if (row_t1 < col_t1) {
                    row_t0 = row_t1;
                    row_t1 = fmin((row_t1 + y_step), 1.0);
                    y += y_dir;
                } else {
                    col_t0 = col_t1;
                    col_t1 = fmin((col_t1 + x_step), 1.0);
                    x += x_dir;
                }

                if (row_t0 == 1.0 || col_t0 == 1.0) {
                    x = floor(point.fX);
                    y = floor(point.fY);
                }

                i16 tile_y = y / TILE_SIZE;
                if (tile_y != tile_y_prev) {
                    TileIncrement ti = { x / TILE_SIZE, fmin(tile_y_prev, tile_y), tile_y - tile_y_prev };
                    tile_increments.push_back(ti);
                    tile_y_prev = tile_y;
                }

                if(row_t0 == 1.0 || col_t0 == 1.0) {
                    break;
                }
            }
        }

        last.fX = point.fX;
        last.fY = point.fY;
    }   

    void MoveTo(VPoint point) { 
        if (last != first) {
            LineTo(first);
        }

        this->first = point;
        this->last = point;
        // TODO: Wrapping div euclid?
        this->tile_y_prev = floor(point.fY);
    }

private:
    std::vector<Increment> increments;
    std::vector<TileIncrement> tile_increments;
    VPoint first;
    VPoint last;
    i16 tile_y_prev;

};


class PathCmd {
    public:
    static std::vector<float> Flatten(std::vector<float>& commands, float tolerance) {
        
        int i = 0;
        VPoint last;
        std::vector<float> output;
        while (i < commands.size()) {
            PathVerb verb = static_cast<PathVerb>(commands[i]);
            float cmd, lx, ly;
            switch(verb) {
                case PathVerb::kClose:
                    output.push_back(commands[i++]);
                    break;
                case PathVerb::kMove:
                case PathVerb::kLine:
                    output.push_back(commands[i++]);
                    output.push_back(commands[i++]);
                    output.push_back(commands[i++]);
                    break;
                case PathVerb::kQuad: {
                    VPoint control = { commands[i+1], commands[i+2] };
                    VPoint point =   { commands[i+3], commands[i+4] };

                    float dt = sqrt((4.0 * tolerance) / (last - 2.0 * control + point).Length());
                    float t = 0.0;
                    while (t < 1.0) {
                        t = fmin(t + dt, 1.0);
                        VPoint p01  = last.Lerp(control, t);
                        VPoint p12  = control.Lerp(point, t);
                        VPoint line = p01.Lerp(p12, t);
                        output.push_back(static_cast<float>(PathVerb::kLine));
                        output.push_back(line.fX);
                        output.push_back(line.fY);
                    }

                    i += 7;
                }
            }
        }

        return std::move(output);
    } 
};

int pixel_sad8x8_simd(const uint8_t* px1, uint32_t stride1, const uint8_t* px2, uint32_t stride2) {
    __m128i v1, v2, res, sum;

    sum = _mm_setzero_si128();
    
    for (int i = 0; i < 8; i++) {
        v1 = _mm_loadl_epi64 ((const __m128i*)px1); 
        v2 = _mm_loadl_epi64 ((const __m128i*)px2); 
        res = _mm_sad_epu8(v1, v2); 
        sum = _mm_add_epi16(sum, res);

        px1 += stride1;
        px2 += stride2;
    }

    return _mm_extract_epi16(sum, 0); 
}

uint32_t pixel_sad8x8(const uint8_t* px1, uint32_t stride1, const uint8_t* px2, uint32_t stride2) {
    uint32_t sum = 0;
    for (int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            sum += abs(px1[j] - px2[j]);
        } 
        px1 += stride1;
        px2 += stride2;
    }

    return sum;
}


class VertexBuilder: public TileBuilder {
private:
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint8_t> atlas;
    uint16_t next_col;
    uint16_t next_row;
    uint32_t counter;

public:
    VertexBuilder(): next_col(1), next_row(0), counter(0) {
        atlas.resize(ATLAS_SIZE * ATLAS_SIZE);
        for(int i = 0; i < ATLAS_SIZE * ATLAS_SIZE; i++) {
            atlas[i] = 255;
        }
    }

    virtual void Tile(uint32_t x, uint32_t y, std::vector<uint8_t>&tile) {
        uint32_t base = vertices.size();
        
        uint32_t nr = next_row * TILE_SIZE * ATLAS_SIZE;
        uint32_t nc = next_col * TILE_SIZE;

    
        for (int row = 0; row < TILE_SIZE; row++) {
            uint32_t ra = row * ATLAS_SIZE;
            uint32_t rt = row * TILE_SIZE;

            for (int col = 0; col < TILE_SIZE; col++) {
                atlas[nr + ra + nc + col] = tile[rt + col];
            }
        }

        uint16_t u1 = next_col * TILE_SIZE;
        uint16_t u2 = (next_col+1) * TILE_SIZE;
        uint16_t v1 = next_row * TILE_SIZE;
        uint16_t v2 = (next_row+1) * TILE_SIZE;
        bool found = false;

        const int step = ATLAS_SIZE / TILE_SIZE;
        for(int i = 1; i < next_col + step * next_row; i++) {

            int sum = pixel_sad8x8_simd(tile.data(), 8, atlas.data() + i * TILE_SIZE, ATLAS_SIZE);
            if (sum < 100) {
                uint32_t r = i / step; 
                uint32_t c = i % step; 
                u1 = c * TILE_SIZE;
                u2 = (c+1) * TILE_SIZE;
                v1 = r * TILE_SIZE;
                v2 = (r+1) * TILE_SIZE;
                found = true;
                break;
            }
        }
        
        Vertex vert0 = { x, y, u1, v1, 255, 0, 0, 255  };
        Vertex vert1 = { x + TILE_SIZE, y, u2, v1, 255, 0, 0, 255  };
        Vertex vert2 = { x + TILE_SIZE, y + TILE_SIZE, u2, v2, 255, 0, 0, 255  };
        Vertex vert3 = { x, y + TILE_SIZE, u1, v2, 255, 0, 0, 255  };

        vertices.push_back(vert0);
        vertices.push_back(vert1);
        vertices.push_back(vert2);
        vertices.push_back(vert3);

        std::vector<uint32_t> new_indices = { base, base + 1, base + 2, base, base + 2, base + 3 };
        indices.insert(indices.end(), std::begin(new_indices), std::end(new_indices));

        if (!found) {
            next_col += 1;
            if (next_col == step) {
                next_col = 0;
                next_row += 1;
            }
        }
    };

    virtual void Span(int16_t x, int16_t y, int16_t width) {
        uint32_t base = vertices.size();

        Vertex vert0 = { x, y, 0, 0, 255, 0, 0, 255  };
        Vertex vert1 = { x + width, y, 0, 0, 255, 0, 0, 255  };
        Vertex vert2 = { x + width, y + TILE_SIZE, 0, 0, 255, 0, 0, 255  };
        Vertex vert3 = { x, y + TILE_SIZE, 0, 0, 255, 0, 0, 255  };

        vertices.push_back(vert0);
        vertices.push_back(vert1);
        vertices.push_back(vert2);
        vertices.push_back(vert3);

        std::vector<uint32_t> new_indices = { base, base + 1, base + 2, base, base + 2, base + 3 };
        indices.insert(indices.end(), std::begin(new_indices), std::end(new_indices));
    };

    virtual const uint8_t* GetAtlas() const {
        return atlas.data();
    }

    virtual const std::vector<uint32_t> GetIndices() const  {
        return indices;
    }

    virtual const std::vector<Vertex> GetVertices() const  {
        return vertices;
    }
};


const char* VERT = R"(#version 330
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
};)";

const char* FRAG = R"(#version 330
uniform sampler2D tex;

in vec2 v_uv;
in vec4 v_col;

out vec4 f_col;

void main() {
    //f_col =  vec4(1.0, 0.0, 0.0, 1.0); //v_col * vec4(texture(tex, v_uv).r);
    f_col =  vec4(1.0, 0.0, 0.0, 1.0) * vec4(1.0, 1.0, 1.0, texture(tex, v_uv).r);
}
;)";

int main() {
    std::cout << "compiles" << std::endl;
    const int width = 1280;
    const int height = 720;

    std::vector<float> commands;
    commands.push_back(static_cast<float>(PathVerb::kMove));
    commands.push_back(100);
    commands.push_back(200);

    commands.push_back(static_cast<float>(PathVerb::kLine));
    commands.push_back(300);
    commands.push_back(200);

    commands.push_back(static_cast<float>(PathVerb::kLine));
    commands.push_back(300);
    commands.push_back(400);

    commands.push_back(static_cast<float>(PathVerb::kLine));
    commands.push_back(100);
    commands.push_back(300);


    auto buildCommands = PathCmd::Flatten(commands, TOLERANCE);
    for(int i = 0; i < buildCommands.size(); i++) printf("-- %f\n", buildCommands[i]);
    buildCommands = Stroke(buildCommands, 1.f);

    Rasterizer r;
    for(int i = 0; i < buildCommands.size();) {
        PathVerb v =  static_cast<PathVerb>(buildCommands[i]);

        printf("%f %f %f\n", buildCommands[i], buildCommands[i+1], buildCommands[i+2]);
        switch(v) {
            case PathVerb::kLine:{
                r.LineTo({ buildCommands[i+1], buildCommands[i+2] });   
                i+=3;
                break;
            }
            case PathVerb::kMove: {
                r.MoveTo({ buildCommands[i+1], buildCommands[i+2] });
                i+=3;
                break;
            }

            case PathVerb::kClose: {
                i++;
                //r.Close();
                break;
            }
        }
    }
    VertexBuilder vb;

    std::cout << "finish" << std::endl;
    r.Finish(vb);

    std::cout << "finish2" << std::endl;

    if (!glfwInit()) {
        throw;
    }



    GLFWwindow* window = glfwCreateWindow(width, height, "Render view", NULL, NULL);
    glfwMakeContextCurrent(window);

  
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
		glfwTerminate();
		return -1;
	}
    std::cout << "Glad & GLFW Initialized" << std::endl;
    std::cout << glGetString(GL_VENDOR) << std::endl;
    std::cout << glGetString(GL_RENDERER) << std::endl;
    std::cout << glGetString(GL_VERSION) << std::endl;
    std::cout << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    GLuint vert = CompileShader(VERT, GL_VERTEX_SHADER);
    GLuint frag = CompileShader(FRAG, GL_FRAGMENT_SHADER);
    GLuint prog = LinkShaderProgram(vert, frag);
    std::cout << prog << std::endl;
    

    GLuint atlasTexture;
    glGenTextures(1, &atlasTexture);
    glBindTexture(GL_TEXTURE_2D, atlasTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, ATLAS_SIZE, ATLAS_SIZE, 0, GL_RED, GL_UNSIGNED_BYTE, vb.GetAtlas());

    std::string name = "texture";
    write_to_img(name, vb.GetAtlas(), ATLAS_SIZE, ATLAS_SIZE);

    GLuint vao, vbo, ibo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    std::vector<Vertex> vertices = vb.GetVertices();
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);

    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    std::vector<uint32_t> indices = vb.GetIndices();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(), GL_STREAM_DRAW);

    const int stride =  sizeof(Vertex);
    GLuint pos, uv, col;
    pos = glGetAttribLocation(prog, "pos");
    glEnableVertexAttribArray(pos);

    glVertexAttribPointer(pos, 2, GL_SHORT, GL_FALSE, stride, 0);

    uv = glGetAttribLocation(prog, "uv");
    glEnableVertexAttribArray(uv);
    glVertexAttribPointer(uv, 2, GL_UNSIGNED_SHORT, GL_FALSE, stride, reinterpret_cast<const void*>(offsetof(Vertex, uv)));

    col = glGetAttribLocation(prog, "col");
    glEnableVertexAttribArray(col);
    glVertexAttribPointer(col, 4, GL_UNSIGNED_BYTE, GL_FALSE, stride, reinterpret_cast<const void*>(offsetof(Vertex, color)));

    glUseProgram(prog);

    GLuint res, atlasSize, texUniform;
    res = glGetUniformLocation(prog, "res");
    glUniform2ui(res, width, height);

    atlasSize = glGetUniformLocation(prog, "atlas_size");
    glUniform2ui(atlasSize, ATLAS_SIZE, ATLAS_SIZE);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, atlasTexture);
    texUniform =  glGetUniformLocation(prog, "tex");
    glUniform1i(texUniform, 0);

    glViewport(0, 0, width,height);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    while(!glfwWindowShouldClose(window)) {
        // pass
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, (void*)0);
        glfwSwapBuffers(window);
    }
}