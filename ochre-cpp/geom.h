#pragma once

#include <iostream>
#include <cmath>
#include <GL/glew.h>
#define VScalar float

struct VPoint {
    VScalar fX;
    VScalar fY;
    //VPoint(const VPoint& other) {
    //    fX = other.fX;
    //    fY = other.fY;
    //}

    static constexpr VPoint Make(VScalar x, VScalar y) {
        return {x, y};
    }

    /** Returns x-axis value of VPoint or vector.

        @return  fX
    */
    constexpr VScalar x() const { return fX; }

    /** Returns y-axis value of VPoint or vector.

        @return  fY
    */
    constexpr VScalar y() const { return fY; }
    bool isZero() const { return (0 == fX) & (0 == fY); }
    VPoint transform(VScalar* t) {
        VScalar x = fX * t[0] + fY * t[2] + t[4];
        VScalar y = fX * t[1] + fY * t[3] + t[5];
        return { x, y };
    }

    VPoint inverseTransform(VScalar* t) {
        // fX * t0 + fY * t2 = (x - t4)
        // fX * t1 + fY * t3 = (y - t5)
        // https://stackoverflow.com/questions/19619248/solve-system-of-two-equations-with-two-unknowns

        VScalar a = t[0];
        VScalar d = t[3];

        VScalar b = t[2];
        VScalar c = t[1];

        VScalar e = fX - t[4];
        VScalar f = fY - t[5];

        VScalar determinant = a*d - b*c;
        if (determinant != 0) {
            VScalar x = (e*d - b*f) / determinant;
            VScalar y = (a*f - e*c) / determinant;
            return { x, y };
        } 

        return { 0, 0 };
    }

    VPoint Lerp(const VPoint& other, float t) {
        return (1.0 - t) * (*this) + t * other;
    }
 
    VPoint scale(VScalar* transform) {
        return { fX / transform[0], fY / transform[3] };
    }
    
    VScalar Dot(const VPoint& other) {
        return this->fX * other.fX + this->fY * other.fY;
    }

    VScalar Cross(const VPoint& other) {
        return this->fX * other.fY - this->fY * other.fX;
    }

    VScalar Length() {
        return sqrt(this->Dot(*this));
    }

    VScalar Distance(const VPoint& other) {
        return (other - *this).Length();
    }

    friend VPoint operator*(const VPoint& a, float scalar) {
        return {a.fX * scalar, a.fY * scalar};
    } 

    friend VPoint operator*(float scalar, const VPoint& a) {
        return {a.fX * scalar, a.fY * scalar};
    } 

    VPoint Normalized() {
        return  (*this) * (1.0f / this->Length());
    }

    friend VPoint operator+(const VPoint& a, VScalar scalar) {
        return {a.fX + scalar, a.fY + scalar};
    } 

    friend VPoint operator-(const VPoint& a, VScalar scalar) {
        return {a.fX + scalar, a.fY + scalar};
    } 

    friend VPoint operator*(const VPoint& a, const VPoint& b) {
        return {a.fX * b.fX, a.fY * b.fY};
    } 

    friend VPoint operator/(const VPoint& a, const VPoint& b) {
        return {a.fX / b.fX, a.fY / b.fY};
    }    

    friend VPoint operator+(const VPoint& a, const VPoint& b) {
        return {a.fX + b.fX, a.fY + b.fY};
    }

    friend VPoint operator-(const VPoint& a, const VPoint& b) {
        return {a.fX - b.fX, a.fY - b.fY};
    }

    friend bool operator==(const VPoint& a, const VPoint& b) {
        return a.fX == b.fX && a.fY == b.fY;
    }

    friend bool operator!=(const VPoint& a, const VPoint& b) {
        return !(a == b);
    }
};



GLuint CompileShader(const char* src, GLuint type) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);
  GLint success;
  GLchar infoLog[512];

  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(shader, 512, NULL, infoLog);
    std::cerr << "Shader Compilation failed\n" << infoLog << std::endl;
    return -1;
  }

  return shader;
}

GLuint LinkShaderProgram(GLuint vertexShader, GLuint fragmentShader) {
  // Link  shaders
  GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);

  GLint success;
  GLchar infoLog[512];
  // Check for linking errors
  glGetProgramiv(program, GL_LINK_STATUS, &success);

  if (!success) {
    glGetProgramInfoLog(program, 512, NULL, infoLog);
    std::cerr << "ERROR::LINKING_FAILED\n" << infoLog << std::endl;
    return -1;
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return program;
}

