import pyglet
import pyglet.gl
import pyglet.clock

import pyshaders

import numpy as np

from math import pi, radians as to_radians, tan, cos, sin

from scipy.spatial.transform import Rotation


def v2(x=0, y=0):
    return np.array([x, y], dtype=float)


def v3(x=0, y=0, z=0):
    return np.array([x, y, z], dtype=float)


def v4(x=0, y=0, z=0, w=1):
    return np.array([x, y, z, w], dtype=float)


def v4_dir(v):
    return v4(v[0], v[1], v[2], 0)


def v4_pos(v):
    return v4(v[0], v[1], v[2], 1)


def as_affine(m3):
    return np.array([
        list(row) + [0] for row in m3
    ] + [[0, 0, 0, 1]])


def translation(by_v):
    x, y, z = by_v[0:3]
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=float)


def rotation(rotvec):
    return as_affine(Rotation.from_rotvec(rotvec).as_matrix())


idt = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)


def scale(x=1, y=1, z=1):
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ], dtype=float)


MAX_SCENE_SQUARES = 256

frag = """
#version 300 es

precision mediump float;
//precision mediump vec4;
//precision mediump mat4;

out lowp vec4 output_color;

uniform vec4 eye_pos;

uniform vec4 view_origin;
uniform vec4 view_x;
uniform vec4 view_y;

uniform mat4[MAX_SCENE_SQUARES] xy_square_transforms;
uniform lowp vec3[MAX_SCENE_SQUARES] xy_square_colors;
uniform int xy_squares_count;

void main() {
  vec4 location_on_view = view_origin + gl_FragCoord.x * view_x + gl_FragCoord.y * view_y;
  vec4 ray_pos = eye_pos;
  vec4 ray_dir = normalize(location_on_view - eye_pos);

  float nearest_distance = 100000.0;
  lowp vec3 nearest_color = vec3(0.8, 0.8, 1.0);

  for (int i = 0; i < xy_squares_count; ++i) {
    mat4 transform = xy_square_transforms[i];
    vec4 projected_ray_pos = transform * ray_pos;
    vec4 projected_ray_dir = transform * ray_dir;

    if (projected_ray_dir.z != 0.0) {
      float dist_to_xy_plane = -projected_ray_pos.z / projected_ray_dir.z;
      if (nearest_distance > dist_to_xy_plane && dist_to_xy_plane > 0.01) {
        vec4 hit_on_xy_plane = projected_ray_pos + projected_ray_dir * dist_to_xy_plane;
        if (hit_on_xy_plane.x >= -1.0 && hit_on_xy_plane.x <= 1.0 && hit_on_xy_plane.y >= -1.0 && hit_on_xy_plane.y <= 1.0) {
          nearest_distance = dist_to_xy_plane;
          nearest_color = xy_square_colors[i];
        }
      }
    }
  }

  output_color = vec4(nearest_color, 1);
}
""".replace("MAX_SCENE_SQUARES", str(MAX_SCENE_SQUARES))

vert = """
#version 300 es
layout(location = 0)in vec2 vert;

void main() {
  gl_Position = vec4(vert, 1, 1);
}
"""

from copy import deepcopy


class Square:
    def __init__(self, position=None, normal=None, extents=None, color=None):
        self.position = position if position is not None else v3()
        self.normal = normal if normal is not None else v3(0, 0, 1)
        self.extents = extents if extents is not None else v2(1, 1)
        self.color = color if color is not None else v3(1.0, 0.0, 0.0)

    def has_same_transform_as(self, o) -> bool:
        return o is not None and \
               np.array_equal(self.position, o.position) and \
               np.array_equal(self.normal, o.normal) and \
               np.array_equal(self.extents, o.extents)


class GPUSquare:
    def __init__(self, square):
        self.square = square
        self._cached_square = None
        self._cached_transform = None

    def compute_transform(self):
        if self.square.has_same_transform_as(self._cached_square):
            return self._cached_transform

        # subroutines from https://stackoverflow.com/a/13849249

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        self._cached_square = deepcopy(self.square)
        mat_scale = scale(1 / self.square.extents[0], 1 / self.square.extents[1], 1)
        mat_rotate = rotation(np.cross(self.square.normal, v3(z=1)) *
                                          angle_between(v3(z=1), self.square.normal))
        mat_translate = translation(-self.square.position)
        self._cached_transform = np.matmul(np.matmul(mat_scale, mat_rotate), mat_translate)
        return self._cached_transform


class Camera:

    def __init__(self):
        self.position = v3(0, 0, 10)
        self.direction = v3(0, 0, -1)
        self.up = v3(0, 1, 0)
        self.viewport_size = v2(80, 60)
        self.screen_resolution = v2(800, 600)
        self.fov = 60

        self._cached_view_origin = v4()
        self._cached_view_x = v4(w=0)
        self._cached_view_y = v4(w=0)

    def update_view_basis(self,
                          viewport_size=None,
                          screen_resolution=None):
        self.viewport_size = viewport_size or self.viewport_size
        self.screen_resolution = screen_resolution or self.screen_resolution

        view_scale_x = self.viewport_size[0] / self.screen_resolution[0]
        view_scale_y = self.viewport_size[1] / self.screen_resolution[1]

        basis_x = np.cross(self.direction, self.up)
        basis_x = basis_x / np.linalg.norm(basis_x)
        basis_y = Rotation.from_rotvec(self.direction * (-pi / 2)).apply(basis_x)

        self._cached_view_x = v4_dir(basis_x * view_scale_x)
        self._cached_view_y = v4_dir(basis_y * view_scale_y)
        self._cached_view_origin = v4_pos(self.position - basis_x * self.viewport_size[0] * 0.5 - \
            basis_y * self.viewport_size[1] * 0.5)

    def update_shader_uniforms(self, shader):
        shader.uniforms.eye_pos = tuple(v4_pos(self.position - self.direction * \
                                               tan(to_radians(self.fov)) * 0.5 * self.viewport_size[0]))
        shader.uniforms.view_origin = tuple(self._cached_view_origin)
        shader.uniforms.view_x = tuple(self._cached_view_x)
        shader.uniforms.view_y = tuple(self._cached_view_y)


class Scene:
    def __init__(self):
        self._gpu_squares = []

    def add_square(self, square):
        self._gpu_squares.append(GPUSquare(square))

    def delete_square(self, square):
        self._gpu_squares = [s for s in self._gpu_squares if s.square is not square]

    def update_shader_uniforms(self, shader):
        gpu_squares = self._gpu_squares
        if len(gpu_squares) > MAX_SCENE_SQUARES:
            gpu_squares = gpu_squares[:MAX_SCENE_SQUARES]
        shader.uniforms.xy_squares_count = len(gpu_squares)
        shader.uniforms.xy_square_transforms = tuple(tuple(tuple(r for r in s.compute_transform())) for s in gpu_squares)
        shader.uniforms.xy_square_colors = tuple(tuple(s.square.color) for s in gpu_squares)


class KlossRoyaleWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(KlossRoyaleWindow, self).__init__(**kwargs)

        self.shader = pyshaders.from_string(vert, frag)
        self.shader.use()

        self.screen_vertices = pyglet.graphics.vertex_list(6, ('v2f', (-1.0, 1.0, 1.0, -1, -1, -1,
                                                                       -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)))

        self.scene = Scene()
        self.scene.add_square(Square(position=v3(20, -20, 0), normal=v3(0, 0, 1), extents=v2(5, 5)))
        self.scene.add_square(Square(position=v3(20, 20, 0), normal=v3(0, 0, 1), extents=v2(5, 5)))
        self.scene.add_square(Square(position=v3(-20, 20, 0), normal=v3(0, 0, 1), extents=v2(5, 5)))
        self.scene.add_square(Square(position=v3(-20, -20, 0), normal=v3(0, 0, 1), extents=v2(5, 5)))
        self.camera = Camera()
        self.camera.position = v3(10, 10, 10)
        self.camera.direction = -self.camera.position / np.linalg.norm(self.camera.position)
        self.camera.up = v3(0, 0, 1)
        self.camera.update_view_basis()
        self.cur_angle = 0
        pyglet.clock.schedule(self.update, .1)

    def update(self, dt, _desired_dt):
        self.cur_angle += 2 * pi * dt / 5
        self.camera.position = 10 * v3(cos(self.cur_angle) * cos(to_radians(45)),
                                       sin(self.cur_angle) * cos(to_radians(45)),
                                       sin(to_radians(45)))
        self.camera.direction = -self.camera.position / np.linalg.norm(self.camera.position)
        self.camera.up = v3(0, 0, 1)
        self.camera.update_view_basis()

    def on_draw(self):
        self.clear()
        self.camera.update_shader_uniforms(self.shader)
        self.scene.update_shader_uniforms(self.shader)
        self.screen_vertices.draw(pyglet.gl.GL_TRIANGLES)

    def on_key_press(self, symbol, modifiers):
        pass


if __name__ == "__main__":
    # pyshaders.transpose_matrices(False)
    window = KlossRoyaleWindow(visible=True, width=800, height=600, resizable=True)
    pyglet.app.run()
