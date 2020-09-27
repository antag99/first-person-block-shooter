import pyglet
import pyglet.gl
import pyglet.clock
from pyglet.gl import *
from pyglet.window import key

import pyshaders

import numpy as np

from math import pi, radians as to_radians, tan, cos, sin, floor, sqrt, ceil, copysign

from scipy.spatial.transform import Rotation
from random import Random


def v2(x=0.0, y=0.0):
    return np.array([x, y], dtype=float)


def v3(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z], dtype=float)


def v4(x=0.0, y=0.0, z=0.0, w=1.0):
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


MAX_SCENE_SQUARES = 512

frag = """
#version 300 es

in lowp vec4 point_color;
out lowp vec4 output_color;

void main() {
  output_color = point_color;
}
"""

vert = """
#version 300 es

layout(location = 0) in vec2 pixel_coordinates;

precision mediump float;

out lowp vec4 point_color;

uniform vec4 eye_pos;

uniform vec4 view_origin;
uniform vec4 view_x;
uniform vec4 view_y;

uniform float point_size;
uniform mat4[MAX_SCENE_SQUARES] xy_square_transforms;
uniform lowp vec3[MAX_SCENE_SQUARES] xy_square_colors;
uniform int xy_squares_count;

void main() {
  vec4 location_on_view = view_origin + pixel_coordinates.x * view_x + pixel_coordinates.y * view_y;
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

  gl_Position = vec4(pixel_coordinates, 1, 1);
  gl_PointSize = point_size;
  float fog = 450.0;
  vec3 sampled_color = nearest_color * clamp(1.0 - (nearest_distance / fog) * (nearest_distance / fog), 0.0, 1.0);
  point_color = vec4(sqrt(sampled_color.x), sqrt(sampled_color.y), sqrt(sampled_color.z), 1);  // gamma correction
}
""".replace("MAX_SCENE_SQUARES", str(MAX_SCENE_SQUARES))

from copy import deepcopy


class Square:
    def __init__(self, position=None, normal=None, extents=None, color=None):
        self.position = position if position is not None else v3()
        self.normal = normal if normal is not None else v3(0, 0, 1)
        self.extents = extents if extents is not None else v2(1, 1)
        assert self.extents[0] > 0 and self.extents[1] > 0
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
                                          angle_between(self.square.normal, v3(z=1)))
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
        self.viewport_size = viewport_size if viewport_size is not None else self.viewport_size
        self.screen_resolution = screen_resolution if screen_resolution is not None else self.screen_resolution

        view_scale_x = self.viewport_size[0] / 2.0
        view_scale_y = self.viewport_size[1] / 2.0

        basis_x = np.cross(self.direction, self.up)
        basis_x = basis_x / np.linalg.norm(basis_x)
        basis_y = Rotation.from_rotvec(self.direction * (-pi / 2)).apply(basis_x)

        self._cached_view_x = v4_dir(basis_x * view_scale_x)
        self._cached_view_y = v4_dir(basis_y * view_scale_y)
        self._cached_view_origin = v4_pos(self.position + self.direction * tan(to_radians(self.fov)) * 0.5 * self.viewport_size[0])

    def update_shader_uniforms(self, shader):
        shader.uniforms.eye_pos = tuple(v4_pos(self.position))
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
            print("WARNING: max squares exceeded!")
            gpu_squares = gpu_squares[:MAX_SCENE_SQUARES]
        shader.uniforms.xy_squares_count = len(gpu_squares)
        shader.uniforms.xy_square_transforms = tuple(tuple(tuple(r for r in s.compute_transform())) for s in gpu_squares)
        shader.uniforms.xy_square_colors = tuple(tuple(s.square.color) for s in gpu_squares)


class Room:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direct_connections = []
        self.adjacent_rooms = []

    def compute_reachable(self, test, accumulate):
        if test(self):
            return
        accumulate(self)
        for connected_room in self.direct_connections:
            connected_room.compute_reachable(test, accumulate)


class Maze:

    def __init__(self):
        self.maze_size = 3200.0
        self.room_size = self.maze_size / 8.0

        self.port_width = self.room_size * 0.2
        self.port_height = 120.0
        self.wall_height = 200

        self.rooms = []
        self.rooms_by_coords = dict()

        for i in range(0, 8):
            for j in range(0, 8):
                room = Room(i, j)
                self.rooms.append(room)
                self.rooms_by_coords[i, j] = room

        for i in range(0, 8):
            for j in range(0, 8):
                room = self.rooms_by_coords[i, j]
                room.adjacent_rooms = [self.rooms_by_coords[i + ioff, j + joff]
                                       for ioff in [-1, 0, 1]
                                       for joff in [-1, 0, 1]
                                       if i + ioff in range(0, 8)
                                       and j + joff in range(0, 8)
                                       and abs(ioff) + abs(joff) == 1]

        random = Random()

        not_completely_connected = list(self.rooms)

        while len(not_completely_connected) > 0:
            room = random.choice(not_completely_connected)
            reachable = list()
            room.compute_reachable(reachable.__contains__, reachable.append)
            non_reachable_adjacent = [adjacent for adjacent in room.adjacent_rooms if adjacent not in reachable]
            if len(non_reachable_adjacent) > 0:
                room_connected = random.choice(non_reachable_adjacent)
                room.direct_connections.append(room_connected)
                room_connected.direct_connections.append(room)
            else:
                not_completely_connected.remove(room)

    def hit_ray_on_wall(self, pos, dir):
        pos_in_room_coords = pos / self.room_size
        room_x = floor(pos_in_room_coords[0])
        room_y = floor(pos_in_room_coords[1])

        dist_x = None

        if dir[0] > 0:
            dist_x = ((ceil(pos_in_room_coords[0]) - pos_in_room_coords[0]) * self.room_size - 0.05) / dir[0]
        elif dir[0] < 0:
            dist_x = ((floor(pos_in_room_coords[0]) - pos_in_room_coords[0]) * self.room_size + 0.05) / dir[0]

        dist_y = None

        if dir[1] > 0:
            dist_y = ((ceil(pos_in_room_coords[1]) - pos_in_room_coords[1]) * self.room_size - 0.05) / dir[1]
        elif dir[1] < 0:
            dist_y = ((floor(pos_in_room_coords[1]) - pos_in_room_coords[1]) * self.room_size + 0.05) / dir[1]

        room_offset_x = copysign(1.0, dir[0])
        room_offset_y = copysign(1.0, dir[1])

        assert not (dist_x is None and dist_y is None)

        min_dist = min(dist_x or 100000, dist_y or 100000)

        hit_coord = pos + dir * min_dist

        hit_port = False

        if min_dist == dist_x:
            hit_mod_y = (hit_coord[1] % self.room_size) - 0.5 * self.room_size
            hit_port = abs(hit_mod_y) < self.port_width * 0.5 and \
                       self.is_connected(room_x, room_y, room_x + room_offset_x, room_y)
        elif min_dist == dist_y:
            hit_mod_x = (hit_coord[0] % self.room_size) - 0.5 * self.room_size
            hit_port = abs(hit_mod_x) < self.port_width * 0.5 and \
                        self.is_connected(room_x, room_y, room_x, room_y + room_offset_y)

        if hit_port:  # check collision in the next room...
            return min_dist + copysign(0.2, min_dist) + self.hit_ray_on_wall(pos + hit_coord + dir * copysign(0.2, min_dist), dir)
        else:
            return min_dist

    def is_connected(self, fx, fy, tx, ty):
        if (fx, fy) not in self.rooms_by_coords:
            return False

        return any(r.x == tx and r.y == ty
                   for r in self.rooms_by_coords[fx, fy].direct_connections)

    def to_squares(self):

        squares = [Square(position=v3(self.maze_size, self.maze_size) * 0.5,
                          extents=v2(self.maze_size, self.maze_size) * 0.5, color=v3(0.2, 0.2, 0.2))]

        left_wall_begin_pos = v3(0, 0, 0)
        for i in range(0, 9):
            for j in range(0, 9):
                if j == 8:
                    left_wall_end_pos = v3(i, j) * self.room_size + v3(0, 0, self.wall_height)
                    new_begin_pos = v3(i + 1, 0, 0) * self.room_size
                else:
                    left_wall_end_pos = v3(i, j + 0.5) * self.room_size + v3(0, -self.port_width * 0.5, self.wall_height)
                    new_begin_pos = v3(i, j + 0.5) * self.room_size + v3(0, self.port_width * 0.5)

                if self.is_connected(i, j, i - 1, j) or j == 8:
                    squares.append(Square(
                        position=(left_wall_end_pos + left_wall_begin_pos) * 0.5,
                        extents=v2(left_wall_end_pos[2] - left_wall_begin_pos[2],
                                   left_wall_end_pos[1] - left_wall_begin_pos[1]) * 0.5,
                        normal=v3(1, 0, 0)))
                    if self.is_connected(i, j, i - 1, j):
                        squares.append(Square(
                            position=(new_begin_pos + left_wall_end_pos) * 0.5 + v3(0, 0, self.wall_height*0.5 - (self.wall_height - self.port_height) * 0.5),
                            extents=v2((self.wall_height - self.port_height) * 0.5,
                                       self.port_width * 0.5),
                            normal=v3(1, 0, 0)))
                    left_wall_begin_pos = new_begin_pos

        bottom_wall_begin_pos = v3(0, 0, 0)
        for j in range(0, 9):
            for i in range(0, 9):
                if i == 8:
                    bottom_wall_end_pos = v3(i, j) * self.room_size + v3(0, 0, self.wall_height)
                    new_begin_pos = v3(0, j + 1, 0) * self.room_size
                else:
                    bottom_wall_end_pos = v3(i + 0.4, j) * self.room_size + v3(0, 0, self.wall_height)
                    new_begin_pos = v3(i + 0.6, j, 0) * self.room_size

                has_port = self.is_connected(i, j, i, j - 1)

                if has_port or i == 8:
                    squares.append(Square(
                        position=(bottom_wall_end_pos + bottom_wall_begin_pos) * 0.5,
                        extents=v2(bottom_wall_end_pos[0] - bottom_wall_begin_pos[0],
                                   bottom_wall_end_pos[2] - bottom_wall_begin_pos[2]) * 0.5,
                        normal=v3(0, 1, 0)))
                    if self.is_connected(i, j, i, j - 1):
                        squares.append(Square(
                            position=(new_begin_pos + bottom_wall_end_pos) * 0.5 + v3(0, 0, self.wall_height*0.5 - (self.wall_height - self.port_height) * 0.5),
                            extents=v2((self.wall_height - self.port_height) * 0.5,
                                       self.port_width * 0.5),
                            normal=v3(0, 1, 0)))
                    bottom_wall_begin_pos = new_begin_pos
        return squares


class KlossRoyaleWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(KlossRoyaleWindow, self).__init__(**kwargs)

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

        self.shader = pyshaders.from_string(vert, frag)
        self.shader.use()

        self.pixel_size = 1
        self.pixel_vertices = None
        self._build_vertex_list()

        self.maze = Maze()
        self.scene = Scene()

        for s in self.maze.to_squares():
            self.scene.add_square(s)

        self.controls = [
            (key.W, v3(1, 0, 0)),
            (key.A, v3(0, 1, 0)),
            (key.S, v3(-1, 0, 0)),
            (key.D, v3(0, -1, 0)),
        ]
        self.yaw = 0
        self.pitch = 0
        self.eye_pos = v3(20, 20, 80)
        self.mouse_sensitivity = 2 * pi / 320
        self.active_keys = set()

        self.camera = Camera()
        self._update_camera()
        pyglet.clock.schedule(self.update, .1)

    def _compute_look_at_direction(self):
        return v3(cos(self.yaw) * cos(self.pitch), sin(self.yaw) * cos(self.pitch), sin(self.pitch))

    def _update_camera(self):
        self.camera.position = self.eye_pos
        self.camera.direction = self._compute_look_at_direction()
        self.camera.up = v3(0, 0, 1)
        self.camera.update_view_basis()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        area = 80 * 60
        scale = sqrt(area / (width * height))
        self.camera.update_view_basis(v2(width * scale, height * scale), v2(width, height))
        self._build_vertex_list()

    def on_activate(self):
        self._build_vertex_list()

    def _build_vertex_list(self):
        width, height = self.get_size()
        pixel_count = 16000
        pixel_area = width * height / pixel_count
        pixel_size = floor(sqrt(pixel_area))
        pixel_count_x = ceil(width / pixel_size)
        pixel_count_y = ceil(height / pixel_size)
        pixel_begin_x = floor((width - pixel_count_x * pixel_size) * 0.5)
        pixel_begin_y = floor((height - pixel_count_y * pixel_size) * 0.5)
        vertices = []
        for i in range(0, pixel_count_x):
            for j in range(0, pixel_count_y):
                vertices += [(pixel_begin_x + pixel_size * i * 2.0) / width - 1.0,
                             (pixel_begin_y + pixel_size * j * 2.0) / height - 1.0]
        self.pixel_vertices = pyglet.graphics.vertex_list(len(vertices) // 2, ('v2f', tuple(vertices)))
        self.pixel_size = pixel_size

    def update(self, dt, _desired_dt):
        move_direction = sum((control[1] for control in self.controls if control[0] in self.active_keys), v3())
        if np.array_equal(move_direction, v3()):
            return

        source = self.eye_pos
        dir = Rotation.from_euler('z', self.yaw).apply(move_direction)
        dist = 400 * dt

        max_dist = self.maze.hit_ray_on_wall(source, dir)

        if dist > max_dist:
            dist = max_dist - 0.01

        self.eye_pos += dist * dir
        self._update_camera()

    def on_draw(self):
        self.clear()
        self.shader.uniforms.point_size = self.pixel_size
        self.camera.update_shader_uniforms(self.shader)
        self.scene.update_shader_uniforms(self.shader)
        self.pixel_vertices.draw(GL_POINTS)

    def on_key_press(self, symbol, modifiers):
        self.active_keys.add(symbol)

    def on_key_release(self, symbol, modifiers):
        self.active_keys.discard(symbol)

    def on_mouse_motion(self, x, y, dx, dy):
        self.yaw -= self.mouse_sensitivity * dx
        self.yaw %= 2 * pi
        self.pitch += self.mouse_sensitivity * dy
        self.pitch = max(-pi / 2.0, min(pi / 2.0, self.pitch))
        self._update_camera()


if __name__ == "__main__":
    window = KlossRoyaleWindow(visible=True, resizable=True)
    window.set_mouse_visible(False)
    window.set_exclusive_mouse(True)
    # window.set_fullscreen(True)
    pyglet.app.run()
