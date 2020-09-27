import operator
from functools import reduce

import pyglet
import pyglet.gl
import pyglet.clock
from pyglet.gl import *
from pyglet.window import key
from enum import Enum
from collections import namedtuple

import pyshaders

import numpy as np

from math import pi, radians as to_radians, tan, cos, sin, floor, sqrt, ceil, copysign, atan2

from scipy.spatial.transform import Rotation
from random import Random

from copy import deepcopy, copy

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


def rotation_from_rotvec(rotvec):
    return as_affine(Rotation.from_rotvec(rotvec).as_matrix())


def rotation_from_euler(x=0.0, y=0.0, z=0.0):
    return as_affine(Rotation.from_euler('xyz', (x, y, z)).as_matrix())


idt = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)


def scale(x=1.0, y=1.0, z=1.0):
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ], dtype=float)


MAX_SCENE_SQUARES = 512

frag = """
#version 300 es

precision mediump float;

uniform vec2 crosshair_location;
uniform float crosshair_size;
uniform float crosshair_t;

in lowp vec4 point_color;
out lowp vec4 output_color;

void main() {
  vec2 diff = crosshair_location - gl_FragCoord.xy;
  if (abs(diff.x) <= crosshair_size * 0.5 && abs(diff.y) <= crosshair_size * 0.5 &&
      (abs(diff.x) <= crosshair_t*0.5 || abs(diff.y) <= crosshair_t*0.5)) {
    output_color = vec4(1, 1, 1, 1);
  } else {
    output_color = point_color;
  }
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
        if (hit_on_xy_plane.x >= 0.0 && hit_on_xy_plane.x <= 1.0 && hit_on_xy_plane.y >= 0.0 && hit_on_xy_plane.y <= 1.0) {
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


Square = namedtuple('Square', 'transform color')


class Camera:

    def __init__(self):
        self.position = v3(0, 0, 10)
        self.direction = v3(0, 0, -1)
        self.up = v3(0, 1, 0)
        self.viewport_size = v2(80, 60)
        self.screen_resolution = v2(800, 600)
        self.fov = 120

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
        self._cached_view_origin = v4_pos(self.position + self.direction * self.viewport_size[0] * 0.5 / tan(to_radians(self.fov * 0.5)))

    def update_shader_uniforms(self, shader):
        shader.uniforms.eye_pos = tuple(v4_pos(self.position))
        shader.uniforms.view_origin = tuple(self._cached_view_origin)
        shader.uniforms.view_x = tuple(self._cached_view_x)
        shader.uniforms.view_y = tuple(self._cached_view_y)


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


class Entity:
    def __init__(self, game_state):
        self.game_state = game_state

        self.pos = v3()
        self.eye_offset = v3(0, 0, 80)
        self.eye_pos = v3()
        self.look_dir = v3(1.0, 0, 0)
        self.movement_speed = 400.0
        self.extent = v3()

    def update(self, dt):
        self.eye_pos = self.pos + self.eye_offset

    def to_squares(self):
        return []


class Maze(Entity):

    def __init__(self, game_state):
        super().__init__(game_state)
        self.maze_size = 3200.0
        self.room_size = self.maze_size / 8.0

        self.port_width = self.room_size * 0.2
        self.port_height = 120.0
        self.wall_height = 800

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

        self._cached_squares = None

    def find_room_with_pos(self, pos):
        pos_in_room_coords = pos / self.room_size
        room_x = floor(pos_in_room_coords[0])
        room_y = floor(pos_in_room_coords[1])
        return self.rooms_by_coords[room_x, room_y]

    @classmethod
    def find_path_between_rooms(self, from_room, to_room):
        visited = set()

        # NOTE: there is only *one* path between two rooms, due to the way the maze is generated.
        def visit(room, accumulated_path):
            if room in visited:
                return None

            visited.add(room)

            if room == to_room:
                return accumulated_path

            for connected in room.direct_connections:
                ret = visit(connected, accumulated_path + [connected])
                if ret:
                    return ret

        return visit(from_room, [])

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
            next_pos = pos + hit_coord + dir * copysign(0.2, min_dist)
            return np.linalg.norm(next_pos - pos) + self.hit_ray_on_wall(next_pos, dir)
        else:
            return min_dist

    def is_connected(self, fx, fy, tx, ty):
        if (fx, fy) not in self.rooms_by_coords:
            return False

        return any(r.x == tx and r.y == ty
                   for r in self.rooms_by_coords[fx, fy].direct_connections)

    def to_squares(self):
        if self._cached_squares is None:
            self._cached_squares = self.generate_squares()
        return self._cached_squares

    def generate_squares(self):
        squares = [Square(transform=scale(1/self.maze_size, 1/self.maze_size, 1), color=v3(0.2, 0.2, 0.2))]

        def square_between(pos0, pos1):
            delta = pos1 - pos0
            delta_xy = delta * v3(1, 1, 0)
            delta_z = delta * v3(0, 0, 1)
            width = np.linalg.norm(delta_xy)
            height = np.linalg.norm(delta_z)
            rot_z = atan2(delta_xy[1], delta_xy[0])

            mat_translate = translation(-pos0)
            mat_rotate_to_xz = rotation_from_euler(z=-rot_z)
            mat_rotate_to_xy = rotation_from_euler(x=-pi/2)
            mat_scale = scale(1/width, 1/height)

            return Square(transform=np.matmul(mat_scale, np.matmul(mat_rotate_to_xy, np.matmul(mat_rotate_to_xz, mat_translate))), color=v3(1, 0, 0))

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
                    squares.append(square_between(left_wall_begin_pos, left_wall_end_pos))
                    left_wall_begin_pos = new_begin_pos
                    if self.is_connected(i, j, i - 1, j):
                        squares.append(square_between(left_wall_end_pos + v3(z=self.port_height-self.wall_height), new_begin_pos + v3(z=self.wall_height)))

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
                    squares.append(square_between(bottom_wall_begin_pos, bottom_wall_end_pos))
                    if self.is_connected(i, j, i, j - 1):
                        squares.append(square_between(bottom_wall_end_pos + v3(z=self.port_height-self.wall_height), new_begin_pos + v3(z=self.wall_height)))
                    bottom_wall_begin_pos = new_begin_pos
        return squares


class Enemy(Entity):
    def __init__(self, game_state):
        super().__init__(game_state)
        self.random = Random()
        self.width = 40.0
        self.height = 100.0
        self.extent = v3(self.width, self.width, self.height)
        self.color = v3(0, 1, 0)
        self.current_state = self.IdleState(self)
        self.normal_speed = 200
        self.chasing_speed = 300
        self.movement_speed = self.normal_speed
        self.sight_distance = 400
        self.time_from_sight_to_shot = 1.2
        self.target = self.Target(self, game_state.player)

    def to_squares(self):
        transforms = []
        for i in range(0, 4):
            mat_translate = translation(-self.pos)
            mat_rotate_0 = rotation_from_euler(z=pi*i/2)
            mat_translate_2 = translation(v3(-self.width * 0.5, self.width*0.5))
            mat_rotate = rotation_from_euler(y=pi/2)
            mat_scale = scale(1.0/self.height, 1.0/self.width, 1.0)
            transforms.append(np.matmul(mat_scale, np.matmul(mat_rotate, np.matmul(mat_translate_2, np.matmul(mat_rotate_0, mat_translate)))))
        return [Square(transform, self.color) for transform in transforms]

    class IdleState:
        def __init__(self, entity):
            self.entity = entity

        def __call__(self, dt):
            self.entity.movement_speed = self.entity.normal_speed
            current_room = self.entity.game_state.maze.find_room_with_pos(self.entity.pos)
            while True:
                random_room = self.entity.random.choice(self.entity.game_state.maze.rooms)
                if random_room is not current_room:
                    path_between_rooms = self.entity.game_state.maze.find_path_between_rooms(current_room, random_room)
                    resulting_path = [v3(room.x + 0.5, room.y + 0.5, 0) * self.entity.game_state.maze.room_size
                                      for room in [current_room] + path_between_rooms]
                    return self.entity.WalkState(self.entity, resulting_path)

    class DyingState:
        def __init__(self, entity):
            self.entity = entity
            self.anim_counter = 0.0

        def __call__(self, dt):
            self.anim_counter += dt
            if self.anim_counter > 1.0:
                self.entity.game_state.entities.remove(self.entity)
                self.entity.game_state.enemies_eliminated += 1

            if self.anim_counter < 0.33333:
                self.entity.color = v3(1, 0, 0)
            elif self.anim_counter < 0.66666:
                self.entity.color = v3(0, 1, 0)
            else:
                self.entity.color = v3(1, 0, 0)

            return self

    class WalkState:
        def __init__(self,
                     entity,
                     position_queue):
            self.entity = entity
            self.position_queue = position_queue

        def __call__(self, dt):
            if self.entity.target.can_spot_target:
                return self.entity.TargetingPlayerState(self.entity)

            if len(self.position_queue) == 0:
                return self.entity.IdleState(self.entity)

            to_pos = self.position_queue[0]

            current_delta = to_pos - self.entity.pos
            current_dist = np.linalg.norm(current_delta)

            walk_dist = self.entity.movement_speed * dt
            if walk_dist < current_dist:
                self.entity.pos += (current_delta / current_dist) * walk_dist
            else:
                self.entity.pos = to_pos
                self.position_queue = self.position_queue[1:]
                return self(dt - current_dist / self.entity.movement_speed)

            return self

    class TargetingPlayerState:
        def __init__(self, entity):
            self.entity = entity
            self.aiming_time = 0.0

        def __call__(self, dt):
            if self.entity.target.can_spot_target:
                self.aiming_time += dt
                if self.aiming_time > self.entity.time_from_sight_to_shot:
                    self.entity.game_state.game_over = True
                return self
            else:  # chase player one room
                self.entity.movement_speed = self.entity.chasing_speed
                maze = self.entity.game_state.maze
                my_room = maze.find_room_with_pos(self.entity.pos)
                target_room = maze.find_room_with_pos(self.entity.target.target_entity.pos)
                path = maze.find_path_between_rooms(my_room, target_room)
                return self.entity.WalkState(self.entity, [v3(room.x + 0.5, room.y + 0.5, 0) * maze.room_size for room in path])

    class Target:
        def __init__(self, my_entity, target_entity):
            self.my_entity = my_entity
            self.target_entity = target_entity

        @property
        def delta_to_target(self):
            return self.target_entity.pos - self.my_entity.pos

        @property
        def distance_to_target(self):
            return np.linalg.norm(self.delta_to_target)

        @property
        def direction_to_target(self):
            return self.delta_to_target / self.distance_to_target

        @property
        def distance_to_wall_in_direction_of_target(self):
            return self.my_entity.game_state.maze.hit_ray_on_wall(self.my_entity.pos, self.direction_to_target)

        @property
        def can_spot_target(self):
            return self.distance_to_target < min(self.distance_to_wall_in_direction_of_target, self.my_entity.sight_distance)

    @property
    def is_dying(self):
        return self.current_state is self.DyingState

    def update(self, dt):
        self.current_state = self.current_state(dt) or self.IdleState(self)
        super().update(dt)


class Player(Entity):
    def __init__(self, game_state):
        super().__init__(game_state)

        self.eye_pos = v3()
        self.eye_offset = v3(0, 0, 80)

        self.active_controls = set()

        self.attack = False

        self.yaw = 0.0
        self.pitch = 0.0

        self.control_to_movement_dir = [
            (Controls.MOVE_FORWARDS, v3(1, 0, 0)),
            (Controls.MOVE_LEFT, v3(0, 1, 0)),
            (Controls.MOVE_BACKWARDS, v3(-1, 0, 0)),
            (Controls.MOVE_RIGHT, v3(0, -1, 0)),
        ]

    def update(self, dt):
        self.look_dir = v3(cos(self.yaw) * cos(self.pitch), sin(self.yaw) * cos(self.pitch), sin(self.pitch))

        movement_dir_rel_to_facing = sum((dir for control, dir in self.control_to_movement_dir
                                          if control in self.active_controls), v3())
        if not np.array_equal(movement_dir_rel_to_facing, v3()):
            movement_dir = Rotation.from_euler('Z', self.yaw).apply(movement_dir_rel_to_facing)
            movement_distance = self.movement_speed * dt

            distance_to_collision = self.game_state.maze.hit_ray_on_wall(self.pos, movement_dir)

            if movement_distance > distance_to_collision:
                movement_distance = distance_to_collision - 0.01

            self.pos += movement_dir * movement_distance

        super().update(dt)

        if self.attack:
            self.attack = False
            target, dist = self.game_state.hit_entity(self.eye_pos, self.look_dir)
            if isinstance(target, Enemy) and not target.is_dying:
                target.current_state = target.DyingState(target)


class Controls(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_FORWARDS = 2
    MOVE_BACKWARDS = 3


class GameState:

    def __init__(self, enemy_count=12):
        self.maze = Maze(self)
        self.player = Player(self)
        occupied_rooms = {(0, 0)}
        self.player.pos = v3(0.5, 0.5, 0) * self.maze.room_size
        self.entities = [self.maze, self.player]
        self.game_over = False
        self.enemy_count = enemy_count
        self.enemies_eliminated = 0
        random = Random()
        for _ in range(0, enemy_count):
            enemy = Enemy(self)
            self.entities.append(enemy)
            while True:
                room = random.choice(self.maze.rooms)
                pos = (room.x, room.y)
                if pos not in occupied_rooms:
                    occupied_rooms.add(pos)
                    enemy.pos = v3(room.x + 0.5, room.y + 0.5, 0) * self.maze.room_size
                    break

    @property
    def game_finished(self):
        return self.game_over or self.enemy_count == self.enemies_eliminated

    @property
    def victory(self):
        return self.enemy_count == self.enemies_eliminated

    def hit_entity(self, pos, dir):
        min_dist = 1000000
        nearest_entity = None

        for entity in self.entities:
            for square in entity.to_squares():
                transformed_pos = np.matmul(square.transform, v4_pos(pos))
                transformed_dir = np.matmul(square.transform, v4_dir(dir))
                if transformed_dir[2] != 0:
                    dist = -transformed_pos[2] / transformed_dir[2]
                    if 0 < dist < min_dist:
                        dest = transformed_dir * dist + transformed_pos
                        if 0 <= dest[0] < 1 and 0 <= dest[1] < 1:
                            min_dist = dist
                            nearest_entity = entity

        return nearest_entity, nearest_entity and min_dist

    def update(self, dt):
        for entity in list(self.entities):  # make a copy of list to handle removal during update.
            entity.update(dt)


class KlossRoyaleWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(KlossRoyaleWindow, self).__init__(**kwargs)

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

        self.paused = False

        self.shader = pyshaders.from_string(vert, frag)

        self.pixel_size = 1
        self.pixel_vertices = None
        self._build_vertex_list()

        self.game_state = GameState()

        self.controls_by_key = {
            key.W: Controls.MOVE_FORWARDS,
            key.S: Controls.MOVE_BACKWARDS,
            key.A: Controls.MOVE_LEFT,
            key.D: Controls.MOVE_RIGHT,
        }
        self.mouse_sensitivity = 2 * pi / 1000

        self.camera = Camera()
        self._update_camera()
        self.progress_label = pyglet.text.Label(".", font_name="Arial", font_size=16, bold=True)
        self.game_end_label = pyglet.text.Label(".", font_name="Arial", font_size=16, bold=True)
        pyglet.clock.schedule(self.update, .1)

    def _update_camera(self):
        self.camera.position = self.game_state.player.eye_pos
        self.camera.direction = self.game_state.player.look_dir
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
        if not self.game_state.game_finished:
            if not self.paused:
                self.game_state.update(dt)
            self.progress_label.text = f"ENEMY ELIMINATION: {floor(self.game_state.enemies_eliminated * 100 / self.game_state.enemy_count)}%"
        self._update_camera()
        self.set_exclusive_mouse(not self.paused and not self.game_state.game_finished)

    def on_draw(self):
        self.clear()
        self.shader.use()
        self.shader.uniforms.point_size = self.pixel_size
        self.camera.update_shader_uniforms(self.shader)

        squares = reduce(operator.concat, (entity.to_squares() for entity in self.game_state.entities))
        self.shader.uniforms.xy_squares_count = len(squares)
        self.shader.uniforms.xy_square_transforms = tuple(tuple(tuple(r for r in s.transform)) for s in squares)
        self.shader.uniforms.xy_square_colors = tuple(tuple(s.color) for s in squares)

        self.shader.uniforms.crosshair_size = 20 if not (self.game_state.game_finished or self.paused) else 0
        self.shader.uniforms.crosshair_t = 2
        self.shader.uniforms.crosshair_location = (self.get_viewport_size()[0] * 0.5, self.get_viewport_size()[1] * 0.5)

        self.pixel_vertices.draw(GL_POINTS)
        self.shader.clear()

        window_w, window_h = self.get_viewport_size()
        if self.game_state.game_finished or self.paused:
            if self.game_state.game_finished:
                self.game_end_label.text = "VICTORY" if self.game_state.victory else "YOU WERE SHOT. GAME OVER."
            else:
                self.game_end_label.text = "PAUSED"
            self.game_end_label.x = (window_w-self.game_end_label.content_width)*0.5
            self.game_end_label.y = (window_h-self.game_end_label.content_height)*0.5
            self.game_end_label.draw()
        else:
            self.progress_label.x = window_w - max(self.progress_label.content_width, 100)
            self.progress_label.y = window_h - self.progress_label.content_height
            self.progress_label.draw()

    def on_key_press(self, symbol, modifiers):
        try:
            self.game_state.player.active_controls.add(self.controls_by_key[symbol])
        except KeyError:
            pass

        if symbol == key.SPACE:
            self.paused = not self.paused

    def on_key_release(self, symbol, modifiers):
        try:
            self.game_state.player.active_controls.discard(self.controls_by_key[symbol])
        except KeyError:
            pass

    def on_mouse_motion(self, x, y, dx, dy):
        self.game_state.player.yaw -= self.mouse_sensitivity * dx
        self.game_state.player.yaw %= 2 * pi
        self.game_state.player.pitch += self.mouse_sensitivity * dy
        self.game_state.player.pitch = max(-pi / 2.0, min(pi / 2.0, self.game_state.player.pitch))
        self._update_camera()

    def on_mouse_press(self, x, y, button, modifiers):
        self.game_state.player.attack = True


if __name__ == "__main__":
    window = KlossRoyaleWindow(visible=True, resizable=True)
    window.set_mouse_visible(False)
    window.set_exclusive_mouse(True)
    pyglet.app.run()
