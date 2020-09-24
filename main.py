import pyglet
import pyglet.gl
import pyglet.clock

import pyshaders

import numpy as np


def v2(x=None, y=None):
    return np.array([float(x or 0.0), float(y or 0.0)])


frag = """
#version 300 es
out mediump vec4 color_frag;
uniform mediump vec3 color;

uniform mediump vec3[256] enemy_positions_and_radii;
uniform int enemy_n;

void main()
{
  bool any_hit = false;

  for (int i = 0; i < enemy_n; ++i) {
    mediump vec2 enemy_pos = enemy_positions_and_radii[i].xy;
    mediump float enemy_radius = enemy_positions_and_radii[i].z;
    mediump float delta_x = enemy_pos.x - float(gl_FragCoord.x);
    mediump float delta_y = enemy_pos.y - float(gl_FragCoord.y);

    if (delta_x*delta_x + delta_y*delta_y <= enemy_radius*enemy_radius) {
      any_hit = true;
    }
  }

  if (any_hit) {
    color_frag = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    color_frag = vec4(1.0, 1.0, 1.0, 1.0);
  }
}
"""

vert = """
#version 300 es
layout(location = 0)in vec2 vert;

void main() {
  gl_Position = vec4(vert, 1, 1);
}
"""

from random import random
from math import cos, sin, pi


class Enemy:
    def __init__(self):
        self.position = v2(400, 300)
        a = random() * 2 * pi
        self.velocity = v2(cos(a), sin(a)) * random() * 300.0 + 300
        self.radius = 5.0 * random() + 5.0

    def update(self, dt):
        self.position += self.velocity * dt

        size = (800, 600)
        for dim in (0, 1):
            if self.position[dim] - self.radius < 0:
                self.position[dim] = self.radius
                self.velocity[dim] = abs(self.velocity[dim])
            if self.position[dim] + self.radius > size[dim]:
                self.position[dim] = size[dim] - self.radius
                self.velocity[dim] = -abs(self.velocity[dim])


class KlossRoyaleWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(KlossRoyaleWindow, self).__init__(**kwargs)

        self.shader = pyshaders.from_string(vert, frag)
        self.shader.use()

        self.enemies = [Enemy() for _ in range(0, 100)]

        self.screen_vertices = pyglet.graphics.vertex_list(6, ('v2f', (-1.0, 1.0, 1.0, -1, -1, -1,
                                                                       -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)))
        self.shader.uniforms.enemy_n = 0
        pyglet.clock.schedule(self.update, .1)

    def update(self, dt, _desired_dt):
        for enemy in self.enemies:
            enemy.update(dt)

        self.shader.uniforms.enemy_n = len(self.enemies)
        self.shader.uniforms.enemy_positions_and_radii = tuple((e.position[0], e.position[1], e.radius)
                                                               for e in self.enemies)

    def on_draw(self):
        self.clear()
        self.screen_vertices.draw(pyglet.gl.GL_TRIANGLES)

    def on_key_press(symbol, modifiers):
        pass


if __name__ == "__main__":
    window = KlossRoyaleWindow(visible=True, width=800, height=600, resizable=True)
    pyglet.app.run()
