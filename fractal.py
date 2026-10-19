from asyncore import loop
from typing import overload
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.metal)
n = 320
pixels = ti.field(dtype=float, shape=(n*2, n)) # creates a 640 x 320 2D array 

@ti.func # taichi function, which can only be accessed once already in the taichi kernel
def complex_square(z):
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

@ti.kernel
def paint(t:float): # type hinting necessary for @ti.kernel
    for i, j in pixels:
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i/n - 1, j/n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_square(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res = (n*2, n))

i = 0
while gui.running:
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
    i += 1 # so time is indeed continuous here, the cos() makes it responsible for oscillating
