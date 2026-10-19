from re import X
import taichi as ti
import taichi.math as tm


ti.init(arch=ti.metal)
pixels = ti.Vector.field(n=3, dtype=float, shape = (800, 600))
gui = ti.GUI("Monte Carlo Raytracer", res = (800, 600))
active_mat_id = ti.field(dtype=int, shape=())
active_mat_id_2 = ti.field(dtype=int, shape=())
active_mat_id[None] = 1 # left ball
active_mat_id_2[None] = 1 # right ball

# need to setup GUI

@ti.func
def hit_sphere(ray_origin, ray_dir, sphere_center, radius):
    oc = sphere_center - ray_origin
    a = tm.dot(ray_dir, ray_dir)
    b = -2.0 * tm.dot(ray_dir, oc)
    c = tm.dot(oc, oc) - radius ** 2
    hit_distance = 1000000.0 # this is the value of the ray does NOT hit anything (goes off into the distance)

    is_hit = False
    normal_vector = tm.vec3(0.0)
    collision_coordinate = tm.vec3(0.0)
    discriminant = b ** 2 - 4 * a * c
    if discriminant > 0: # < 0 means no intersection with circle. == 0 means tangent line, which doesn't reflect.
        solution = ( -b - tm.sqrt(discriminant) ) / ( 2 * a ) # only care about the lesser t, as the greater t would give us the ray going out the back of the sphere
        # solution is also our distance, since we are multiplying it against a vector whose magnitude is 1
        if solution > 0: # if solution is a positive t value
            collision_coordinate = ray_origin + solution * ray_dir # this allows us to add each component w/o having to break everything up
            is_hit = True
            # now calculate the normal vector by taking the vector from the center to the collision_point
            normal_vector = collision_coordinate - sphere_center
            # normalize
            normal_vector = tm.normalize(normal_vector)
            hit_distance = solution
    return is_hit, collision_coordinate, normal_vector, hit_distance

# checks what object in our world we hit first (floor? or sphere?)
@ti.func
def hit_world(ray_origin, ray_direction):
    closest_t = 1000000.0
    final_hit = False
    final_coord = tm.vec3(0.0)
    final_normal = tm.vec3(0.0)
    final_albedo = tm.vec3(0.0)
    final_mat_id = 0 # 0 = matte, 1 = metal
    # sphere first?
    hit1, coord1, normal1, distance1 = hit_sphere(ray_origin, ray_direction, tm.vec3(-0.6, 0.0, -2.0), 0.5)
    # floor first?
    hit2, coord2, normal2, distance2 = hit_sphere(ray_origin, ray_direction, tm.vec3(0.0, -100.5, -2.0), 100.0)
    # other sphere first?
    hit3, coord3, normal3, distance3 = hit_sphere(ray_origin, ray_direction, tm.vec3(0.6, 0.0, -2.0), 0.5)

    # now compare values
    if hit1 and distance1 < closest_t:
        closest_t = distance1
        final_hit = hit1
        final_coord = coord1
        final_normal = normal1
        final_albedo = tm.vec3(0.8, 0.8, 0.8)
        final_mat_id = active_mat_id[None] # since we hit the ball, we know this is metal
    
    if hit2 and distance2 < closest_t:
        closest_t = distance2
        final_hit = hit2
        final_coord = coord2
        final_normal = normal2
        final_albedo = tm.vec3(0.2, 0.6, 0.2)
        final_mat_id = 0 # floor is matte

    if hit3 and distance3 < closest_t:
        closest_t = distance3
        final_hit = hit3
        final_coord = coord3
        final_normal = normal3
        final_albedo = tm.vec3(1.0, 1.0, 1.0)
        final_mat_id = active_mat_id_2[None] # glass ball
    # and return the hit (true/false), hit coordinate, and normal vector of that hit coordinate of the CLOSER object
    return final_hit, final_coord, final_normal, final_albedo, final_mat_id

@ti.func
def random_unit_vector():
    z = 2.0 * ti.random() - 1.0
    r = tm.sqrt(1.0 - z ** 2)
    theta = 2.0 * tm.pi * ti.random()
    x = r * tm.cos(theta)
    y = r * tm.sin(theta)
    return tm.vec3(x, y, z)

# to reflect light for metal
@ti.func
def reflect(v, n):
    return v - 2.0 * tm.dot(v, n) * n # specific normal vector if our material is metal (very reflective)

@ti.func
def refract(v, n, eta_ratio):
    cos_theta = min(tm.dot(-v, n), 1.0)
    r_out_perp = eta_ratio * (v + cos_theta * n)
    r_out_parallel = -tm.sqrt(tm.max(0.0, 1.0 - tm.dot(r_out_perp, r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def ray_color(ray_origin, ray_direction):
    color_multiplier = tm.vec3(1.0) # represents how much light the vector still carries
    sky_color = tm.vec3(0.0)
    bounce = 0
    for i in range(10):
        hit, collision_coordinate, normal, albedo, mat_id = hit_world(ray_origin, ray_direction)
        if hit:
            ray_origin = collision_coordinate
            if mat_id == 0: # matte ball, just uses normal vector
                ray_direction = tm.normalize(normal + random_unit_vector())
            elif mat_id == 1: # metal ball, uses reflect()
                ray_direction = reflect(ray_direction, normal)
            elif mat_id == 2: # glass ball, uses refract()
                front_face = tm.dot(ray_direction, normal) < 0.0
                outward_normal = normal if front_face else -normal

                #index of refraction
                eta_ratio = 1.0 / 1.5 if front_face else 1.5

                cos_theta = cos_theta = tm.min(tm.dot(-ray_direction, outward_normal), 1.0)
                sin_theta = tm.sqrt(1.0 - cos_theta**2)

                # internal reflection
                if eta_ratio * sin_theta > 1.0:
                    ray_direction = reflect(ray_direction, outward_normal)
                else:
                    ray_direction = refract(ray_direction, outward_normal, eta_ratio)
            color_multiplier *= albedo
            bounce += 1
        else:
            a = 0.5 * (ray_direction.y + 1.0)
            top_color = tm.vec3(0.5, 0.7, 1.0)
            bottom_color = tm.vec3(0.1, 0.1, 0.1)
            sky_color = a * top_color + (1.0-a) * bottom_color
            break
    if bounce >= 9:
        color_multiplier = tm.vec3(0)

    return color_multiplier * sky_color

@ti.kernel
def render():
    ray_origin = tm.vec3(0.0)
    for i, j in pixels:
        color = tm.vec3(0.0)
        for sample in range(50):
            u = ((i + ti.random()) / 400 - 1) * 4/3 # multiply by aspect ratio to make a rectangular image
            v = (j + ti.random()) / 300 - 1
            ray_direction = tm.vec3(u, v, -1.0) # goes from eye (origin, z=0) to screen (z=-1)
            ray_direction = tm.normalize(ray_direction) # normalized so that magnitude is exactly 1
            color += ray_color(ray_origin, ray_direction)
        pixels[i, j] = tm.sqrt(color / 50.0) # gamma correction

while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == '1':
            active_mat_id[None] = 0  # switch to matte 
        elif gui.event.key == '2':
            active_mat_id[None] = 1  # switch to metal
        elif gui.event.key == '3':
            active_mat_id[None] = 2  # switch to glass
        elif gui.event.key == '8':
            active_mat_id_2[None] = 0  # switch to Matte
        elif gui.event.key == '9':
            active_mat_id_2[None] = 1  # switch to Metal
        elif gui.event.key == '0':
            active_mat_id_2[None] = 2  # switch to Glass
            
    render()
    gui.set_image(pixels)
    gui.show()