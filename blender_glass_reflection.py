import bpy
import os, sys, random, math
import numpy as np

# --- Config ---
random.seed(42)
SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
OUT_DIR = '/home/bobby/OCEC/blender_out'
os.makedirs(OUT_DIR, exist_ok=True)

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]

W, H = 512, 512
VARIANTS = 4

# --- Setup scene ---
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    # scene.render.image_settings.use_dither = True  # not in 4.2
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.color_depth = '8'
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_samples = 16
    scene.eevee.use_bloom = True
    scene.eevee.bloom_threshold = 0.6
    scene.eevee.bloom_intensity = 0.8
    scene.eevee.bloom_radius = 6.0
    scene.eevee.use_ssr = True
    scene.eevee.use_ssr_refraction = True
    scene.eevee.ssr_quality = 0.5
    scene.eevee.use_gtao = True
    scene.view_layers[0].samples = 64
    # No background
    scene.world = None
    return scene

def create_eye_plane(img_path):
    """Create a plane with eye image as texture, facing camera."""
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    plane = bpy.context.active_object
    # Load image
    img = bpy.data.images.load(img_path, check_existing=False)
    img.colorspace_settings.name = 'Non-Color'
    # Create material
    mat = bpy.data.materials.new('eye_mat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.image.colorspace_settings.name = 'Non-Color'
    # Use emission so it's self-lit (like a display)
    emit = mat.node_tree.nodes.new('ShaderNodeEmission')
    mat.node_tree.links.new(tex.outputs['Color'], emit.inputs['Color'])
    out = mat.node_tree.nodes['Material Output']
    mat.node_tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])
    plane.data.materials.append(mat)
    return plane

def create_glass_lens(radius_x=0.7, radius_y=0.85, thickness=0.02, 
                      curve_depth=0.08, rotation_x=0.0, rotation_y=0.0,
                      offset_x=0.0, offset_y=0.0):
    """Create a curved glass lens in front of the eye."""
    # Use a subdivided plane with curvature
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0.15))
    lens = bpy.context.active_object
    # Subdivide for curvature
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=12)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Apply curve deformation
    for v in lens.data.vertices:
        x, y = v.co.x, v.co.y
        # Normalize to -1..1
        nx = x / 0.5
        ny = y / 0.5
        # Elliptical mask
        r2 = (nx/radius_x)**2 + (ny/radius_y)**2
        if r2 < 1.0:
            # Parabolic curve
            z = curve_depth * (1.0 - r2)
        else:
            z = 0
        v.co.z += z
        # Scale
        v.co.x *= radius_x
        v.co.y *= radius_y
    
    # Glass material
    mat = bpy.data.materials.new('glass_lens')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Transmission Weight'].default_value = 0.95
    bsdf.inputs['Roughness'].default_value = 0.02
    bsdf.inputs['IOR'].default_value = 1.52
    bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Specular IOR Level'].default_value = 1.0
    
    # Thin film / anti-reflective coating simulation via clearcoat
    bsdf.inputs['Coat Weight'].default_value = 0.3
    bsdf.inputs['Coat Roughness'].default_value = 0.01
    bsdf.inputs['Coat IOR'].default_value = 1.8
    
    lens.data.materials.append(mat)
    
    # Position and rotate
    lens.location = (offset_x, offset_y, 0.15)
    lens.rotation_euler = (rotation_x, rotation_y, 0)
    lens.name = 'glass_lens'
    return lens

def setup_lighting(env_img_path, ir_light_pos=None):
    """Setup IR point lights and environment."""
    # Main IR light (strong, from upper-side - typical NIR camera setup)
    bpy.ops.object.light_add(type='POINT', location=(1.2, -0.8, 2.5))
    main_light = bpy.context.active_object
    main_light.data.energy = 800
    main_light.data.color = (1.0, 1.0, 1.0)  # White = IR in grayscale
    main_light.data.shadow_soft_size = 0.3
    
    # Secondary fill light
    bpy.ops.object.light_add(type='POINT', location=(-0.8, 0.5, 1.5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 300
    fill_light.data.color = (1.0, 1.0, 1.0)
    
    # Random bright spots (simulating dashboard/screen reflections)
    for _ in range(random.randint(2, 4)):
        x = random.uniform(-1.5, 1.5)
        y = random.uniform(-1.0, 1.0)
        bpy.ops.object.light_add(type='POINT', location=(x, y, random.uniform(0.5, 2.0)))
        spot = bpy.context.active_object
        spot.data.energy = random.uniform(100, 400)
        spot.data.color = (1.0, 1.0, 1.0)
        spot.data.shadow_soft_size = 0.5
    
    # Area light for dashboard reflection (wide, from below-front)
    bpy.ops.object.light_add(type='AREA', location=(0, 0.3, -0.3))
    area = bpy.context.active_object
    area.data.energy = 200
    area.data.color = (1.0, 1.0, 1.0)
    area.data.size = 3.0
    area.data.size_y = 1.5
    area.rotation_euler = (math.radians(60), 0, 0)

def setup_camera():
    """Setup orthographic camera looking straight at eye."""
    cam = bpy.context.scene.camera
    if cam is None:
        bpy.ops.object.camera_add(location=(0, 0, 3))
        cam = bpy.context.active_object
        bpy.context.scene.camera = cam
    else:
        cam.location = (0, 0, 3)
    cam.data.clip_end = 20
    cam.data.lens = 50
    cam.rotation_euler = (0, 0, 0)
    # Look at origin
    track = cam.constraints.new('TRACK_TO')
    track.target = bpy.data.objects['Plane'] if 'Plane' in bpy.data.objects else bpy.context.scene.objects[0]
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    return cam

def render_one(img_path, out_path, variant_idx):
    """Render one eye image with glass reflection."""
    # Clean scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in bpy.data.images:
        bpy.data.images.remove(img, do_unlink=True)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    
    # Create eye background
    eye_plane = create_eye_plane(img_path)
    
    # Create glass lens with randomized params
    random.seed(42 + variant_idx * 1000 + hash(img_path) % 10000)
    create_glass_lens(
        radius_x=random.uniform(0.55, 0.85),
        radius_y=random.uniform(0.6, 0.95),
        curve_depth=random.uniform(0.04, 0.12),
        thickness=0.02,
        rotation_x=math.radians(random.uniform(-8, 8)),
        rotation_y=math.radians(random.uniform(-8, 8)),
        offset_x=random.uniform(-0.15, 0.15),
        offset_y=random.uniform(-0.15, 0.15),
    )
    
    # Lighting
    setup_lighting(None)
    
    # Camera
    setup_camera()
    
    # Render
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

def main():
    reset_scene()
    
    for fname in samples:
        img_path = os.path.join(SRC_DIR, fname)
        base = os.path.splitext(fname)[0]
        for v in range(VARIANTS):
            out_path = os.path.join(OUT_DIR, f'{base}_blender_v{v}.png')
            print(f'Rendering {fname} variant {v}...')
            render_one(img_path, out_path, v)
    
    print(f'Done! Output in {OUT_DIR}')
    print(os.listdir(OUT_DIR))

if __name__ == '__main__':
    main()
