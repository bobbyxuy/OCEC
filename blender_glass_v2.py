import bpy
import os, sys, random, math
import numpy as np

random.seed(42)
SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
OUT_DIR = '/home/bobby/OCEC/blender_out2'
os.makedirs(OUT_DIR, exist_ok=True)

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]

W, H = 512, 512
VARIANTS = 4

env_files = sorted([f for f in os.listdir(ENV_DIR) if f.endswith('.jpg')])

def clear_all():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)

def setup_scene():
    scene = bpy.context.scene
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.color_depth = '8'
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_samples = 32
    scene.eevee.use_bloom = True
    scene.eevee.bloom_threshold = 0.5
    scene.eevee.bloom_intensity = 1.0
    scene.eevee.bloom_radius = 8.0
    scene.eevee.use_ssr = True
    scene.eevee.use_ssr_refraction = True
    scene.eevee.ssr_quality = 1.0
    scene.view_layers[0].samples = 64

def create_eye_bg(img_path):
    """Eye image as emissive background plane at z=0."""
    bpy.ops.mesh.primitive_plane_add(size=2.2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = 'eye_bg'
    
    img = bpy.data.images.load(img_path, check_existing=False)
    img.colorspace_settings.name = 'Non-Color'
    
    mat = bpy.data.materials.new('eye_bg_mat')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes['Principled BSDF']
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.image.colorspace_settings.name = 'Non-Color'
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Strength'].default_value = 2.0
    links.new(tex.outputs['Color'], emit.inputs['Color'])
    links.new(emit.outputs['Emission'], nodes['Material Output'].inputs['Surface'])
    plane.data.materials.append(mat)
    return plane

def create_env_panels():
    """Place environment photos as emissive panels around the scene to be reflected."""
    panels = []
    configs = [
        # (name, position, rotation, scale, description)
        ('dashboard', (0, 1.5, 0.3), (math.radians(-50), 0, 0), (3.0, 1.5, 1), '仪表盘'),
        ('steering', (1.2, 0.8, 0.1), (math.radians(-70), math.radians(20), 0), (1.8, 1.8, 1), '方向盘'),
        ('sky', (0, -2.5, 1.5), (math.radians(30), 0, 0), (5.0, 3.0, 1), '天空/窗外'),
        ('side_panel', (-1.8, 0, 0.5), (0, math.radians(70), 0), (2.5, 2.0, 1), '侧面'),
        ('ceiling_light', (0.3, 0.3, 2.5), (math.radians(60), math.radians(-10), 0), (3.0, 2.0, 1), '顶灯'),
    ]
    
    used = []
    for name, pos, rot, scale, desc in configs:
        env_f = random.choice(env_files)
        while env_f in used and len(used) < len(env_files):
            env_f = random.choice(env_files)
        used.append(env_f)
        
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=pos)
        panel = bpy.context.active_object
        panel.name = f'env_{name}'
        panel.location = pos
        panel.rotation_euler = rot
        panel.scale = scale
        
        img = bpy.data.images.load(os.path.join(ENV_DIR, env_f), check_existing=False)
        img.colorspace_settings.name = 'Non-Color'
        
        mat = bpy.data.materials.new(f'env_mat_{name}')
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes['Principled BSDF']
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = img
        tex.image.colorspace_settings.name = 'Non-Color'
        emit = nodes.new('ShaderNodeEmission')
        # Bright emission so reflections are visible
        emit.inputs['Strength'].default_value = random.uniform(3.0, 8.0)
        links.new(tex.outputs['Color'], emit.inputs['Color'])
        links.new(emit.outputs['Emission'], nodes['Material Output'].inputs['Surface'])
        panel.data.materials.append(mat)
        panels.append(panel)
    
    return panels

def create_glass_lens(params):
    """Create curved glass lens with realistic material."""
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0.2))
    lens = bpy.context.active_object
    lens.name = 'glass_lens'
    
    # Subdivide and curve
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=20)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    rx = params['radius_x']
    ry = params['radius_y']
    curve = params['curve_depth']
    
    for v in lens.data.vertices:
        nx = v.co.x / 0.5
        ny = v.co.y / 0.5
        r2 = (nx/rx)**2 + (ny/ry)**2
        if r2 < 1.0:
            v.co.z += curve * (1.0 - r2**2)
        v.co.x *= rx
        v.co.y *= ry
    
    # Glass material - HIGH reflection
    mat = bpy.data.materials.new('glass')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    
    # Make it more reflective than transparent for visible reflections
    bsdf.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.01  # Very smooth = sharp reflections
    bsdf.inputs['Transmission Weight'].default_value = 0.3  # Low transmission = more reflection visible
    bsdf.inputs['IOR'].default_value = 1.52
    bsdf.inputs['Specular IOR Level'].default_value = 2.0  # Strong specular
    bsdf.inputs['Coat Weight'].default_value = 1.0  # Full clearcoat
    bsdf.inputs['Coat Roughness'].default_value = 0.0
    bsdf.inputs['Coat IOR'].default_value = 2.0
    bsdf.inputs['Thin Film IOR'].default_value = 1.8  # Anti-reflective coating color shift
    
    lens.data.materials.append(mat)
    
    lens.location = (params['offset_x'], params['offset_y'], 0.2)
    lens.rotation_euler = (params['rot_x'], params['rot_y'], 0)
    return lens

def setup_lights():
    """Strong directional and point lights for visible specular highlights."""
    # Strong key light from upper-left (NIR camera position)
    bpy.ops.object.light_add(type='SPOT', location=(0.8, -1.0, 3.0))
    key = bpy.context.active_object
    key.data.energy = 5000
    key.data.color = (1.0, 1.0, 1.0)
    key.data.spot_size = math.radians(60)
    key.data.spot_blend = 0.3
    key.rotation_euler = (math.radians(-50), 0, math.radians(-30))
    
    # Dashboard light strip (area light from below)
    bpy.ops.object.light_add(type='AREA', location=(0, 0.5, -0.5))
    dash = bpy.context.active_object
    dash.data.energy = 3000
    dash.data.color = (1.0, 1.0, 1.0)
    dash.data.size = 4.0
    dash.data.size_y = 1.0
    dash.rotation_euler = (math.radians(70), 0, 0)
    
    # Screen/display light from right
    bpy.ops.object.light_add(type='AREA', location=(2.0, 0.5, 1.0))
    screen = bpy.context.active_object
    screen.data.energy = 2000
    screen.data.color = (1.0, 1.0, 1.0)
    screen.data.size = 1.5
    screen.data.size_y = 2.5
    screen.rotation_euler = (0, math.radians(-60), 0)
    
    # Small bright IR LED points (create sharp glint spots)
    for _ in range(3):
        x = random.uniform(-1.0, 1.0)
        z = random.uniform(1.5, 3.0)
        bpy.ops.object.light_add(type='POINT', location=(x, -0.5, z))
        led = bpy.context.active_object
        led.data.energy = random.uniform(500, 2000)
        led.data.color = (1.0, 1.0, 1.0)
        led.data.shadow_soft_size = 0.02  # Sharp shadow = crisp specular

def setup_camera():
    bpy.ops.object.camera_add(location=(0, 0, 4))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    cam.data.lens = 50
    cam.data.clip_end = 20
    return cam

def render_one(img_path, out_path, variant_idx):
    clear_all()
    
    random.seed(42 + variant_idx * 1000 + hash(img_path) % 10000)
    
    setup_scene()
    create_eye_bg(img_path)
    create_env_panels()
    
    create_glass_lens({
        'radius_x': random.uniform(0.5, 0.8),
        'radius_y': random.uniform(0.55, 0.9),
        'curve_depth': random.uniform(0.06, 0.15),
        'rot_x': math.radians(random.uniform(-10, 10)),
        'rot_y': math.radians(random.uniform(-10, 10)),
        'offset_x': random.uniform(-0.1, 0.1),
        'offset_y': random.uniform(-0.1, 0.1),
    })
    
    setup_lights()
    setup_camera()
    
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

def main():
    for fname in samples:
        img_path = os.path.join(SRC_DIR, fname)
        base = os.path.splitext(fname)[0]
        for v in range(VARIANTS):
            out_path = os.path.join(OUT_DIR, f'{base}_blender_v{v}.png')
            print(f'Rendering {fname} variant {v}...')
            render_one(img_path, out_path, v)
    
    print(f'Done! {len(os.listdir(OUT_DIR))} images in {OUT_DIR}')

if __name__ == '__main__':
    main()
