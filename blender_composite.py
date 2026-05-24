import bpy
import os, random, math

SRC_DIR = '/home/bobby/OCEC/glass1_samples'
ENV_DIR = '/home/bobby/OCEC/env_reflection_textures'
REFL_DIR = '/home/bobby/OCEC/blender_out3/reflections'
os.makedirs(REFL_DIR, exist_ok=True)

samples = [
    'awake_0_s0036_05832_1_1_1_0_0_01.png',
    'sleepy_0_s0037_04771_1_1_0_1_0_01.png',
    'awake_2_s0035_00444_0_1_1_0_1_02.png',
    'sleepy_1_s0012_02860_0_1_0_2_1_01.png',
]

W, H = 512, 512
VARIANTS = 4
env_files = sorted([f for f in os.listdir(ENV_DIR) if f.endswith('.jpg')])

import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')

def clear_all():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in list(bpy.data.images):
        if img.name not in ('Render Result',):
            bpy.data.images.remove(img, do_unlink=True)

def setup_render():
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
    scene.eevee.bloom_threshold = 0.3
    scene.eevee.bloom_intensity = 1.2
    scene.eevee.bloom_radius = 10.0
    scene.view_layers[0].samples = 64

def create_reflection_scene(env_indices, light_params):
    """Create environment panels and lights - this is what gets reflected."""
    used = list(env_indices)
    random.shuffle(used)
    
    configs = [
        # Full dome coverage - panels in all directions around camera
        # Front/below (dashboard area)
        ('db_0', (0, 1.5, 0.1), (math.radians(-50), 0, 0), (4.0, 2.5, 1)),
        ('db_1', (0.8, 1.2, 0.0), (math.radians(-60), math.radians(15), 0), (3.0, 2.0, 1)),
        ('db_2', (-0.8, 1.2, 0.0), (math.radians(-60), math.radians(-15), 0), (3.0, 2.0, 1)),
        # Sides
        ('side_r', (2.0, 0, 0.5), (0, math.radians(80), 0), (4.0, 3.0, 1)),
        ('side_l', (-2.0, 0, 0.5), (0, math.radians(-80), 0), (4.0, 3.0, 1)),
        # Above
        ('top_0', (0, 0, 3.5), (math.radians(75), 0, 0), (5.0, 4.0, 1)),
        ('top_1', (1.0, 0.5, 3.0), (math.radians(70), math.radians(10), 0), (4.0, 3.0, 1)),
        ('top_2', (-1.0, 0.5, 3.0), (math.radians(70), math.radians(-10), 0), (4.0, 3.0, 1)),
        # Behind/above
        ('back', (0, -2.5, 2.5), (math.radians(40), math.radians(180), 0), (5.0, 3.5, 1)),
        # Far front (sky/windshield view)
        ('front', (0, -3.5, 1.5), (math.radians(25), 0, 0), (6.0, 4.0, 1)),
        # Diagonal corners
        ('diag_r', (1.8, -0.8, 1.2), (math.radians(-35), math.radians(40), 0), (3.5, 2.5, 1)),
        ('diag_l', (-1.8, -0.8, 1.2), (math.radians(-35), math.radians(-40), 0), (3.5, 2.5, 1)),
    ]
    
    for i, (name, pos, rot, scale) in enumerate(configs):
        env_f = env_files[used[i % len(used)]]
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=pos)
        panel = bpy.context.active_object
        panel.location = pos
        panel.rotation_euler = rot
        panel.scale = scale
        
        img = bpy.data.images.load(os.path.join(ENV_DIR, env_f), check_existing=False)
        img.colorspace_settings.name = 'Non-Color'
        
        mat = bpy.data.materials.new(f'env_{name}')
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = img
        tex.image.colorspace_settings.name = 'Non-Color'
        emit = nodes.new('ShaderNodeEmission')
        emit.inputs['Strength'].default_value = random.uniform(5.0, 12.0)
        links.new(tex.outputs['Color'], emit.inputs['Color'])
        links.new(emit.outputs['Emission'], nodes['Material Output'].inputs['Surface'])
        panel.data.materials.append(mat)
    
    # Strong lights for specular highlights
    bpy.ops.object.light_add(type='SPOT', location=(0.6, -1.2, 3.5))
    key = bpy.context.active_object
    key.data.energy = 8000
    key.data.spot_size = math.radians(50)
    key.data.spot_blend = 0.2
    key.rotation_euler = (math.radians(-55), 0, math.radians(-25))
    
    # Dashboard strip
    bpy.ops.object.light_add(type='AREA', location=(0, 0.8, -0.8))
    dash = bpy.context.active_object
    dash.data.energy = 4000
    dash.data.size = 5.0
    dash.data.size_y = 1.2
    dash.rotation_euler = (math.radians(75), 0, 0)
    
    # Screen
    bpy.ops.object.light_add(type='AREA', location=(2.5, 0.3, 1.2))
    scr = bpy.context.active_object
    scr.data.energy = 3000
    scr.data.size = 1.8
    scr.data.size_y = 3.0
    scr.rotation_euler = (0, math.radians(-65), 0)
    
    # Sharp IR LED points for glint
    for _ in range(4):
        x = random.uniform(-1.2, 1.2)
        z = random.uniform(2.0, 4.0)
        bpy.ops.object.light_add(type='POINT', location=(x, -0.3, z))
        led = bpy.context.active_object
        led.data.energy = random.uniform(1000, 3000)
        led.data.shadow_soft_size = 0.01

def render_reflection_only(out_path):
    """Render just the reflection environment (no eye, no glass)."""
    clear_all()
    setup_render()
    
    env_indices = list(range(len(env_files)))
    create_reflection_scene(env_indices, None)
    
    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 4))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    cam.data.lens = 20  # wide angle for fuller coverage
    
    # Black background (no world)
    bpy.context.scene.world = None
    
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

# composite removed - done separately in Python

def main():
    for i, fname in enumerate(samples):
        base = os.path.splitext(fname)[0]
        for v in range(VARIANTS):
            refl_path = os.path.join(REFL_DIR, f'{base}_refl_v{v}.png')
            print(f'Rendering reflection for {fname} v{v}...')
            render_reflection_only(refl_path)
    print(f'Done! {len(os.listdir(REFL_DIR))} reflection images')

if __name__ == '__main__':
    main()
