"""Procedural materials (Principled BSDF). Textures come from textures.py output in TEX_DIR."""
import os

import bpy

TEX_DIR = "/tmp/teardown_tex"
_cache = {}
GLASS = [0.2, 0.35]   # screen glass roughness, specular level (config scene.glass_rough / glass_spec)
BOARD = {}       # config [board]: mask colour and roughness
MAT = {}         # config [material]: case, key, trackpad, floor colours
WEAR = [0.0]      # 0..1 multiplier from config scene.wear; 0 builds no wear nodes at all
DISPLAY = {}     # config [display]: backlight, display-stack films and glass


def _srgb(h):
    """#rrggbb or 0..255 triple -> linear RGBA for a Principled colour input."""
    c = [v / 255.0 for v in h] if not isinstance(h, str) else [int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    lin = [(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in c]
    return (*lin, 1.0)


def _new(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    return m, nt, nt.nodes["Principled BSDF"]


def _set(b, **kw):
    for k, v in kw.items():
        key = k.replace("_", " ")
        try:
            b.inputs[key].default_value = v
        except Exception as exc:                                  # a renamed socket must not kill the build
            print("MAT_WARN", b.id_data.name, key, exc)


def _bump(nt, b, scale=400.0, strength=0.04, dist=0.0003, stretch=(1, 1, 1), detail=6.0):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale * stretch[0], scale * stretch[1], scale * stretch[2])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Detail"].default_value = detail
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = strength
    bp.inputs["Distance"].default_value = dist
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return nz


def _wear(nt, b, rough, amount=0.18, scale=38.0):
    """Fingerprint / dust smudges: a soft noise pushes roughness up in patches, plus fine streaks."""
    amount *= WEAR[0]
    if amount <= 0:
        return
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = scale
    n1.inputs["Detail"].default_value = 5.0
    n1.inputs["Roughness"].default_value = 0.6
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = scale * 9
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 7.0, 1.0)
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], n2.inputs["Vector"])
    r1 = nt.nodes.new("ShaderNodeMapRange")
    r1.inputs["From Min"].default_value = 0.30
    r1.inputs["From Max"].default_value = 0.80
    r1.inputs["To Min"].default_value = rough
    r1.inputs["To Max"].default_value = rough + amount
    nt.links.new(n1.outputs["Fac"], r1.inputs["Value"])
    ad = nt.nodes.new("ShaderNodeMath")
    ad.operation = "ADD"
    ad.inputs[1].default_value = 0.0
    r2 = nt.nodes.new("ShaderNodeMapRange")
    r2.inputs["From Min"].default_value = 0.4
    r2.inputs["From Max"].default_value = 0.7
    r2.inputs["To Min"].default_value = 0.0
    r2.inputs["To Max"].default_value = amount * 0.35
    nt.links.new(n2.outputs["Fac"], r2.inputs["Value"])
    nt.links.new(r1.outputs["Result"], ad.inputs[0])
    nt.links.new(r2.outputs["Result"], ad.inputs[1])
    nt.links.new(ad.outputs["Value"], b.inputs["Roughness"])


def _image(nt, name, nonc=False):
    t = nt.nodes.new("ShaderNodeTexImage")
    path = os.path.join(TEX_DIR, name)
    t.image = bpy.data.images.load(path, check_existing=True)
    t.image.colorspace_settings.name = "Non-Color" if nonc else "sRGB"
    t.interpolation = "Cubic"
    t.extension = "EXTEND"
    return t


def get(name):
    if name not in _cache:
        _cache[name] = globals()["_m_" + name]()
    return _cache[name]


def _metal(name, color, rough, bump=(900, 0.03), stretch=(1, 1, 1), aniso=0.0):
    m, nt, b = _new(name)
    _set(b, Base_Color=_srgb(color), Metallic=1.0, Roughness=rough)
    if aniso:
        _set(b, Anisotropic=aniso)
    if bump:
        _bump(nt, b, scale=bump[0], strength=bump[1], stretch=stretch)
    return m


def _m_alu_grey():     # space-grey anodised aluminium, fine bead-blast texture
    return _m_alu_grey_named("alu_grey")


def _m_alu_grey_named(name):
    m, nt, b = _new(name)
    _set(b, Base_Color=_srgb(MAT["alu"]), Metallic=1.0, Roughness=MAT["alu_rough"], Coat_Weight=0.0)
    _bump(nt, b, scale=1600, strength=0.06, dist=0.00008, detail=8)        # fine bead-blast grain
    _wear(nt, b, MAT["alu_rough"], 0.03, 260.0)
    return m


def _m_alu_shell():    # same anodised metal as alu_grey; a separate material so the exploded film can dim the outer shells
    return _m_alu_grey_named("alu_shell")


def _m_alu_grey_dark():
    m, nt, b = _new("alu_grey_dark")
    _set(b, Base_Color=_srgb((44, 46, 50)), Metallic=1.0, Roughness=0.5)
    _bump(nt, b, scale=1200, strength=0.08, dist=0.00012)
    _wear(nt, b, 0.5, 0.025, 260.0)
    return m


def _m_alu_raw():      # bare aluminium / fan: satin
    return _metal("alu_raw", (200, 202, 206), 0.32, bump=(800, 0.03), stretch=(1, 40, 1))


def _m_steel():
    return _metal("steel", (176, 178, 182), 0.28, bump=(900, 0.03))


def _m_shield():       # tin-plated EMI can: soft satin steel
    return _metal("shield", (105, 108, 113), 0.55, bump=(900, 0.025))


def _m_steel_dark():
    return _metal("steel_dark", (70, 72, 76), 0.34, bump=(900, 0.03))


def _m_copper():
    return _metal("copper", (232, 140, 104), 0.2, bump=(500, 0.05), stretch=(1, 30, 1))


def _m_gold():
    return _metal("gold", (240, 190, 96), 0.22, bump=None)


def _m_black_plastic():
    m, nt, b = _new("black_plastic")
    _set(b, Base_Color=_srgb((14, 14, 16)), Roughness=0.46, Specular_IOR_Level=0.5)
    _bump(nt, b, scale=900, strength=0.07, dist=0.0002)
    return m


def _m_trackpad_glass():   # etched glass tinted to the case, soft sheen
    m, nt, b = _new("trackpad_glass")
    _set(b, Base_Color=_srgb(MAT["trackpad"]), Roughness=MAT["trackpad_rough"], Coat_Weight=0.25, Coat_Roughness=0.2, Specular_IOR_Level=0.5)
    _bump(nt, b, scale=2500, strength=0.03, dist=0.00005, detail=2)
    _wear(nt, b, 0.2, 0.06, 220.0)
    return m


def _m_black_glossy():
    m, nt, b = _new("black_glossy")
    _set(b, Base_Color=_srgb((8, 8, 9)), Roughness=0.12, Coat_Weight=0.5, Coat_Roughness=0.05)
    return m


def _m_rubber():
    m, nt, b = _new("rubber")
    _set(b, Base_Color=_srgb((16, 16, 17)), Roughness=0.8)
    _bump(nt, b, scale=700, strength=0.15, dist=0.0003)
    return m


def _m_adhesive():     # pale double-sided tape / glue line
    m, nt, b = _new("adhesive")
    _set(b, Base_Color=_srgb((176, 174, 164)), Roughness=0.55, Specular_IOR_Level=0.3)
    _bump(nt, b, scale=900, strength=0.1, dist=0.0002)
    return m


def _m_cell_wrap():    # blue-grey insulating film around pouch cells
    m, nt, b = _new("cell_wrap")
    _set(b, Base_Color=_srgb((64, 70, 82)), Roughness=0.3, Coat_Weight=0.4, Coat_Roughness=0.12)
    _bump(nt, b, scale=2600, strength=0.05, dist=0.0001, detail=3)
    return m


def _m_keycap():
    m, nt, b = _new("keycap")
    t = _image(nt, "keys.png")
    t.interpolation = "Linear"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    _set(b, Roughness=0.55, Specular_IOR_Level=0.3)
    _bump(nt, b, scale=1800, strength=0.04, dist=0.0001, detail=3)
    if DISPLAY["backlight"] > 0:                                   # light only through the legends: the legend ink is the mask
        tint = nt.nodes.new("ShaderNodeVectorMath")
        tint.operation = "MULTIPLY"
        tint.inputs[1].default_value = DISPLAY["backlight_color"]
        nt.links.new(t.outputs["Color"], tint.inputs[0])
        nt.links.new(tint.outputs["Vector"], b.inputs["Emission Color"])
        _powered(m, b, DISPLAY["backlight"])
    return m


def _powered(m, b, strength):
    """Emission that render.py scales by the pose's power (on while the lid is open, off once it is shut)."""
    m["power_base"] = strength
    _set(b, Emission_Strength=strength)


def _m_kb_membrane():  # under the caps: dark, with an even faint halo through every key gap when the backlight is on
    m, nt, b = _new("kb_membrane")
    _set(b, Base_Color=_srgb((10, 10, 11)), Roughness=0.7, Emission_Color=(*DISPLAY["backlight_color"], 1.0))
    if DISPLAY["halo"] > 0:
        _powered(m, b, DISPLAY["halo"])
    return m


def _pcb(name, tex, rough=None):
    rough = BOARD["mask_rough"] if rough is None else rough
    m, nt, b = _new(name)
    t = _image(nt, tex + ".png")
    mt = _image(nt, tex + "_metal.png", nonc=True)
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(mt.outputs["Color"], b.inputs["Metallic"])
    _set(b, Roughness=rough, Specular_IOR_Level=0.3, Coat_Weight=0.0)
    _bump(nt, b, scale=800, strength=0.05, dist=0.0001, detail=2)
    return m


def _m_pcb_main():
    return _pcb("pcb_main", "pcb_main")


def _m_pcb_ssd():
    return _pcb("pcb_ssd", "pcb_ssd")


def _m_pcb_wifi():
    return _pcb("pcb_wifi", "pcb_wifi")


def _m_pcb_io():
    return _pcb("pcb_io", "pcb_io")


def _m_pcb_edge():     # side/bottom of boards and the SoC package substrate: dark laminate, near the mask
    m, nt, b = _new("pcb_edge")
    _set(b, Base_Color=_srgb(MAT["pcb_edge"]), Roughness=0.45)
    return m


def _chip(name, tex, rough=0.42):
    m, nt, b = _new(name)
    t = _image(nt, tex)
    t.interpolation = "Linear"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    _set(b, Roughness=rough, Specular_IOR_Level=0.5)
    _bump(nt, b, scale=1500, strength=0.06, dist=0.00012, detail=3)
    return m


def _m_chip_ram():
    return _chip("chip_ram", "chip_ram.png")


def _m_chip_nand():
    return _chip("chip_nand", "chip_nand.png")


def _m_chip_ctrl():
    return _chip("chip_ctrl", "chip_ctrl.png")


def _m_chip_soc():
    m = _chip("chip_soc", "chip_soc.png", rough=0.28)
    m.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.9
    return m


def _m_cap():          # ceramic MLCC: tan
    m, nt, b = _new("cap")
    _set(b, Base_Color=_srgb((142, 112, 78)), Roughness=0.5)
    return m


def _m_res():          # thick-film resistor: black with silver ends
    m, nt, b = _new("res")
    _set(b, Base_Color=_srgb((78, 76, 74)), Roughness=0.45)
    return m


def _m_polycap():
    m, nt, b = _new("polycap")
    _set(b, Base_Color=_srgb((150, 118, 66)), Roughness=0.35, Coat_Weight=0.3)
    return m


def _m_inductor():
    m, nt, b = _new("inductor")
    _set(b, Base_Color=_srgb((92, 94, 98)), Roughness=0.4, Metallic=0.3)
    return m


def _m_connector():
    m, nt, b = _new("connector")
    _set(b, Base_Color=_srgb((30, 24, 20)), Roughness=0.4)
    return m


def _m_latch():
    m, nt, b = _new("latch")
    _set(b, Base_Color=_srgb((196, 192, 180)), Roughness=0.5)
    return m


def _m_polyimide():    # amber tape / flex
    m, nt, b = _new("polyimide")
    _set(b, Base_Color=_srgb((150, 92, 28)), Roughness=0.3, Coat_Weight=0.5, Coat_Roughness=0.08)
    return m


def _m_pouch():        # aluminium laminate cell
    m, nt, b = _new("pouch")
    _set(b, Base_Color=_srgb((150, 152, 158)), Metallic=0.95, Roughness=0.36)
    _bump(nt, b, scale=2200, strength=0.12, dist=0.0002, stretch=(1, 1, 1))
    return m


def _m_sticker():      # blank paper label
    m, nt, b = _new("sticker")
    _set(b, Base_Color=_srgb((176, 176, 172)), Roughness=0.7)
    _bump(nt, b, scale=1500, strength=0.05, dist=0.0001)
    return m


def _m_wire_red():
    m, nt, b = _new("wire_red")
    _set(b, Base_Color=_srgb((140, 20, 20)), Roughness=0.4)
    return m


def _m_wire_black():
    m, nt, b = _new("wire_black")
    _set(b, Base_Color=_srgb((16, 16, 18)), Roughness=0.4)
    return m


def _m_wire_grey():
    m, nt, b = _new("wire_grey")
    _set(b, Base_Color=_srgb((110, 112, 116)), Roughness=0.4)
    return m


def _m_foam():
    m, nt, b = _new("foam")
    _set(b, Base_Color=_srgb((24, 24, 26)), Roughness=0.9)
    return m


def _m_speaker_cone():
    m, nt, b = _new("speaker_cone")
    _set(b, Base_Color=_srgb((186, 188, 192)), Metallic=0.6, Roughness=0.4)
    return m


def _m_mesh_dark():
    m, nt, b = _new("mesh_dark")
    _set(b, Base_Color=_srgb((10, 10, 11)), Roughness=0.7)
    return m


def _m_screen():       # the LCD's lit face: wallpaper emission, matte under the cover glass so the glass carries the only gloss
    m, nt, b = _new("screen")
    t = _image(nt, "screen.png")
    t.interpolation = "Linear"
    _set(b, Base_Color=(0.003, 0.003, 0.004, 1), Roughness=0.4, Specular_IOR_Level=0.0)
    nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
    _powered(m, b, MAT["screen_emission"])
    return m


def _see_through(name, color, opacity, rough, spec, fresnel=None):
    """Thin sheet: a mix of a transparent pass and a coloured surface, so shadows and the parts behind show through.

    opacity is the share of the surface; with fresnel=(lo, hi, blend) it ramps from lo face-on to hi at grazing angles."""
    m, nt, b = _new(name)
    _set(b, Base_Color=_srgb(color) if not isinstance(color, tuple) else color, Roughness=rough, Specular_IOR_Level=spec, Coat_Weight=0.0)
    out = nt.nodes["Material Output"]
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    if fresnel:
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = fresnel[2]
        rng = nt.nodes.new("ShaderNodeMapRange")
        rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = fresnel[0], fresnel[1]
        nt.links.new(lw.outputs["Fresnel"], rng.inputs["Value"])
        nt.links.new(rng.outputs["Result"], mix.inputs["Fac"])
    else:
        mix.inputs["Fac"].default_value = opacity
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(b.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def _m_cover_glass():  # clear glass, dark gloss only at the edges of view; never a silver mirror
    lo, hi = DISPLAY["glass_opacity"]
    return _see_through("cover_glass", (0.002, 0.002, 0.003, 1), None, GLASS[0], GLASS[1], fresnel=(lo, hi, DISPLAY["glass_fresnel"]))


def _m_glass_ink():    # black ink printed round the inside of the cover glass: the bezel
    return _plain("glass_ink", (6, 6, 7), 0.5)


def _m_cell_glass():   # the LCD cell's glass seen edge-on and from behind: dark, glossy, a faint blue-green glass tint
    return _plain("cell_glass", DISPLAY["cell_glass"], 0.18, coat=0.6)


def _m_glass_edge():   # polished edge of the cover glass: the green-black of thick glass seen end-on, sharp highlight
    return _plain("glass_edge", DISPLAY["glass_edge"], 0.08, coat=1.0)


def _m_panel_steel():  # stamped stainless back and tray of the backlight unit: satin, brushed along the panel
    return _metal("panel_steel", DISPLAY["panel_steel"], DISPLAY["panel_steel_rough"], bump=(900, 0.03), stretch=(1, 30, 1))


def _m_fpc():          # dark amber polyimide flex
    return _plain("fpc", DISPLAY["fpc"], 0.3, coat=0.3)




def _m_panel_edge():
    m, nt, b = _new("panel_edge")
    _set(b, Base_Color=_srgb((12, 12, 14)), Roughness=0.5)
    return m


def _plain(name, color, rough, coat=0.0, metal=0.0, bump=None):
    m, nt, b = _new(name)
    _set(b, Base_Color=_srgb(color), Roughness=rough, Metallic=metal, Coat_Weight=coat, Coat_Roughness=0.15)
    if bump:
        _bump(nt, b, scale=bump[0], strength=bump[1], dist=0.0002)
    return m


def _m_cell_black():   # matte black pouch-cell wrap, a faint film sheen
    return _plain("cell_black", MAT["cell"], 0.68, bump=(1800, 0.05))


def _m_cell_label():   # blank label panel on a cell, a shade lighter than the wrap
    return _plain("cell_label", MAT["cell_label"], 0.62)


def _m_tray_dark():    # battery tray and dividers: dark satin plastic
    return _plain("tray_dark", MAT["tray"], 0.42, bump=(900, 0.05))


def _m_sink_dark():    # heat-sink plate and fan housing: dark painted metal
    return _plain("sink_dark", MAT["sink"], 0.38, metal=0.6, bump=(1200, 0.04))


def _m_fan_blade():    # fine black impeller
    return _plain("fan_blade", MAT["blade"], 0.4)


def _m_copper_fin():   # thin copper fin stack, a touch duller than the pipes
    return _metal("copper_fin", MAT["copper_fin"], 0.3, bump=None)


def _m_zif():          # ivory FPC connector body
    return _plain("zif", MAT["zif"], 0.5)


def _m_flex_dark():    # black-coated flex cable
    return _plain("flex_dark", MAT["flex"], 0.35, coat=0.4)


def _m_wire_yellow():
    return _plain("wire_yellow", MAT["wire_yellow"], 0.4)


def _m_floor():
    m, nt, b = _new("floor")
    _set(b, Base_Color=_srgb(MAT["floor"]), Roughness=MAT["floor_rough"], Specular_IOR_Level=MAT["floor_spec"])
    _bump(nt, b, scale=60, strength=0.02, dist=0.0005, detail=4)
    return m
