# Smoke-screen 3D schematic provenance

- Generated: 2026-08-24 (Asia/Shanghai)
- Method: OpenAI built-in image generation
- Source basis: 2025 CUMCM Problem A PDF supplied in the workspace
- Output role: internal scientific design draft; journal submission eligibility not verified
- Human QA: checked missile-to-decoy flight direction, missile-to-target visibility cone, target geometry, smoke-cloud placement, settling direction, labels, and the not-to-scale disclosure

## Scientific figure contract

The figure shows that effective smoke-screen occlusion occurs when a spherical smoke cloud intersects the visibility cone from an incoming missile to the protected cylindrical target, even though the missile's physical trajectory points toward the decoy target.

Required source facts retained in the artwork:

- Decoy target at `O = (0, 0, 0)`.
- True target base center at `T = (0, 200, 0)`.
- True target radius `r = 7 m` and height `H = 10 m`.
- M1 initial position conceptually `(20000, 0, 2000)` and speed `300 m s^-1`, directed toward O.
- Effective smoke-cloud radius `R = 10 m`.
- Smoke-cloud settling speed `3 m s^-1`.
- Main scene uses perspective compression and is explicitly marked not to scale.

## Final generation prompt

Create a 16:10 landscape, publication-quality 3D scientific schematic in a restrained Nature/Science editorial style on a white background. It must explain smoke-screen occlusion in the 2025 CUMCM Problem A geometry.

Scene: oblique isometric x-y ground plane with z vertical. A single incoming missile labeled "M1" is far away and elevated, starting conceptually at (20000,0,2000). Its thin graphite dashed flight arrow points toward the decoy at the origin, labeled exactly "O (decoy)", with velocity label "300 m s^-1". The protected true target is a brick-red upright cylinder labeled exactly "T (true target)", displaced along +y from the decoy, with coordinate relationship T=(0,200,0), radius label "r = 7 m", height label "H = 10 m".

Crucial geometry: the missile flight path points to O, but a separate sensor visibility cone points from M1 to the true target T. Draw that cone as thin translucent warm-amber tangent rays/surface with apex at M1 and base tightly enclosing the target cylinder. Put one semi-transparent muted teal spherical smoke cloud between M1 and T, centered inside and visibly intersecting the visibility cone. Label its center "Cs" and radius "R = 10 m". The amber line of sight becomes attenuated/blocked behind the smoke toward the cylinder, with the blocked region labeled exactly "occluded". Add a short vertical downward arrow from Cs labeled exactly "3 m s^-1" plus a faint dotted settling path.

Because the range is about 20 km while the cloud radius is 10 m, add a subtle perspective compression or axis-break cue and the small note exactly "schematic, not to scale". Include a clean upper-right magnified inset with a thin gray border, showing the local occlusion geometry: viewpoint apex, two tangent rays around the cylindrical target, the smoke sphere intercepting the rays, and shaded blocked angular region. Label "visibility cone".

Use only short labels and axes. Clean sans-serif text, fine leader lines, generous whitespace, vector-like technical edges with restrained 3D depth and translucent volumetric smoke. Palette: graphite, muted teal-blue, amber, brick red, pale gray; low saturation and color-blind safe.

Scientific invariants: exactly one missile, one smoke sphere, one true target cylinder, one decoy; T is offset +200 m in y from O; missile path points to O; visibility cone points to T; smoke sphere lies between M1 and T and intersects the cone; z is vertical. Do not invent a smoke coordinate or occlusion duration.

Avoid explosions, fire, battlefield scenery, people, flags, radar waves, extra missiles or clouds, logos, equations, invented values, glossy sci-fi UI, dark backgrounds, handwritten labels, and garbled extra prose.
