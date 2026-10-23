"""Builds the MuJoCo model (MJCF XML) for a flat robot arm.

The arm lies on a table seen from above. Every joint is a hinge that spins
around the vertical (z) axis, so gravity never matters: it's a pure
"move your fingertip to the dot" problem.

`link_lengths` decides the body: [0.1, 0.1] is a 2-joint arm,
[0.1, 0.1, 0.1] is a 3-joint arm. Week 2 only has to change this list.
"""


def make_arm_xml(link_lengths=(0.1, 0.1), motor_gear=0.05):
    reach = sum(link_lengths)
    arena = reach + 0.05

    # Each link is a child of the previous one, so we build the XML from
    # the inside out: the last link first, then wrap each parent around it.
    inner = '<site name="fingertip" pos="{:.3f} 0 0" size="0.01" rgba="0.9 0.2 0.2 1"/>'.format(
        link_lengths[-1]
    )
    for i in reversed(range(len(link_lengths))):
        length = link_lengths[i]
        offset = 0 if i == 0 else link_lengths[i - 1]
        inner = f"""
        <body name="link{i}" pos="{offset:.3f} 0 0">
          <joint name="joint{i}" type="hinge" axis="0 0 1" {'' if i == 0 else 'range="-170 170" limited="true"'}/>
          <geom type="capsule" fromto="0 0 0 {length:.3f} 0 0" size="0.01" rgba="0.2 0.5 0.9 1"/>
          {inner}
        </body>"""

    motors = "\n".join(
        f'<motor joint="joint{i}" gear="{motor_gear}" ctrlrange="-1 1" ctrllimited="true"/>'
        for i in range(len(link_lengths))
    )

    return f"""
<mujoco model="planar_arm">
  <compiler angle="degree"/>
  <option timestep="0.01" integrator="RK4"/>
  <default>
    <joint damping="0.002" armature="0.0005"/>
    <geom contype="0" conaffinity="0" density="1000"/>
  </default>
  <visual>
    <global offwidth="480" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1"/>
    <camera name="top" pos="0 0 {arena * 2.6:.3f}" xyaxes="1 0 0 0 1 0"/>
    <geom type="plane" size="{arena:.3f} {arena:.3f} 0.01" rgba="0.92 0.92 0.92 1"/>
    <geom type="cylinder" fromto="0 0 0 0 0 0.02" size="0.015" rgba="0.3 0.3 0.3 1"/>
    {inner}
    <body name="target" mocap="true" pos="{reach * 0.5:.3f} 0 0">
      <geom type="sphere" size="0.012" rgba="0.1 0.8 0.3 1"/>
    </body>
  </worldbody>
  <actuator>
    {motors}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    import mujoco

    model = mujoco.MjModel.from_xml_string(make_arm_xml())
    print(f"joints: {model.njnt}, motors: {model.nu}, timestep: {model.opt.timestep}s")
