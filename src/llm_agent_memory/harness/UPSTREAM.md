# Open Gemini robot harness adaptation

Source: https://github.com/ehehee/open-gemini-robot-harness
Revision: `470e1d2f943e3f8bb832b9aef28e4310e81344bc`
Article: https://ehehee.github.io/blog/open-gemini-harness/

`plan_program.py` comes from the source's restricted Code-as-Policies parser.
`upstream.py` contains its primitive signatures, orientation helpers, system
prompt, and curl response parser. The runner substitutes the robot/model/frame
descriptions and automatic-check instructions. Model-produced shell commands
and Python are parsed as data, never executed as shell or arbitrary Python.

The remaining modules implement the single Panda simulator adapter. The
private worker derives from the existing pinned inspect-robots-libero worker.

This is not an exact hardware reproduction or the unchanged LLM_LIBERO
baseline. Relevant differences:

- One seven-joint Panda replaces the original YAM hardware. No bimanual or
  parallel execution. Panda jaw axes are remapped explicitly.
- MuJoCo camera calibration and rendered depth replace the physical cameras
  and stereo estimator. Two upright 512x512 views receive a 2 cm world grid,
  pinch/jaw/wrist reticles, cropping, and ray-plane/depth deprojection.
- Automatic damped-Jacobian IK, joint limits and sampled MuJoCo contacts replace
  the source's gap_guard/human review. Checking uses simulator scene geometry;
  object poses are not exposed to the model. This is extra infrastructure
  information compared with a strictly RGB/proprioception-only benchmark.
- Each Cartesian segment is checked immediately before execution; there is no
  whole-program preview, gripper sweep preview, hardware emergency stop, or
  YAM-specific IK fallback. A rejected later segment can follow executed ones.
- Commands use the existing LIBERO OSC controller. Speed, contact lag, holding
  and grasp failure are approximate Panda equivalents. Holding is a measured
  finger-gap proxy, rechecked during moves, not a ground-truth grasp label.
- LIBERO's own success predicate establishes the episode outcome. There is no
  manual success override. The default limits remain 80 decisions/900 steps.
- Fresh context each episode, full episode text history and the latest two
  observation image pairs. No demonstrations or memory across episodes.

Every run records its exact prompt and these differences in protocol.json.
Raw model request/response files, annotated observations, tool feedback,
per-step motion, and raw-camera video are retained for inspection.
