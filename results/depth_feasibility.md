# Depth from the X2Real/Isaac sim: feasibility (read-only investigation)

## 1. Is depth available in sim with no code changes?
**No on the wire, yes in the simulator.** The sim can render depth, but nothing sends it to the policy server.

- `C:\x2\x2real\manaenv\assets\robots\ex001_6r.py` line 371: `head_camera` is `data_types=["rgb"]`. The wrist cameras are fisheye cfgs. The `camera_obs` ObsTerms (`left_wrist_cam`, `right_wrist_cam`, `head_cam`, lines 594-620) all use `mdp_isaac_lab.image` with `data_type: "rgb"`.
- **Depth already exists as an opt-in.** `_apply_optional_camera_depth_semantic()` (lines 630-689) adds `distance_to_image_plane` (Z-depth along the camera +Z axis, the right kind for box height) and `semantic_segmentation` to all three cameras.
  - It also adds ObsTerms named `head_cam_distance_to_image_plane`, `left_wrist_cam_distance_to_image_plane` and `right_wrist_cam_distance_to_image_plane`. `normalize=False` is set, so values are in metres.
  - The flag is `camera_depth_semantic: bool = False` (ctor, line 700; applied at line 735). It is plumbed from `manaenv\pipeline\factories\robot_factory.py` (field at line 51, passed at lines 288-289).
  - Nothing under `C:\x2\x2real` turns it on. The only matches for `camera_depth_semantic` are those definitions, and no task YAML sets it.
- **The Policy Space bridge drops it anyway.** `manaenv\policy\policy_implementation\policy_space.py` `_build_obs` (lines 1334-1360) loops only over `self._camera_map`. That map comes from the embodiment YAML `obs_field_map` (`defination\embodiments\policy_space\EX0016REmbodiment.yaml`): `camera_front: head_cam`, `camera_left: left_wrist_cam`, `camera_right: right_wrist_cam`.
  - Each camera goes through `_to_rgb_uint8()` (line 161) and `_resize_rgb()`. These would reject or mangle a 1-channel float depth image, and depth is not in the map in any case.
- **The schema is RGB-only.** `bimanual_rgb_ee@1.yaml` has `images: roles [camera_front, camera_left, camera_right], uint8 [h,w,3]`. The installed schemas are `bimanual_rgb_ee@1`, `bimanual_rgb_joint@1`, `mobile_bimanual_rgb@1` and `single_arm_rgb_joint@1`. There is no rgbd or depth schema.

## 2. Minimal change to get depth
**Schema change: none on the server side.** `validate_observation_input` (`3rd\policy_space\src\policy_space\conformance.py`, line 11) only checks that each required role is present, uint8 and HxWx3. It does not reject extra `images` keys, so a new key such as `images["camera_front_depth"]` passes validation.

Three small edits, all on the sim box. I did not test any of them.

1. **Enable the render and obs terms.** Set `camera_depth_semantic: true` in the robot cfg for the task. The field already exists in `robot_factory`'s robot config dataclass, so this is likely a one-line YAML change wherever the EX001-6R robot block lives; I did not locate that file. If the head camera should skip segmentation, edit line 641 in `ex001_6r.py` to drop `semantic_segmentation`, which saves render cost.
2. **Map depth into the wire.** In `policy_space.py` `_build_obs` (about line 1348), add a second pass for a depth map. For example, `{"head_cam_distance_to_image_plane": "camera_front_depth"}`:
   - Take the env item and `np.squeeze` it to HxW float32.
   - Skip `_to_rgb_uint8`, and resize with `cv2.INTER_NEAREST` only if the RGB image is resized.
   - Add the result to `images`.
   - The wire codec is msgpack-numpy (raw arrays), so float32 passes through. `image_encode: raw` is already set in the embodiment YAML.
   - About 15 lines. Also keep `self._camera_map` handling (lines 517-522) from treating the new keys as RGB.
3. **Consume it on our side.** In `labelloop/robots/x2real_server.py` `X2RealLink.step`:
   - Currently it reads only `obs["images"][src]` for the three RGB roles. Extra keys are ignored without error, so nothing breaks.
   - Add `depth = obs["images"].get("camera_front_depth")`.
   - Pass it to the harness as an optional argument (`harness.step(images, state, depth=None)`).
   - Do not forward depth to WALLX. `encode_jpeg_b64` is applied only to RGB images.
   - Note that `policies/labelloop/model.py` only passes `obs` through to the link. It keeps no key allowlist, so it needs no change.
   - Our harness would also need the head camera intrinsics and extrinsics (FOV and pose from `ex001_6r.py` lines 365-385) to convert depth to height or footprint.

**Effort:** about 2-3 hours including a smoke test. About 30 lines in `policy_space.py`, about 10 in our link, one config flag.

**Risks to the stock pipeline:**
- Editing `C:\x2\x2real` code is the main risk, since it is the organizers' tree and may be overwritten or checked. A fork or shim of `PolicySpaceImpl` is safer than patching it in place. A subclass in our own policy dir is not possible because the Impl runs in the sim process.
- Rendering cost: depth plus segmentation on 3 cameras adds GPU time and per-step payload. A 640x480 float32 depth image is about 1.2 MB per step; downscale or cast to uint16 millimetres to cut it.
- A depth-only obs path must not break the stock `bimanual_rgb_ee@1` runs. Keep it behind a flag that is off by default.
- Sim depth is perfect, with no noise or holes. A box-height routine tuned on it may not transfer to real RealSense-style depth.
- Depth is only enabled for `ex001_6r.py` (`camera_depth_semantic` exists only there). `ex001.py` only has a comment (line 225) about a chassis depth option.

## 3. What the event docs say about depth for the harness
- **BRIGHT Rules PDF (lines 243-244):** "Head-mounted and eye-in-hand RGB-D depth cameras providing high-frequency image streams." The robot hardware is RGB-D.
- **API Tutorial (condensed) and Participant Guide:** the model API takes three JPEG base64 cameras only (`observation.images.front`, `left_w...`, `right_...`, lines 378-384). The Agent/harness receives "the real-time three-camera feeds, robot states, and task objectives" (Guide, lines 639-642). There is no depth field, point cloud, RealSense or depth endpoint in any of the five PDFs.
- **Rules PDF (lines 250-251):** "APIs provide high-frequency joint angles, end-effector poses ... and real-time motor current/torque feedback." There is no mention of a depth API.
- So depth is a hardware capability, but the documented event client and harness path carries RGB only. Using depth on the real robot would need an undocumented low-level robot or camera SDK, which the docs do not offer. Ask the organizers whether it is exposed.

## 4. Recommendation
- **Do not build the depth path as a core dependency.** The documented harness input is RGB-only, so a depth-dependent routine would work in sim and likely not on the real robot, and the sim-vs-real gap is a risk.
- Whether depth is available to the harness on the real robot is unconfirmed. Ask the organizers first.
- If the organizers confirm a depth stream on the real robot, do the 3-edit change above. It is cheap, and the schema accepts the extra key.
- Otherwise keep box height and footprint estimates RGB-only (known box sizes from the label, grasp feedback from the gripper and current).
- If you want sim-only experiments for height estimation, enable `camera_depth_semantic` and the small `_build_obs` patch on a copy of the sim tree, off by default.
