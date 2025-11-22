"""Demo: differentiable IK for Franka Panda using pytorch_kinematics.DifferentiableIK

This script shows how to:
 - build a SerialChain from `franka_panda.urdf`
 - sample a ground-truth joint configuration and compute its end-effector pose
 - run the differentiable IK solver `DifferentiableIK.solve_unrolled` to recover joint angles
 - verify forward kinematics of the recovered joints matches the target pose
 - demonstrate backpropagating a scalar loss through the IK solver to the target pose

Run (from repository root):
    python panda_ik_demo.py

Note: this demo expects the repository to be importable (i.e. run from repo root or add repo to PYTHONPATH).
"""

import torch
import pytorch_kinematics as pk
from pytorch_kinematics.ik import ee_pose_to_joint_positions

class DiffIk:
    def __init__(self, device="cpu", dtype=torch.float32, iters = 50):
        self.device = torch.device(device)
        self.dtype = dtype

        urdf_path = "franka_panda.urdf"
        with open(urdf_path, "r") as f:
            urdf = f.read()

        self.chain = pk.build_serial_chain_from_urdf(urdf, "panda_hand").to(device=device, dtype=dtype)
        self.iters = iters
        self.init_q = None
    def get_ik(self, pos, rot):
        self.init_q = ee_pose_to_joint_positions(serial_chain = self.chain,
                                ee_pos = pos,
                                ee_rot = rot,
                                rot_type = "quat",
                                iters = self.iters,
                                lr = self.lr,
                                orientation_weight = self.orientation_weight,
                                init_q =self.init_q,
                                device = self.device,
                                dtype = self.dtype)
        return self.init_q

def oldmain(device="cpu", dtype=torch.float32):
    device = torch.device(device)

    # 1) Build the chain for the Panda (uses the URDF in the repo root)
    urdf_path = "franka_panda.urdf"
    with open(urdf_path, "r") as f:
        urdf = f.read()

    chain = pk.build_serial_chain_from_urdf(urdf, "panda_hand").to(device=device, dtype=dtype)
    dof = len(chain.get_joint_parameter_names())
    print(f"Loaded chain with DOF={dof}")

    # 2) Create a ground-truth joint configuration and compute its EE pose
    # Use a random but repeatable q within joint limits (if available)
    try:
        limits = chain.get_joint_limits()  # returns (low, high)
        low = torch.tensor(limits[0], device=device, dtype=dtype)
        high = torch.tensor(limits[1], device=device, dtype=dtype)
        q_gt = (low + high) * 0.5  # midpoint
    except Exception:
        q_gt = torch.zeros(dof, device=device, dtype=dtype)

    print("Ground-truth joint angles (q_gt):", q_gt.cpu().numpy())

    # make batched (N=1)
    q_gt_b = q_gt.unsqueeze(0)
    ee_tf = chain.forward_kinematics(q_gt_b)
    target_tf = ee_tf  # Transform3d

    print("Target end-effector pose (homogeneous matrix):")
    print(target_tf.get_matrix()[0].cpu().numpy())

    # 3) Create differentiable IK solver and solve
    dik = pk.DifferentiableIK(chain, iters=40, lr=0.1, orientation_weight=1.0)

    # Solve: returns (N, DOF) tensor with grad graph to the target pose
    q_sol = dik.solve_unrolled(target_tf)

    print("Solved joint angles (q_sol):", q_sol.detach().cpu().numpy())

    # 4) Verify forward kinematics of q_sol matches target
    fk_sol_tf = chain.forward_kinematics(q_sol)
    fk_mat = fk_sol_tf.get_matrix()
    target_mat = target_tf.get_matrix()

    pos_err = (fk_mat[:, :3, 3] - target_mat[:, :3, 3]).norm(dim=-1)
    # for rotation error, convert to quaternions and measure axis-angle norm
    targ_q = pk.transforms.rotation_conversions.matrix_to_quaternion(target_mat[:, :3, :3])
    sol_q = pk.transforms.rotation_conversions.matrix_to_quaternion(fk_mat[:, :3, :3])
    # quaternion that rotates sol -> targ: targ * inv(sol)
    diff_q = pk.transforms.rotation_conversions.quaternion_multiply(targ_q, pk.transforms.rotation_conversions.quaternion_invert(sol_q))
    axis_angle = pk.transforms.rotation_conversions.quaternion_to_axis_angle(diff_q)
    rot_err = axis_angle.norm(dim=-1)

    print(f"Position error (m): {pos_err.cpu()}")
    print(f"Rotation error (rad): {rot_err.cpu()}")

    # 5) Demonstrate gradient flow: create a small scalar loss and backprop to target translation
    # We'll make the target pose require gradients by perturbing it slightly and setting requires_grad
    target_mat = target_mat.clone().detach().requires_grad_(True)

    # wrap into a Transform3d-like object that the solver expects; Transform3d can be built from matrix
    target_tf_var = pk.transforms.Transform3d(matrix=target_mat)

    q_sol2 = dik.solve_unrolled(target_tf_var)

    # Helper: compute joint positions in end-effector frame (end-effector at origin)
    def joint_positions_in_eef_frame(chain, q):
        """Return joint positions in end-effector frame for batch q.

        Args:
            chain: SerialChain
            q: tensor (N, DOF)

        Returns:
            positions: tensor (N, K, 3) where K is number of serial frames (joints)
                       positions are expressed in end-effector coordinates (end effector at origin)
        """
        # compute all frame transforms
        all_tfs = chain.forward_kinematics(q, end_only=False)
        # collect joint/world positions in serial frame order
        joints_world = []
        for f in chain._serial_frames:
            tf_f = all_tfs[f.name]
            mat = tf_f.get_matrix()
            joints_world.append(mat[:, :3, 3])
        # stack into (N, K, 3)
        joints_world = torch.stack(joints_world, dim=1)
        # end-effector transform
        ee_tf = all_tfs[chain._serial_frames[-1].name]
        ee_inv = ee_tf.inverse()
        # manually apply batched homogeneous transform to avoid external linalg helpers
        # joints_world: (N, K, 3)
        N = joints_world.shape[0]
        K = joints_world.shape[1]
        device = joints_world.device
        dtype = joints_world.dtype
        ones = torch.ones((N, K, 1), device=device, dtype=dtype)
        joints_h = torch.cat([joints_world, ones], dim=-1)  # (N, K, 4)
        ee_inv_mat = ee_inv.get_matrix()  # (N, 4, 4)
        transformed = torch.matmul(joints_h, ee_inv_mat.transpose(-1, -2))  # (N, K, 4)
        # normalize homogeneous coordinate
        denom = transformed[..., 3:].clamp(min=1e-8)
        joints_in_ee = transformed[..., :3] / denom
        return joints_in_ee

    # compute joint positions (these depend on q_sol2 which depends on target_tf_var)
    joints_in_ee = joint_positions_in_eef_frame(chain, q_sol2)

    # simple loss: bring all joint positions close to the origin (example). This will produce gradients
    # that flow back from joint positions -> q_sol2 -> target_tf_var (pose)
    loss = joints_in_ee.pow(2).sum()
    loss.backward()

    # gradient w.r.t. target translation (the [0:3,3] entries)
    grad_pos = target_mat.grad[0, :3, 3]
    print("Gradient of loss w.r.t. target translation:", grad_pos.cpu().numpy())
    # Final joint positions: prefer the solution computed with the differentiable target (q_sol2)
    final_q = q_sol2.detach().cpu().numpy()
    print("Final joint positions (rad):", final_q)

    # Also provide joint XYZ positions in end-effector frame (N, K, 3)
    joint_xyz = joints_in_ee.detach().cpu().numpy()
    print("Joint XYZ positions in end-effector frame (m):", joint_xyz[:,[0,2,4,5,6,8,9]])
    try:
        import numpy as _np
        _np.save("panda_joint_positions.npy", joint_xyz[:0,2,4,5,6,8,9])
        print("Saved joint XYZ positions to panda_joint_positions.npy")
    except Exception:
        pass

    # save to file for easy consumption by other tools/scripts
    try:
        import numpy as _np
        _np.save("panda_ik_solution.npy", final_q)
        print("Saved final joint positions to panda_ik_solution.npy")
    except Exception:
        # numpy not available or save failed; continue silently
        pass

    print("Demo complete.")


if __name__ == '__main__':
    main()
