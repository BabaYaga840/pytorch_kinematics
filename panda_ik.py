import pytorch_kinematics as pk
import torch
from timeit import default_timer as timer

device = "cuda"

urdf_path = "franka_panda.urdf"
chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), "panda_hand")
chain = chain.to(device=device)

# robot frame
pos = torch.tensor([0.0, 0.0, 0.0], device=device)
rot = torch.tensor([0.0, 0.0, 0.0], device=device)
rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

# world frame goal
M = 1000
# generate random goal joint angles (so these are all achievable)
# use the joint limits to generate random joint angles
lim = torch.tensor(chain.get_joint_limits(), device=device)
goal_q = torch.rand(M, 7, device=device) * (lim[1] - lim[0]) + lim[0]

# get ee pose (in robot frame)
goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

# transform to world frame for visualization
goal_tf = rob_tf.compose(goal_in_rob_frame_tf)
goal = goal_tf.get_matrix()
goal_pos = goal[..., :3, 3]
goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

ik = pk.PseudoInverseIK(
    chain,
    max_iterations=30,
    num_retries=10,
    joint_limits=lim.T,
    early_stopping_any_converged=True,
    early_stopping_no_improvement="all",
    # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
    debug=False,
    lr=0.2,
)

# do IK
timer_start = timer()
sol = ik.solve(goal_in_rob_frame_tf)
timer_end = timer()
print("IK took %f seconds" % (timer_end - timer_start))
print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.numel()))
print("IK took %d iterations" % sol.iterations)
print("IK solved %d / %d goals" % (sol.converged_any.sum(), M))

# check that solving again produces the same solutions
sol_again = ik.solve(goal_in_rob_frame_tf)
assert torch.allclose(sol.solutions, sol_again.solutions)
assert torch.allclose(sol.converged, sol_again.converged)