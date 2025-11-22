from pytorch_kinematics.chain import SerialChain
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.transforms import rotation_conversions
from typing import NamedTuple, Union, Optional, Callable
import typing
import torch
import inspect
from matplotlib import pyplot as plt, cm as cm


class IKSolution:
    def __init__(self, dof, num_problems, num_retries, pos_tolerance, rot_tolerance, device="cpu"):
        self.iterations = 0
        self.device = device
        self.num_problems = num_problems
        self.num_retries = num_retries
        self.dof = dof
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance

        M = num_problems
        # N x DOF tensor of joint angles; if converged[i] is False, then solutions[i] is undefined
        self.solutions = torch.zeros((M, self.num_retries, self.dof), device=self.device)
        self.remaining = torch.ones(M, dtype=torch.bool, device=self.device)

        # M is the total number of problems
        # N is the total number of attempts
        # M x N tensor of position and rotation errors
        self.err_pos = torch.zeros((M, self.num_retries), device=self.device)
        self.err_rot = torch.zeros_like(self.err_pos)
        # M x N boolean values indicating whether the solution converged (a solution could be found)
        self.converged_pos = torch.zeros((M, self.num_retries), dtype=torch.bool, device=self.device)
        self.converged_rot = torch.zeros_like(self.converged_pos)
        self.converged = torch.zeros_like(self.converged_pos)

        # M whether any position and rotation converged for that problem
        self.converged_pos_any = torch.zeros_like(self.remaining)
        self.converged_rot_any = torch.zeros_like(self.remaining)
        self.converged_any = torch.zeros_like(self.remaining)

    def update_remaining_with_keep_mask(self, keep: torch.tensor):
        self.remaining = self.remaining & keep
        return self.remaining

    def update(self, q: torch.tensor, err: torch.tensor, use_keep_mask=True, keep_mask=None):
        err = err.reshape(-1, self.num_retries, 6)
        err_pos = err[..., :3].norm(dim=-1)
        err_rot = err[..., 3:].norm(dim=-1)
        converged_pos = err_pos < self.pos_tolerance
        converged_rot = err_rot < self.rot_tolerance
        converged = converged_pos & converged_rot
        converged_any = converged.any(dim=1)

        if keep_mask is None:
            keep_mask = ~converged_any

        # stop considering problems where any converged
        qq = q.reshape(-1, self.num_retries, self.dof)

        if use_keep_mask:
            # those that have converged are no longer remaining
            self.update_remaining_with_keep_mask(keep_mask)

        self.solutions = qq
        self.err_pos = err_pos
        self.err_rot = err_rot
        self.converged_pos = converged_pos
        self.converged_rot = converged_rot
        self.converged = converged
        self.converged_any = converged_any

        return converged_any


# helper config sampling method
def gaussian_around_config(config: torch.Tensor, std: float) -> Callable[[int], torch.Tensor]:
    def config_sampling_method(num_configs):
        return torch.randn(num_configs, config.shape[0], dtype=config.dtype, device=config.device) * std + config

    return config_sampling_method


class LineSearch:
    def do_line_search(self, chain, q, dq, target_pos, target_wxyz, initial_dx, problem_remaining=None):
        raise NotImplementedError()


class BacktrackingLineSearch(LineSearch):
    def __init__(self, max_lr=1.0, decrease_factor=0.5, max_iterations=5, sufficient_decrease=0.01):
        self.initial_lr = max_lr
        self.decrease_factor = decrease_factor
        self.max_iterations = max_iterations
        self.sufficient_decrease = sufficient_decrease

    def do_line_search(self, chain, q, dq, target_pos, target_wxyz, initial_dx, problem_remaining=None):
        N = target_pos.shape[0]
        NM = q.shape[0]
        M = NM // N
        lr = torch.ones(NM, device=q.device) * self.initial_lr
        err = initial_dx.squeeze().norm(dim=-1)
        if problem_remaining is None:
            problem_remaining = torch.ones(N, dtype=torch.bool, device=q.device)
        remaining = torch.ones((N, M), dtype=torch.bool, device=q.device)
        # don't care about the ones that are no longer remaining
        remaining[~problem_remaining] = False
        remaining = remaining.reshape(-1)
        for i in range(self.max_iterations):
            if not remaining.any():
                break
            # try stepping with this learning rate
            q_new = q + lr.unsqueeze(1) * dq
            # evaluate the error
            m = chain.forward_kinematics(q_new).get_matrix()
            m = m.view(-1, M, 4, 4)
            dx, pos_diff, rot_diff = delta_pose(m, target_pos, target_wxyz)
            err_new = dx.squeeze().norm(dim=-1)
            # check if it's better
            improvement = err - err_new
            improved = improvement > self.sufficient_decrease
            # if it's better, we're done for those
            # if it's not better, reduce the learning rate
            lr[~improved] *= self.decrease_factor
            remaining = remaining & ~improved

        improvement = improvement.reshape(-1, M)
        improvement = improvement.mean(dim=1)
        return lr, improvement


class InverseKinematics:
    """Jacobian follower based inverse kinematics solver"""

    def __init__(self, serial_chain: SerialChain,
                 pos_tolerance: float = 1e-3, rot_tolerance: float = 1e-2,
                 retry_configs: Optional[torch.Tensor] = None, num_retries: Optional[int] = None,
                 joint_limits: Optional[torch.Tensor] = None,
                 config_sampling_method: Union[str, Callable[[int], torch.Tensor]] = "uniform",
                 max_iterations: int = 50,
                 lr: float = 0.2, line_search: Optional[LineSearch] = None,
                 regularlization: float = 1e-9,
                 debug=False,
                 early_stopping_any_converged=False,
                 early_stopping_no_improvement="any", early_stopping_no_improvement_patience=2,
                 optimizer_method: Union[str, typing.Type[torch.optim.Optimizer]] = "sgd"
                 ):
        """
        :param serial_chain:
        :param pos_tolerance: position tolerance in meters
        :param rot_tolerance: rotation tolerance in radians
        :param retry_configs: (M, DOF) tensor of initial configs to try for each problem; leave as None to sample
        :param num_retries: number, M, of random initial configs to try for that problem; implemented with batching
        :param joint_limits: (DOF, 2) tensor of joint limits (min, max) for each joint in radians
        :param config_sampling_method: either "uniform" or "gaussian" or a function that takes in the number of configs
        :param max_iterations: maximum number of iterations to run
        :param lr: learning rate
        :param line_search: LineSearch object to use for line search
        :param regularlization: regularization term to add to the Jacobian
        :param debug: whether to print debug information
        :param early_stopping_any_converged: whether to stop when any of the retries for a problem converged
        :param early_stopping_no_improvement: {None, "all", "any", ratio} whether to stop when no improvement is made
        (consecutive iterations no improvement in minimum error - number of consecutive iterations is the patience).
        None means no early stopping from this, "all" means stop when all retries for that problem makes no improvement,
        "any" means stop when any of the retries for that problem makes no improvement, and ratio means stop when
        the ratio (between 0 and 1) of the number of retries that is making improvement falls below the ratio.
        So "all" is equivalent to ratio=0.999, and "any" is equivalent to ratio=0.001
        :param early_stopping_no_improvement_patience: number of consecutive iterations with no improvement before
        considering it no improvement
        :param optimizer_method: either a string or a torch.optim.Optimizer class
        """
        self.chain = serial_chain
        self.dtype = serial_chain.dtype
        self.device = serial_chain.device
        joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(joint_names)
        self.debug = debug
        self.early_stopping_any_converged = early_stopping_any_converged
        self.early_stopping_no_improvement = early_stopping_no_improvement
        self.early_stopping_no_improvement_patience = early_stopping_no_improvement_patience

        self.max_iterations = max_iterations
        self.lr = lr
        self.regularlization = regularlization
        self.optimizer_method = optimizer_method
        self.line_search = line_search

        self.err = None
        self.err_all = None
        self.err_min = None
        self.no_improve_counter = None

        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance
        self.initial_config = retry_configs
        if retry_configs is None and num_retries is None:
            raise ValueError("either initial_configs or num_retries must be specified")

        # sample initial configs instead
        self.config_sampling_method = config_sampling_method
        self.joint_limits = joint_limits
        if retry_configs is None:
            self.initial_config = self.sample_configs(num_retries)
        else:
            if retry_configs.shape[1] != self.dof:
                raise ValueError("initial_configs must have shape (N, %d)" % self.dof)
        # could give a batch of initial configs
        self.num_retries = self.initial_config.shape[-2]

    def clear(self):
        self.err = None
        self.err_all = None
        self.err_min = None
        self.no_improve_counter = None

    def sample_configs(self, num_configs: int) -> torch.Tensor:
        if self.config_sampling_method == "uniform":
            # bound by joint_limits
            if self.joint_limits is None:
                raise ValueError("joint_limits must be specified if config_sampling_method is uniform")
            return torch.rand(num_configs, self.dof, device=self.device) * (
                    self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        elif self.config_sampling_method == "gaussian":
            return torch.randn(num_configs, self.dof, device=self.device)
        elif callable(self.config_sampling_method):
            return self.config_sampling_method(num_configs)
        else:
            raise ValueError("invalid config_sampling_method %s" % self.config_sampling_method)

    def solve(self, target_poses: Transform3d) -> IKSolution:
        """
        Solve IK for the given target poses in robot frame
        :param target_poses: (N, 4, 4) tensor, goal pose in robot frame
        :return: IKSolution solutions
        """
        raise NotImplementedError()


def delta_pose(m: torch.tensor, target_pos, target_wxyz):
    """
    Determine the error in position and rotation between the given poses and the target poses

    :param m: (N x M x 4 x 4) tensor of homogenous transforms
    :param target_pos:
    :param target_wxyz: target orientation represented in unit quaternion
    :return: (N*M, 6, 1) tensor of delta pose (dx, dy, dz, droll, dpitch, dyaw)
    """
    pos_diff = target_pos.unsqueeze(1) - m[:, :, :3, 3]
    pos_diff = pos_diff.view(-1, 3, 1)
    cur_wxyz = rotation_conversions.matrix_to_quaternion(m[:, :, :3, :3])

    # quaternion that rotates from the current orientation to the desired orientation
    # inverse for unit quaternion is the conjugate
    diff_wxyz = rotation_conversions.quaternion_multiply(target_wxyz.unsqueeze(1),
                                                         rotation_conversions.quaternion_invert(cur_wxyz))
    # angular velocity vector needed to correct the orientation
    # if time is considered, should divide by \delta t, but doing it iteratively we can choose delta t to be 1
    diff_axis_angle = rotation_conversions.quaternion_to_axis_angle(diff_wxyz)

    rot_diff = diff_axis_angle.view(-1, 3, 1)

    dx = torch.cat((pos_diff, rot_diff), dim=1)
    return dx, pos_diff, rot_diff


def apply_mask(mask, *args):
    return [a[mask] for a in args]


class PseudoInverseIK(InverseKinematics):
    def compute_dq(self, J, dx):
        # lambda^2*I (lambda^2 is regularization)
        reg = self.regularlization * torch.eye(6, device=self.device, dtype=self.dtype)

        # JJ^T + lambda^2*I (lambda^2 is regularization)
        tmpA = J @ J.transpose(1, 2) + reg
        # (JJ^T + lambda^2I) A = dx
        # A = (JJ^T + lambda^2I)^-1 dx
        A = torch.linalg.solve(tmpA, dx)
        # dq = J^T (JJ^T + lambda^2I)^-1 dx
        dq = J.transpose(1, 2) @ A
        return dq

    def solve(self, target_poses: Transform3d) -> IKSolution:
        self.clear()

        target = target_poses.get_matrix()

        M = target.shape[0]

        target_pos = target[:, :3, 3]
        # jacobian gives angular rotation about x,y,z axis of the base frame
        # convert target rot to desired rotation about x,y,z
        target_wxyz = rotation_conversions.matrix_to_quaternion(target[:, :3, :3])

        sol = IKSolution(self.dof, M, self.num_retries, self.pos_tolerance, self.rot_tolerance, device=self.device)

        q = self.initial_config
        if q.numel() == M * self.dof * self.num_retries:
            q = q.reshape(-1, self.dof)
        elif q.numel() == self.dof * self.num_retries:
            # repeat and manually flatten it
            q = self.initial_config.repeat(M, 1)
        elif q.numel() == self.dof:
            q = q.unsqueeze(0).repeat(M * self.num_retries, 1)
        else:
            raise ValueError(
                f"initial_config must have shape ({M}, {self.num_retries}, {self.dof}) or ({self.num_retries}, {self.dof})")
        # for logging, let's keep track of the joint angles at each iteration
        if self.debug:
            pos_errors = []
            rot_errors = []

        optimizer = None
        if inspect.isclass(self.optimizer_method) and issubclass(self.optimizer_method, torch.optim.Optimizer):
            q.requires_grad = True
            optimizer = torch.optim.Adam([q], lr=self.lr)
        for i in range(self.max_iterations):
            with torch.no_grad():
                # early termination if we're out of problems to solve
                if not sol.remaining.any():
                    break
                sol.iterations += 1
                # compute forward kinematics
                # N x 6 x DOF
                J, m = self.chain.jacobian(q, ret_eef_pose=True)
                # unflatten to broadcast with goal
                m = m.view(-1, self.num_retries, 4, 4)
                dx, pos_diff, rot_diff = delta_pose(m, target_pos, target_wxyz)

                # damped least squares method
                # lambda^2*I (lambda^2 is regularization)
                dq = self.compute_dq(J, dx)
                dq = dq.squeeze(2)

            improvement = None
            if optimizer is not None:
                q.grad = -dq
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    if self.line_search is not None:
                        lr, improvement = self.line_search.do_line_search(self.chain, q, dq, target_pos, target_wxyz,
                                                                          dx, problem_remaining=sol.remaining)
                        lr = lr.unsqueeze(1)
                    else:
                        lr = self.lr
                    q = q + lr * dq

            with torch.no_grad():
                self.err_all = dx.squeeze()
                self.err = self.err_all.norm(dim=-1)
                sol.update(q, self.err_all, use_keep_mask=self.early_stopping_any_converged)

                if self.early_stopping_no_improvement is not None:
                    if self.no_improve_counter is None:
                        self.no_improve_counter = torch.zeros_like(self.err)
                    else:
                        if self.err_min is None:
                            self.err_min = self.err.clone()
                        else:
                            improved = self.err < self.err_min
                            self.err_min[improved] = self.err[improved]

                            self.no_improve_counter[improved] = 0
                            self.no_improve_counter[~improved] += 1

                            # those that haven't improved
                            could_improve = self.no_improve_counter <= self.early_stopping_no_improvement_patience
                            # consider problems, and only throw out those whose all retries cannot be improved
                            could_improve = could_improve.reshape(-1, self.num_retries)
                            if self.early_stopping_no_improvement == "all":
                                could_improve = could_improve.all(dim=1)
                            elif self.early_stopping_no_improvement == "any":
                                could_improve = could_improve.any(dim=1)
                            elif isinstance(self.early_stopping_no_improvement, float):
                                ratio_improved = could_improve.sum(dim=1) / self.num_retries
                                could_improve = ratio_improved > self.early_stopping_no_improvement
                            sol.update_remaining_with_keep_mask(could_improve)

                if self.debug:
                    pos_errors.append(pos_diff.reshape(-1, 3).norm(dim=1))
                    rot_errors.append(rot_diff.reshape(-1, 3).norm(dim=1))

        if self.debug:
            # errors
            fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
            pos_e = torch.stack(pos_errors, dim=0).cpu()
            rot_e = torch.stack(rot_errors, dim=0).cpu()
            ax[0].set_ylim(0, 1.)
            # ignore nan
            ignore = torch.isnan(rot_e)
            axis_max = rot_e[~ignore].max().item()
            ax[1].set_ylim(0, axis_max * 1.1)
            ax[0].set_xlim(0, self.max_iterations - 1)
            ax[1].set_xlim(0, self.max_iterations - 1)
            # draw at most 50 lines
            draw_max = min(50, pos_e.shape[1])
            for b in range(draw_max):
                c = (b + 1) / draw_max
                ax[0].plot(pos_e[:, b], c=cm.GnBu(c))
                ax[1].plot(rot_e[:, b], c=cm.GnBu(c))
            # label these axis
            ax[0].set_ylabel("position error")
            ax[1].set_xlabel("iteration")
            ax[1].set_ylabel("rotation error")
            plt.show()


class DifferentiableIK(InverseKinematics):
    """
    Differentiable inverse kinematics solver using unrolled gradient-based optimization.

    This solver performs a fixed number of gradient descent steps on the joint
    angles q to minimize the pose error to the target pose. Each update is
    constructed with create_graph=True so the final joint angles remain
    differentiable with respect to the target pose (and any inputs used to
    compute the loss).
    """

    def __init__(self, serial_chain: SerialChain, iters: int = 50, lr: float = 1e-2,
                 orientation_weight: float = 1.0, pos_tolerance: float = 1e-4,
                 rot_tolerance: float = 1e-3, device="cpu", dtype=torch.float32):
        # We still call the base constructor with minimal required args
        # Provide a dummy initial_config and num_retries to satisfy the base class
        dummy_init = torch.zeros((1, len(serial_chain.get_joint_parameter_names())), dtype=dtype, device=device)
        super().__init__(serial_chain, pos_tolerance=pos_tolerance, rot_tolerance=rot_tolerance,
                         retry_configs=dummy_init, num_retries=1, joint_limits=None,
                         max_iterations=iters, lr=lr)
        self.iters = iters
        self.orientation_weight = orientation_weight

    def solve_unrolled(self, target_poses: Transform3d, init_q: Optional[torch.Tensor] = None,
                       lr: Optional[float] = None):
        """
        Solve IK differentiably by unrolling gradient descent.

        Args:
            target_poses: Transform3d object of shape (N, 4, 4) representing goal poses.
            init_q: Optional initial joint angles tensor of shape (N, DOF). If None,
                    initializes to mid-point of joint limits if available, else zeros.
            lr: learning rate for the unrolled gradient descent (overrides constructor lr)

        Returns:
            q: tensor of shape (N, DOF) containing the solved joint angles. The returned
               tensor maintains a computational graph connecting it to `target_poses`.
        """
        if lr is None:
            lr = self.lr

        target = target_poses.get_matrix()
        N = target.shape[0]
        target_pos = target[:, :3, 3]
        target_wxyz = rotation_conversions.matrix_to_quaternion(target[:, :3, :3])

        dof = self.dof
        device = self.device
        dtype = self.dtype

        if init_q is None:
            # try mid joint limits, fall back to zeros
            try:
                low, high = self.chain.get_joint_limits()
                low = torch.tensor(low, device=device, dtype=dtype)
                high = torch.tensor(high, device=device, dtype=dtype)
                init_q = ((low + high) / 2.0).unsqueeze(0).repeat(N, 1)
            except Exception:
                init_q = torch.zeros((N, dof), device=device, dtype=dtype)
        else:
            init_q = init_q.to(device=device, dtype=dtype)

        q = init_q.clone().requires_grad_(True)

        for i in range(self.iters):
            # compute forward kinematics for current q
            pose_tf = self.chain.forward_kinematics(q)  # Transform3d for end effector
            m = pose_tf.get_matrix()  # (N, 4, 4)
            # delta_pose expects shape (N, M, 4, 4); we use M=1
            m_exp = m.unsqueeze(1)
            dx, pos_diff, rot_diff = delta_pose(m_exp, target_pos, target_wxyz)
            # dx has shape (N*1, 6, 1) -> reshape
            dx = dx.squeeze(2).view(N, 6)
            pos_err = dx[:, :3].norm(dim=-1)
            rot_err = dx[:, 3:].norm(dim=-1)
            loss = pos_err.mean() + self.orientation_weight * rot_err.mean()

            # compute gradient of loss w.r.t q and take a differentiable gradient step
            grads = torch.autograd.grad(loss, q, create_graph=True)[0]
            q = q - lr * grads

        return q


def ee_pose_to_joint_positions(serial_chain: SerialChain,
                                ee_pos: torch.Tensor,
                                ee_rot: Optional[torch.Tensor] = None,
                                rot_type: str = "quat",
                                iters: int = 50,
                                lr: float = 1e-2,
                                orientation_weight: float = 1.0,
                                init_q: Optional[torch.Tensor] = None,
                                device: Optional[str] = None,
                                dtype: Optional[torch.dtype] = None):
    """
    Given an end-effector pose (position + orientation), solve IK differentiably and
    return the joint positions for each serial frame expressed in the end-effector frame
    (i.e. with the end-effector at the origin). This function is fully differentiable
    (assuming inputs are tensors with requires_grad as needed) because it uses
    DifferentiableIK which unrolls gradient steps.

    Args:
        serial_chain: SerialChain model for the robot (e.g., Panda).
        ee_pos: (..., 3) tensor of end-effector positions in meters.
        ee_rot: optional orientation. Interpretation depends on `rot_type`:
            - 'quat': quaternion [w,x,y,z] (...,4)
            - 'matrix': rotation matrix (...,3,3)
            - 'euler': Euler angles (...,3) using default convention
        rot_type: one of {'quat','matrix','euler'}; when ee_rot is None identity rotation is used.
        iters, lr, orientation_weight: DifferentiableIK solver hyperparameters.
        init_q: optional initial joint angles (..., DOF)
        device, dtype: override device/dtype; defaults to serial_chain.device/dtype

    Returns:
        joint_positions_in_eef: (N, K, 3) tensor of XYZ positions (meters) for each serial frame
                                expressed in the end-effector frame (end effector at origin).
        joint_angles: (N, DOF) tensor of joint angles (radians).

    Notes:
        - Units: positions are in meters (URDF convention) and angles in radians.
        - The function broadcasts single-input tensors to batch size N=1.
    """
    if device is None:
        device = serial_chain.device
    if dtype is None:
        dtype = serial_chain.dtype

    # ensure tensors and batch dims
    if not torch.is_tensor(ee_pos):
        ee_pos = torch.tensor(ee_pos, dtype=dtype, device=device)
    else:
        ee_pos = ee_pos.to(device=device, dtype=dtype)

    if ee_pos.dim() == 1:
        ee_pos = ee_pos.unsqueeze(0)

    if ee_rot is None:
        ee_rot_t = None
    else:
        if not torch.is_tensor(ee_rot):
            ee_rot_t = torch.tensor(ee_rot, dtype=dtype, device=device)
        else:
            ee_rot_t = ee_rot.to(device=device, dtype=dtype)
        if ee_rot_t.dim() == 1:
            ee_rot_t = ee_rot_t.unsqueeze(0)

    # Build Transform3d from provided pos/rot
    if ee_rot_t is None:
        target_tf = Transform3d(pos=ee_pos, device=device, dtype=dtype)
    else:
        if rot_type == "quat":
            target_tf = Transform3d(pos=ee_pos, rot=ee_rot_t, device=device, dtype=dtype)
        elif rot_type == "matrix":
            target_tf = Transform3d(pos=ee_pos, rot=ee_rot_t, device=device, dtype=dtype)
        elif rot_type == "euler":
            target_tf = Transform3d(pos=ee_pos, rot=ee_rot_t, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown rot_type {rot_type}")

    # Create differentiable IK solver
    dik = DifferentiableIK(serial_chain, iters=iters, lr=lr, orientation_weight=orientation_weight,
                           device=device, dtype=dtype)

    # Solve for joint angles (differentiable)
    if init_q is not None:
        if not torch.is_tensor(init_q):
            init_q = torch.tensor(init_q, dtype=dtype, device=device)
        init_q = init_q.to(device=device, dtype=dtype)
        if init_q.dim() == 1:
            init_q = init_q.unsqueeze(0)
        q_sol = dik.solve_unrolled(target_tf, init_q=init_q)
    else:
        q_sol = dik.solve_unrolled(target_tf)

    # Compute joint/world positions for all serial frames
    all_tfs = serial_chain.forward_kinematics(q_sol, end_only=False)
    joints_world = []
    for f in serial_chain._serial_frames:
        tf_f = all_tfs[f.name]
        mat = tf_f.get_matrix()
        joints_world.append(mat[:, :3, 3])
    joints_world = torch.stack(joints_world, dim=1)  # (N, K, 3)

    # Transform positions into end-effector frame (end effector at origin)
    ee_tf = all_tfs[serial_chain._serial_frames[-1].name]
    ee_inv = ee_tf.inverse()
    N = joints_world.shape[0]
    K = joints_world.shape[1]
    ones = torch.ones((N, K, 1), device=device, dtype=dtype)
    joints_h = torch.cat([joints_world, ones], dim=-1)  # (N, K, 4)
    ee_inv_mat = ee_inv.get_matrix()  # (N, 4, 4)
    transformed = torch.matmul(joints_h, ee_inv_mat.transpose(-1, -2))  # (N, K, 4)
    denom = transformed[..., 3:].clamp(min=1e-8)
    joints_in_ee = transformed[..., :3] / denom
    joints_in_ee = joints_in_ee[:, [0,2,4,5,6,8, 9]]
    return joints_in_ee  #, q_sol


def robosuite_ee_world_to_joint_positions(serial_chain: SerialChain,
                                          ee_pos_world: torch.Tensor,
                                          ee_rot_world: Optional[torch.Tensor] = None,
                                          rot_type: str = "quat",
                                          T_chain_base_in_world: Optional[torch.Tensor] = None,
                                          iters: int = 50,
                                          lr: float = 1e-2,
                                          orientation_weight: float = 1.0,
                                          init_q: Optional[torch.Tensor] = None,
                                          device: Optional[str] = None,
                                          dtype: Optional[torch.dtype] = None,
                                          debug: bool = False):
    """
    Convert a robosuite end-effector pose (in world frame) to the chain frame,
    run the differentiable IK, and convert resulting joint positions back to world frame.
    
    The function handles the conversion between Robosuite's world frame and the robot's base frame:
    1. Transforms target pose from world frame to robot base frame
    2. Solves IK in the robot base frame
    3. Transforms resulting joint positions back to world frame
    
    Args:
        serial_chain: SerialChain model for the robot
        ee_pos_world: End-effector position in world frame (N,3)
        ee_rot_world: Optional end-effector orientation in world frame
        rot_type: Type of rotation representation ('quat','matrix','euler')
        T_chain_base_in_world: Transform from world to robot base (4,4 or N,4,4)
        iters: Number of optimization iterations
        lr: Learning rate for optimization
        orientation_weight: Weight for orientation error in optimization
        init_q: Initial guess for joint angles
        device: Computation device
        dtype: Data type for computations

    Returns:
        joints_world: (N, K, 3) joint positions in world frame (meters)
        q_sol: (N, DOF) joint angles in radians
    """
    if device is None:
        device = serial_chain.device
    if dtype is None:
        dtype = serial_chain.dtype

    # prepare tensors
    if not torch.is_tensor(ee_pos_world):
        ee_pos_world_t = torch.tensor(ee_pos_world, dtype=dtype, device=device)
    else:
        ee_pos_world_t = ee_pos_world.to(device=device, dtype=dtype)
    if ee_pos_world_t.dim() == 1:
        ee_pos_world_t = ee_pos_world_t.unsqueeze(0)

    # prepare T_chain_base_in_world
    # Handle robot base transform
    if T_chain_base_in_world is None:
        # If no transform provided, create transform considering robot's base position and orientation
        # Assumes robot base is at world origin with identity rotation unless specified
        T_world_chain = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(ee_pos_world_t.shape[0], 1, 1)
    else:
        # Allow passing either a 4x4 transform or a 3-vector position
        if not torch.is_tensor(T_chain_base_in_world):
            T_in = torch.tensor(T_chain_base_in_world, dtype=dtype, device=device)
        else:
            T_in = T_chain_base_in_world.to(device=device, dtype=dtype)
            
        # Handle the EE offset and rotation from the URDF/XML definition
        # Note: This assumes the right_hand frame is the end effector

        # If a translation vector is provided (3,) or (N,3), build a homogeneous transform with identity rotation
        if T_in.dim() == 1 and T_in.numel() == 3:
            # single translation
            T_world_chain = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(ee_pos_world_t.shape[0], 1, 1)
            T_world_chain[:, :3, 3] = T_in.unsqueeze(0)
        elif T_in.dim() == 2 and T_in.shape[-1] == 3:
            # batched translations (N,3)
            T_world_chain = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(T_in.shape[0], 1, 1)
            T_world_chain[:, :3, 3] = T_in
            if T_world_chain.shape[0] == 1 and ee_pos_world_t.shape[0] > 1:
                T_world_chain = T_world_chain.repeat(ee_pos_world_t.shape[0], 1, 1)
        else:
            # assume full homogeneous matrix or batched matrices
            T_world_chain = T_in
            if T_world_chain.dim() == 2:
                T_world_chain = T_world_chain.unsqueeze(0)
            if T_world_chain.shape[0] == 1 and ee_pos_world_t.shape[0] > 1:
                T_world_chain = T_world_chain.repeat(ee_pos_world_t.shape[0], 1, 1)

    # Convert from robosuite world frame to pytorch_kinematics base frame
    # This includes both translation and coordinate frame alignment
    ee_pos_chain = ee_pos_world_t.clone()
    
    # First, create the coordinate frame transformation matrix
    # This handles the conversion between robosuite and pytorch_kinematics conventions
    # Only Y axis needs to be flipped, X and Z remain the same
    R_align = torch.eye(3, device=device, dtype=dtype)
    R_align[1, 1] = -1  # Flip only Y axis to align frames
    
    if T_chain_base_in_world is not None:
        if T_chain_base_in_world.dim() == 1:
            # 1. Apply translation to robot base
            base_pos = T_chain_base_in_world[:3]
            if ee_pos_chain.dim() == 2:
                base_pos = base_pos.unsqueeze(0)
                R_align = R_align.unsqueeze(0)
            
            # 2. Translate to robot base origin
            ee_pos_chain = ee_pos_world_t - base_pos
            
            # 3. Apply coordinate frame alignment
            ee_pos_chain = torch.matmul(R_align, ee_pos_chain.unsqueeze(-1)).squeeze(-1)
            
        else:
            # Full transform case including coordinate frame alignment
            # Create a full 4x4 transformation that includes both translation and rotation
            T_align = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(ee_pos_world_t.shape[0], 1, 1)
            T_align[:, :3, :3] = R_align
            
            # Combine with base transform
            T_full = torch.matmul(torch.linalg.inv(T_world_chain), T_align)
            
            # Apply transformation
            ones = torch.ones((ee_pos_world_t.shape[0], 1), device=device, dtype=dtype)
            ee_world_h = torch.cat([ee_pos_world_t, ones], dim=-1)  # (N,4)
            ee_chain_h = torch.matmul(T_full, ee_world_h.unsqueeze(-1)).squeeze(-1)  # (N,4)
            ee_pos_chain = ee_chain_h[:, :3]

    # orientation: transform rotation into chain frame if provided
    if ee_rot_world is None:
        ee_rot_chain = None
    else:
        if not torch.is_tensor(ee_rot_world):
            ee_rot_world_t = torch.tensor(ee_rot_world, dtype=dtype, device=device)
        else:
            ee_rot_world_t = ee_rot_world.to(device=device, dtype=dtype)
        if ee_rot_world_t.dim() == 1:
            ee_rot_world_t = ee_rot_world_t.unsqueeze(0)

        if rot_type == "quat":
            R_world = rotation_conversions.quaternion_to_matrix(ee_rot_world_t)
        elif rot_type == "matrix":
            R_world = ee_rot_world_t
        elif rot_type == "euler":
            R_world = rotation_conversions.euler_angles_to_matrix(ee_rot_world_t, "XYZ")
        else:
            raise ValueError(f"Unknown rot_type {rot_type}")

        # Transform rotation according to base transform
        if T_chain_base_in_world is not None:
            if T_chain_base_in_world.dim() == 1:  # Just translation
                ee_rot_chain = R_world  # No rotation transformation needed
            else:  # Full transform
                T_chain_world_inv = torch.linalg.inv(T_world_chain)
                R_chain = torch.matmul(T_chain_world_inv[:, :3, :3], R_world)
                if rot_type == "matrix":
                    ee_rot_chain = R_chain
                else:
                    ee_rot_chain = rotation_conversions.matrix_to_quaternion(R_chain)
        else:
            if rot_type == "matrix":
                ee_rot_chain = R_world
            else:
                ee_rot_chain = rotation_conversions.matrix_to_quaternion(R_world)

    # solve IK in chain frame and get joint angles
    # build Transform3d and call DifferentiableIK directly (similar to ee_pose_to_joint_positions)
    target_tf_chain = Transform3d(pos=ee_pos_chain, rot=ee_rot_chain, device=device, dtype=dtype) if ee_rot_chain is not None else Transform3d(pos=ee_pos_chain, device=device, dtype=dtype)
    dik = DifferentiableIK(serial_chain, iters=iters, lr=lr, orientation_weight=orientation_weight, device=device, dtype=dtype)
    if init_q is not None:
        if not torch.is_tensor(init_q):
            init_q = torch.tensor(init_q, dtype=dtype, device=device)
        if init_q.dim() == 1:
            init_q = init_q.unsqueeze(0)
        q_sol = dik.solve_unrolled(target_tf_chain, init_q=init_q)
    else:
        q_sol = dik.solve_unrolled(target_tf_chain)

    # compute joint positions in world frame
    all_tfs = serial_chain.forward_kinematics(q_sol, end_only=False)
    joints_world_chain = []
    for f in serial_chain._serial_frames:
        tf_f = all_tfs[f.name]
        mat = tf_f.get_matrix()
        joints_world_chain.append(mat[:, :3, 3])
    joints_world_chain = torch.stack(joints_world_chain, dim=1)  # (N, K, 3) in chain frame

    # Convert joint positions from pytorch_kinematics frame back to robosuite world frame
    N, K, _ = joints_world_chain.shape
    
    # Create coordinate frame alignment matrix (inverse of the previous one)
    # For Y-axis flip only, the inverse is the same as the forward transform
    R_align_inv = torch.eye(3, device=device, dtype=dtype)
    R_align_inv[1, 1] = -1  # Flip Y axis back
    
    if T_chain_base_in_world is not None:
        if T_chain_base_in_world.dim() == 1:
            # 1. First undo coordinate frame alignment
            if joints_world_chain.dim() == 3:
                R_align_inv = R_align_inv.unsqueeze(0).unsqueeze(1)
            joints_aligned = torch.matmul(R_align_inv, joints_world_chain.unsqueeze(-1)).squeeze(-1)
            
            # 2. Then translate back to world frame
            base_pos = T_chain_base_in_world[:3]
            if joints_world_chain.dim() == 3:
                base_pos = base_pos.view(1, 1, 3)
            joints_world = joints_aligned + base_pos
        else:
            # Full transform case
            # Create full alignment transform
            T_align_inv = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
            T_align_inv[:, :3, :3] = R_align_inv
            
            # Combine with world transform
            T_full = torch.matmul(T_world_chain, T_align_inv)
            
            # Apply transformation
            ones = torch.ones((N, K, 1), device=device, dtype=dtype)
            joints_h = torch.cat([joints_world_chain, ones], dim=-1)  # (N,K,4)
            
            if T_full.shape[0] == 1 and N > 1:
                T_full = T_full.repeat(N, 1, 1)
            
            # Transform points back to world frame
            joints_h_reshaped = joints_h.view(N*K, 4).transpose(0, 1)
            transformed = torch.matmul(T_full, joints_h_reshaped)
            transformed = transformed.transpose(1, 2).view(N, K, 4)
            
            denom = transformed[..., 3:].clamp(min=1e-8)
            joints_world = transformed[..., :3] / denom
    else:
        # Even without base transform, we still need to align coordinate frames
        if joints_world_chain.dim() == 3:
            R_align_inv = R_align_inv.unsqueeze(0).unsqueeze(1)
        joints_world = torch.matmul(R_align_inv, joints_world_chain.unsqueeze(-1)).squeeze(-1)
    
    # Optional debug: compute end-effector pose from q_sol and compare to requested world pose
    if debug:
        try:
            # forward kinematics (chain frame)
            all_tfs_sol = serial_chain.forward_kinematics(q_sol, end_only=False)
            ee_tf_sol = all_tfs_sol[serial_chain._serial_frames[-1].name]
            ee_mat_chain = ee_tf_sol.get_matrix()  # (N,4,4) in chain frame
            ee_pos_chain_sol = ee_mat_chain[:, :3, 3]

            # Transform achieved position from pytorch_kinematics frame back to robosuite world frame
            R_align_inv = torch.eye(3, device=device, dtype=dtype)
            R_align_inv[1, 1] = -1  # Flip Y axis back (same as forward for Y-only flip)
            
            if T_chain_base_in_world is not None:
                if T_chain_base_in_world.dim() == 1:
                    # 1. First undo coordinate frame alignment
                    if ee_pos_chain_sol.dim() == 2:
                        R_align_inv = R_align_inv.unsqueeze(0)
                    ee_pos_aligned = torch.matmul(R_align_inv, ee_pos_chain_sol.unsqueeze(-1)).squeeze(-1)
                    
                    # 2. Then translate back to world frame
                    base_pos = T_chain_base_in_world[:3]
                    if ee_pos_chain_sol.dim() == 2:
                        base_pos = base_pos.unsqueeze(0)
                    ee_world_pos_sol = ee_pos_aligned + base_pos
                else:
                    # Full transform case combining alignment and world transform
                    T_align_inv = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
                    T_align_inv[:, :3, :3] = R_align_inv
                    T_full = torch.matmul(T_world_chain, T_align_inv)
                    
                    ones_e = torch.ones((N, 1), device=device, dtype=dtype)
                    ee_h_chain = torch.cat([ee_pos_chain_sol, ones_e], dim=-1)  # (N,4)
                    ee_world_h = torch.matmul(T_full, ee_h_chain.unsqueeze(-1)).squeeze(-1)
                    ee_world_pos_sol = ee_world_h[:, :3]
            else:
                # Even without base transform, we still need to align coordinate frames
                if ee_pos_chain_sol.dim() == 2:
                    R_align_inv = R_align_inv.unsqueeze(0)
                ee_world_pos_sol = torch.matmul(R_align_inv, ee_pos_chain_sol.unsqueeze(-1)).squeeze(-1)

            # original target in world frame (ensure same device/dtype)
            tgt_world = ee_pos_world_t.to(device=device, dtype=dtype)
            if tgt_world.dim() == 1:
                tgt_world = tgt_world.unsqueeze(0)

            pos_err = (ee_world_pos_sol - tgt_world).norm(dim=1)
            print('\n[robosuite_ee_world_to_joint_positions debug]')
            print('T_world_chain =')
            print(T_world_chain)
            print('target ee world pos =', tgt_world)
            print('achieved ee world pos =', ee_world_pos_sol)
            print('ee position error (L2) =', pos_err)
        except Exception as e:
            print('Debug forward-kinematics check failed:', e)

    return joints_world, q_sol


def robosuite_ee_world_to_joint_positions_safe(serial_chain: SerialChain,
                                                ee_pos_world: torch.Tensor,
                                                ee_rot_world: Optional[torch.Tensor] = None,
                                                rot_type: str = "quat",
                                                T_chain_base_in_world: Optional[torch.Tensor] = None,
                                                max_reach: float = 0.5,
                                                project_if_unreachable: bool = True,
                                                warn_if_unreachable: bool = True,
                                                iters: int = 50,
                                                lr: float = 1e-2,
                                                orientation_weight: float = 1.0,
                                                init_q: Optional[torch.Tensor] = None,
                                                device: Optional[str] = None,
                                                dtype: Optional[torch.dtype] = None,
                                                debug: bool = False):
    """
    Safety wrapper around `robosuite_ee_world_to_joint_positions`.

    - Computes horizontal distance (XY) from robot base to the requested world target(s).
    - If any target lies outside `max_reach`, optionally project it onto the reachable disk
      (in the XY plane) and optionally warn.
    - Calls the original IK function with (possibly) projected targets and returns
      (joints_world, q_sol, info) where info contains metadata about projection.

    Returns:
        joints_world: (N, K, 3)
        q_sol: (N, DOF)
        info: dict with keys:
            - 'was_projected': Tensor(bool) shape (N,)
            - 'orig_targets': Tensor (N,3)
            - 'proj_targets': Tensor (N,3)
            - 'horiz_dist': Tensor (N,)
    """
    if device is None:
        device = serial_chain.device
    if dtype is None:
        dtype = serial_chain.dtype

    # ensure tensor form and batch
    if not torch.is_tensor(ee_pos_world):
        ee_pos_world_t = torch.tensor(ee_pos_world, dtype=dtype, device=device)
    else:
        ee_pos_world_t = ee_pos_world.to(device=device, dtype=dtype)
    if ee_pos_world_t.dim() == 1:
        ee_pos_world_t = ee_pos_world_t.unsqueeze(0)

    # build T_world_chain same as main function
    if T_chain_base_in_world is None:
        T_world_chain = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(ee_pos_world_t.shape[0], 1, 1)
    else:
        if not torch.is_tensor(T_chain_base_in_world):
            T_in = torch.tensor(T_chain_base_in_world, dtype=dtype, device=device)
        else:
            T_in = T_chain_base_in_world.to(device=device, dtype=dtype)
        if T_in.dim() == 1 and T_in.numel() == 3:
            T_world_chain = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(ee_pos_world_t.shape[0], 1, 1)
            T_world_chain[:, :3, 3] = T_in.unsqueeze(0)
        elif T_in.dim() == 2 and T_in.shape[-1] == 3:
            T_world_chain = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(T_in.shape[0], 1, 1)
            T_world_chain[:, :3, 3] = T_in
            if T_world_chain.shape[0] == 1 and ee_pos_world_t.shape[0] > 1:
                T_world_chain = T_world_chain.repeat(ee_pos_world_t.shape[0], 1, 1)
        else:
            T_world_chain = T_in
            if T_world_chain.dim() == 2:
                T_world_chain = T_world_chain.unsqueeze(0)
            if T_world_chain.shape[0] == 1 and ee_pos_world_t.shape[0] > 1:
                T_world_chain = T_world_chain.repeat(ee_pos_world_t.shape[0], 1, 1)

    # base positions (N,3)
    base_pos = T_world_chain[:, :3, 3]

    # compute horizontal distances
    diff_xy = ee_pos_world_t[:, :2] - base_pos[:, :2]
    horiz_dist = torch.linalg.norm(diff_xy, dim=1)

    was_projected = torch.zeros(horiz_dist.shape, dtype=torch.bool, device=device)
    proj_targets = ee_pos_world_t.clone()

    if (horiz_dist > max_reach).any():
        mask = horiz_dist > max_reach
        was_projected = mask.clone()
        if warn_if_unreachable:
            print(f"[robosuite_ee_world_to_joint_positions_safe] Warning: {mask.sum().item()} target(s) outside max_reach={max_reach:.3f} m; will {'project' if project_if_unreachable else 'not project'}.")
        if project_if_unreachable:
            # project each masked target onto circle of radius max_reach centered at base
            vx = diff_xy[mask, 0]
            vy = diff_xy[mask, 1]
            r = torch.linalg.norm(torch.stack([vx, vy], dim=1), dim=1)
            # avoid divide by zero
            r_clamped = r.clone()
            r_clamped[r_clamped == 0] = 1.0
            scale = (max_reach / r_clamped).unsqueeze(1)
            new_xy = torch.stack([vx, vy], dim=1) * scale
            proj_targets[mask, 0] = base_pos[mask, 0] + new_xy[:, 0]
            proj_targets[mask, 1] = base_pos[mask, 1] + new_xy[:, 1]
            # keep original z

    # Call underlying IK with projected targets (or original if none projected)
    joints_world, q_sol = robosuite_ee_world_to_joint_positions(
        serial_chain=serial_chain,
        ee_pos_world=proj_targets,
        ee_rot_world=ee_rot_world,
        rot_type=rot_type,
        T_chain_base_in_world=T_world_chain,
        iters=iters,
        lr=lr,
        orientation_weight=orientation_weight,
        init_q=init_q,
        device=device,
        dtype=dtype,
        debug=debug,
    )

    info = {
        'was_projected': was_projected,
        'orig_targets': ee_pos_world_t,
        'proj_targets': proj_targets,
        'horiz_dist': horiz_dist,
    }

    return joints_world, q_sol, info


class PseudoInverseIKWithSVD(PseudoInverseIK):
    # generally slower, but allows for selective damping if needed
    def compute_dq(self, J, dx):
        # reg = self.regularlization * torch.eye(6, device=self.device, dtype=self.dtype)
        U, D, Vh = torch.linalg.svd(J)
        m = D.shape[1]

        # tmpA = U @ (D @ D.transpose(1, 2) + reg) @ U.transpose(1, 2)
        # singular_val = torch.diagonal(D)

        denom = D ** 2 + self.regularlization
        prod = D / denom
        # J^T (JJ^T + lambda^2I)^-1 = V @ (D @ D^T + lambda^2I)^-1 @ U^T = sum_i (d_i / (d_i^2 + lambda^2) v_i @ u_i^T)
        # should be equivalent to damped least squares
        inverted = torch.diag_embed(prod)

        # drop columns from V
        Vh = Vh[:, :m, :]
        total = Vh.transpose(1, 2) @ inverted @ U.transpose(1, 2)

        # dq = J^T (JJ^T + lambda^2I)^-1 dx
        dq = total @ dx
        return dq
