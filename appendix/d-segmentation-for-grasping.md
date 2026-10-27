# Appendix D · Segmentation for grasp points

> A mask tells you which pixels belong to the object; a grasp needs two contact points, an approach direction and a jaw width. The step between them is geometry, and it is where most failures happen.

**Masks.** Segment Anything (SAM) is a promptable segmentation model trained on SA-1B, about 1 billion masks on 11 million images, and it transfers zero-shot to new image distributions ([arXiv 2304.02643](https://arxiv.org/abs/2304.02643)). SAM 2 extends it to video with a streaming memory and reports 6× faster image segmentation than SAM ([arXiv 2408.00714](https://arxiv.org/abs/2408.00714)). A prompt (a point, a box, or a detection from a language-conditioned detector) selects the object; the output is a binary mask.

**From mask to grasp candidates (parallel gripper).**

1. Back-project the masked depth pixels to a point cloud in the robot frame with the camera intrinsics and extrinsics of [lessons 19–20](../part-5-perception-and-estimation/20-pixels-to-table.md). Erode the mask by a few pixels first: edge pixels mix object and background depth.
2. Principal axes of the points (PCA): for a top-down grasp, close the jaws along the minor horizontal axis, approach along −z, centre at the centroid. Reject if the extent along the closing axis exceeds the jaw opening.
3. Antipodal check: the surface normals at the two contacts must lie inside the friction cones, the condition of [lesson 16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md).
4. Reachability: IK within joint limits for the pre-grasp, grasp and lift poses ([lesson 10](../part-3-kinematics/10-inverse-kinematics.md)); collision check of the open jaws against neighbouring masks.

**Learned grasp generators.** Contact-GraspNet predicts 6-DoF parallel-jaw grasps directly from a depth point cloud, trained on 17 million simulated grasps, with over 90% success in structured clutter ([arXiv 2103.14127](https://arxiv.org/abs/2103.14127)). Its code uses an optional segmentation map to crop regions around each object (`--local_regions`) and keep only contacts on the object's surface (`--filter_grasps`) ([repository](https://github.com/NVlabs/contact_graspnet)): the mask selects *which* object, the network proposes *how*. Gemini Robotics-ER lists grasp prediction among its embodied-reasoning outputs ([arXiv 2503.20020](https://arxiv.org/abs/2503.20020)).

**Where it breaks.** Masks that bleed into the table move the centroid and inflate the width; thin, transparent or specular objects give missing depth; a correct mask of the wrong object (a language prompt resolved to a similar object) produces a perfect grasp of the wrong thing, the perception counterpart of the goal substitution measured in [lesson 42](../part-10-foundation-models-and-agents/42-llm-agents.md). Verify the grasp physically (jaw contact and lift), not by the mask score.

**On the SO-101.** The colour-threshold detector of [lesson 21](../part-5-perception-and-estimation/21-object-perception.md) is the mask step for single-colour cubes; a SAM-style model replaces it for arbitrary objects. We have not run SAM or Contact-GraspNet in this book; no numbers here are ours.

Sources: [SAM, arXiv 2304.02643](https://arxiv.org/abs/2304.02643) · [SAM 2, arXiv 2408.00714](https://arxiv.org/abs/2408.00714) · [Contact-GraspNet, arXiv 2103.14127](https://arxiv.org/abs/2103.14127) and [code](https://github.com/NVlabs/contact_graspnet) · [Gemini Robotics, arXiv 2503.20020](https://arxiv.org/abs/2503.20020).
