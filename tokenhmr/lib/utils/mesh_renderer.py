import os

# 默认使用 EGL 做 headless 渲染.
# 如果外部已设置,则尊重外部配置,便于在不同机器上切换平台(osmesa/egl).
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import torch.nn.functional as F

from .render_openpose import render_openpose

def create_offscreen_renderer(viewport_width: int, viewport_height: int, point_size: float = 1.0):
    """
    创建 pyrender.OffscreenRenderer.

    背景:
    - headless 环境下通常使用 EGL.
    - 在部分机器上,EGL 的 device 0 初始化会失败(eglInitialize 返回错误).
    - pyrender 支持通过环境变量 `EGL_DEVICE_ID` 切换 EGL device.

    策略:
    - 先按当前环境变量创建.
    - 若失败且当前为 EGL 且 `EGL_DEVICE_ID` 还是默认的 0,则回退到 1 重试一次.
    """
    try:
        return pyrender.OffscreenRenderer(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            point_size=point_size,
        )
    except Exception:
        if os.environ.get('PYOPENGL_PLATFORM') == 'egl' and os.environ.get('EGL_DEVICE_ID', '0') == '0':
            os.environ['EGL_DEVICE_ID'] = '1'
            return pyrender.OffscreenRenderer(
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                point_size=point_size,
            )
        raise

def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class MeshRenderer:

    def __init__(self, cfg, faces=None):
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE
        self.renderer = create_offscreen_renderer(
            viewport_width=self.img_res,
            viewport_height=self.img_res,
            point_size=1.0,
        )
        
        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces

    def visualize(self, vertices, camera_translation, images, focal_length=None, nrow=3, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def visualize_tensorboard(self, vertices, camera_translation, images, pred_keypoints, gt_keypoints, focal_length=None, nrow=5, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        nrow = nrow-1 if gt_keypoints is None else nrow
        nrow = nrow-1 if pred_keypoints is None else nrow
        if pred_keypoints is not None:
            pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
            pred_keypoints = self.img_res * (pred_keypoints + 0.5)
        if gt_keypoints is not None:
            gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)
        keypoint_matches = [(1, 12), (2, 8), (3, 7), (4, 6), (5, 9), (6, 10), (7, 11), (8, 14), (9, 2), (10, 1), (11, 0), (12, 3), (13, 4), (14, 5)]
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()

            if pred_keypoints is not None:
                body_keypoints = pred_keypoints[i, :25]
                extra_keypoints = pred_keypoints[i, -19:]
                for pair in keypoint_matches:
                    body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
                pred_keypoints_img = render_openpose(255 * images_np[i].copy(), body_keypoints) / 255
            if gt_keypoints is not None:
                body_keypoints = gt_keypoints[i, :25]
                extra_keypoints = gt_keypoints[i, -19:]
                for pair in keypoint_matches:
                    if extra_keypoints[pair[1], -1] > 0 and body_keypoints[pair[0], -1] == 0:
                        body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
                gt_keypoints_img = render_openpose(255*images_np[i].copy(), body_keypoints) / 255
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            if pred_keypoints is not None:
                rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2,0,1))
            if gt_keypoints is not None:
                rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2,0,1))
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, focal_length=5000, text=None, resize=None, side_view=False, baseColorFactor=(1.0, 1.0, 0.9, 1.0), rot_angle=90):
        renderer = create_offscreen_renderer(
            viewport_width=image.shape[1],
            viewport_height=image.shape[0],
            point_size=1.0,
        )
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=baseColorFactor)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0.8)[:, :, np.newaxis]
        bg_image = image if not side_view else np.ones_like(image)
        output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * bg_image)
        if resize is not None:
            output_img = cv2.resize(output_img, resize)

        output_img = output_img.astype(np.float32)
        renderer.delete()
        return output_img
