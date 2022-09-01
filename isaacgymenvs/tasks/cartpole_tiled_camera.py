# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
import math

use_cv2 = False
if use_cv2:
  import cv2

#for pytinyopengl3: pip install pytinydiffsim, use latest, at least version >= 0.5.0
import pytinyopengl3 as g

use_cuda_interop = True

use_matplot_lib = False
if use_matplot_lib:
  import matplotlib.pyplot as plt
  plt.ion()
  img = np.random.rand(2000, 2000)
  image = plt.imshow(img, interpolation='none')#, cmap='gray', vmin=0.8, vmax=1)
  ax = plt.gca()

use_image_observation = True


class CartpoleTiledCamera(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = self.cfg["env"].get("episodeLength", 500)

        self.use_camera = self.cfg["env"].get("useCamera", False)
        self.camera_type = self.cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self.cfg["env"].get("cameraWidth", -1)
        self.camera_height = self.cfg["env"].get("cameraHeight", -1)
        self.camera_image_stack = self.cfg["env"].get("cameraImageStack", 1)

        if self.camera_type == "rgb":
            self.camera_channels = 3
        elif self.camera_type == "grey":
            self.camera_channels = 1
        elif self.camera_type == "depth":
            self.camera_channels = 1
        elif self.camera_type == "rgbd":
            self.camera_channels = 4
        else:
            raise NotImplementedError(f"Unsupported camera type {self.camera_type}")

        self.camera_channels + self.camera_image_stack*self.camera_channels
        self.cfg["env"]["numObservations"] = 0#self.camera_height, self.camera_width, num_stacked_channels
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.camera_handles = []
        self.cartpole_handles = []
        self.camera_tensors = []
        
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)
        if self.use_camera:
            self.enabled_tiled = True
      
            self.max_x = math.ceil(math.sqrt(num_envs))
            self.tile_width = self.camera_width
            self.tile_height = self.camera_height
            self.width=self.tile_width * self.max_x
            self.height=self.tile_height * self.max_x
            
            
            num_actors=num_envs
            
            self.viz = g.OpenGLUrdfVisualizer(width=self.width, height=self.height, window_type=2)
            self.viz.opengl_app.set_background_color(1.,1.,1.)
            self.viz.opengl_app.swap_buffer()
            self.viz.opengl_app.swap_buffer()
  
            if use_cuda_interop:
              self.render_texid = self.viz.opengl_app.enable_render_to_texture(self.width, self.height)
              self.viz.opengl_app.swap_buffer()
              self.viz.opengl_app.swap_buffer()
              self.cuda_tex = self.viz.opengl_app.cuda_register_texture_image(self.render_texid, True)
              self.cuda_num_bytes = self.width*self.height*4*2 #4 component half-float, each half-float 2 bytes
              print("cuda_num_bytes=", self.cuda_num_bytes)
              self.ttensor = torch.zeros(self.width*self.height*4, dtype=torch.float16, device="cuda")
              self.cuda_mem = self.ttensor.data_ptr()
              
  
            urdf = g.OpenGLUrdfStructures()
            parser = g.UrdfParser()
            file_name = "../assets/urdf/cartpole.urdf"
            urdf = parser.load_urdf(file_name)
            print("urdf=",urdf)
            texture_path = "laikago_tex.jpg"
            self.viz.path_prefix = g.extract_path(file_name)
            print("viz.path_prefix=",self.viz.path_prefix)
            self.viz.convert_visuals(urdf, texture_path)
            print("create_instances")
  
            self.all_instances = self.viz.create_instances(urdf, texture_path, num_actors)
            #print("self.all_instances=",self.all_instances)
            verbose_print = False
            if verbose_print:
              print("len(self.all_instances)=",len(self.all_instances))
              for pairs in self.all_instances:
                print("len(pairs)=", len(pairs))
                for pair in pairs:
                 print("pair.visual_instance=",pair.visual_instance)
          
            if self.enabled_tiled:
              
              print("tile_width=", self.tile_width)
              print("tile_height=", self.tile_height)
              print("num_envs=", num_envs)
              self.tiles=[]
              x=0
              y=0
              for t in range (num_envs):
                  tile = g.TinyViewportTile()
                  pairs = self.all_instances[t]
                  viz_instances = []
                  for pair in pairs:
                    viz_instances.append(pair.visual_instance)
                  #print("viz_instances=",viz_instances)
                  tile.visual_instances = viz_instances#[t, 512+t, 1024+t]
                  #print("tile.visual_instances=",tile.visual_instances)
                  cam = self.viz.opengl_app.renderer.get_active_camera()
                  tile.projection_matrix = cam.get_camera_projection_matrix()
                  tile.view_matrix = cam.get_camera_view_matrix()
                  tile.viewport_dims=[x*self.tile_width,y*self.tile_height,self.tile_width, self.tile_height]
                  self.tiles.append(tile)
                  x+=1
                  if x>=self.max_x:
                    x=0
                    y+=1
  
            cam = g.TinyCamera()
            cam.set_camera_up_axis(2)
            cam.set_camera_distance(3)
            cam.set_camera_pitch(0)#-30)
            cam.set_camera_yaw(90)#-30)
            cam.set_camera_target_position(0.,0.,2.)
            self.viz.opengl_app.renderer.set_camera(cam)
    

    def compute_reward(self):
        # retrieve environment observations from buffer
        cart_pos = self.dof_pos[:, 0].squeeze()
        cart_vel = self.dof_vel[:, 0].squeeze()
        pole_angle = self.dof_pos[:, 1].squeeze()
        pole_vel = self.dof_vel[:, 1].squeeze()

        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        if self.device != 'cpu':
          self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.use_camera:
          import time
          start_time = time.time()
                    
          rbs = self.rb_states.cpu().numpy()
          rb = rbs.copy()
          rb = rb[:,:7]
          rb = rb.reshape(int(rb.shape[0]/3), int(7*3))
          skip = 0
          if self.enabled_tiled:
            sim_spacing = 0
          else:
            sim_spacing = 10
          self.viz.sync_visual_transforms(self.all_instances, rb, skip, sim_spacing,apply_visual_offset=True)
          
          if use_cuda_interop:
            self.viz.opengl_app.enable_render_to_texture(self.width, self.height)
          
          if self.enabled_tiled:
              
              cam = self.viz.opengl_app.renderer.get_active_camera()
              tile_index = 0
              x=0
              y=0
              #self.max_x
              for tile_index in range (self.num_envs):
                  tile = self.tiles[tile_index]
                  tile_index+=1
                  tile.view_matrix = cam.get_camera_view_matrix()
                  tile.viewport_dims=[x*self.tile_width,y*self.tile_height,self.tile_width, self.tile_height]
                  x+=1
                  if x>=self.max_x:
                    x=0
                    y+=1
              self.viz.render_tiled(self.tiles, do_swap_buffer = False, render_segmentation_mask=False)
          else:
            self.viz.render(do_swap_buffer=False, render_segmentation_mask=False)
          
          ct = time.time()
          if use_cuda_interop:
            self.viz.opengl_app.cuda_copy_texture_image(self.cuda_tex, self.cuda_mem, self.cuda_num_bytes)
            #print("self.ttensor.shape=",self.ttensor.shape)
            #print("self.ttensor=",self.ttensor)
          else:
            pixels = g.ReadPixelBuffer(self.viz.opengl_app)
          et = time.time()
          #print("cuda_copy_texture_image dt=", et-ct)
          
          
          end_time = time.time()
          #print("duration =", end_time-start_time)
          #print("fps =", float(self.num_envs)/(end_time-start_time))
          
          
          if use_matplot_lib:
            if use_cuda_interop:
              ftensor = self.ttensor.type(torch.float32)
              np_img_arr = ftensor.cpu().numpy()
              np_img_arr = np.reshape(np_img_arr, (self.height, self.width, 4))
              np_img_arr = np.flipud(np_img_arr)
            else:
              np_img_arr = pixels.rgba
              np_img_arr = np.reshape(np_img_arr, (self.height, self.width, 4))
              np_img_arr = np_img_arr * (1. / 255.)
              np_img_arr = np.flipud(np_img_arr)
          
            image.set_data(np_img_arr)
            ax.plot([0])
            plt.show()
            plt.pause(0.0001)
  
          self.viz.swap_buffer()
        

          #print("self.dof_pos[env_ids, 0].shape=",self.dof_pos[env_ids, 0].shape)
          sq = self.dof_pos[env_ids, 0].squeeze()
          
          #print("sq.shape=",sq.shape)
          if use_image_observation:
            #this copy/reshaping is sub-optimal, need a person with some pytorch-fu
            ftensor = self.ttensor.type(torch.float32)*255.
            ftensor  = torch.reshape(ftensor, (self.height, self.width, 4))
            #ftensor = torch.flipud(ftensor)
            ftensor = ftensor.reshape(self.max_x, self.tile_width, self.max_x, self.tile_height, 4)
            ftensor = ftensor.swapaxes(1,2)
            ftensor = ftensor.reshape(self.max_x*self.max_x, self.tile_width*self.tile_height*4)
            ftensor = ftensor[:self.num_envs,]
              
            ftensor = ftensor.reshape(self.max_x*self.max_x, self.tile_width,self.tile_height,4)
              
            #print("ftensor.shape=", ftensor.shape)
            #self.obs_buf = ftensor
          
            #self.gym.render_all_camera_sensors(self.sim)
            #self.gym.start_access_image_tensors(self.sim)
            stk = self.camera_image_stack
            #print("stk=", stk)
            if stk > 1:
                # move the previous (stack-1) frames 1 step forward in the buffer
                self.obs_buf[:, :, :, (1) * self.camera_channels: (stk) * self.camera_channels]
                self.obs_buf[:, :, :, (0) * self.camera_channels: (stk-1) * self.camera_channels]
                  
            #print("self.obs_buf.shape=",self.obs_buf.shape)
            #print("ftensor.shape=", ftensor.shape)
            #print("self.obs_buf.shape=", self.obs_buf.shape)
            
            self.obs_buf[:, :, :, 0:self.camera_channels] = ftensor
              
            for id in np.arange(self.num_envs):
                #    camera_gpu_tensor = self.camera_tensors[id][:, :, 0:self.camera_channels].clone()
                #    #if (id==0):
                #    #  print("camera_gpu_tensor=",camera_gpu_tensor.float())
                #    #print(camera_gpu_tensor.shape)
                #    self.obs_buf[id, :, :, 0:self.camera_channels] = camera_gpu_tensor.float()
                if use_cv2:
                   if id == 2:
                     #cv2.imshow("image", self.obs_buf[id, :, :, 0:3].cpu().numpy() / 255.)
                     cv2.imshow("image", self.obs_buf[id, :, :, 0:3].cpu().numpy())
                     cv2.waitKey(1000)

        else:
          self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
          self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
          self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
          self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        if self.use_camera: 
            #self.obs_buf["camera"][env_ids] = 0.
            #self.camera_obs_buf[env_ids] = 0.0
            #print(self.obs_dict)
            self.obs_buf[env_ids, :, :, :] = 0.0
              
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
