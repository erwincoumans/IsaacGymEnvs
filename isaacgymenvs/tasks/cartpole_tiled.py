
#for pytinyopengl3: pip install pytinydiffsim, use latest, at least version >= 0.5.0
import pytinyopengl3 as g
import math
import torch

from numpngw import write_apng

import numpy as np
use_cv2 = False
if use_cv2:
  import cv2



use_image_observation = True

class CartpoleTiled:
    def __init__(self, num_envs=256, camera_width = 64, camera_height  = 64, enable_tiled = True, sim_spacing = 0. , use_matplot_lib = False, use_cuda_interop = True, use_headless = True):
            if use_matplot_lib:
              import matplotlib.pyplot as plt
              self.plt = plt
              plt.ion()
              img = np.random.rand(2000, 2000)
              self.image = plt.imshow(img, interpolation='none')#, cmap='gray', vmin=0.8, vmax=1)
              self.ax = plt.gca()


            self.use_cuda_interop = use_cuda_interop
            
            self.enable_tiled = enable_tiled
            self.sim_spacing = sim_spacing
            
            self.use_matplot_lib = use_matplot_lib
            self.num_envs = num_envs
      
            self.max_x = math.ceil(math.sqrt(num_envs))
            self.tile_width = camera_width
            self.tile_height = camera_height
            self.width=self.tile_width * self.max_x
            self.height=self.tile_height * self.max_x
            
            
            num_actors=num_envs

            if use_headless:
               window_type = 2
            else:
               window_type = 0
            
            self.viz = g.OpenGLUrdfVisualizer(width=self.width, height=self.height, window_type=window_type)
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
          
            if self.enable_tiled:
              
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

    def sync_transforms_cpu(self, rb_transforms):

        skip = 0
          
          
        #print("len(self.all_instances)=",len(self.all_instances))
        self.viz.sync_visual_transforms(self.all_instances, rb_transforms, skip, self.sim_spacing,apply_visual_offset=True)

    def update_observations(self, write_transforms, camera_positions=None):
                
          if self.use_cuda_interop:
            self.viz.opengl_app.enable_render_to_texture(self.width, self.height)
          
          if self.enable_tiled:
              
              cam = self.viz.opengl_app.renderer.get_active_camera()
              tile_index = 0
              x=0
              y=0
              #self.max_x
              for tile_index in range (self.num_envs):
                  tile = self.tiles[tile_index]
                  #tile_index+=1
                  tile.view_matrix = cam.get_camera_view_matrix()
                  tile.viewport_dims=[x*self.tile_width,y*self.tile_height,self.tile_width, self.tile_height]
                  x+=1
                  if x>=self.max_x:
                    x=0
                    y+=1
              self.viz.render_tiled(self.tiles, do_swap_buffer = False, render_segmentation_mask=False)
          else:
            self.viz.render(do_swap_buffer=False, render_segmentation_mask=False)
          
          
          if self.use_cuda_interop:
            self.viz.opengl_app.cuda_copy_texture_image(self.cuda_tex, self.cuda_mem, self.cuda_num_bytes)
            #print("self.ttensor.shape=",self.ttensor.shape)
            #print("self.ttensor=",self.ttensor)
          else:
            pixels = g.ReadPixelBuffer(self.viz.opengl_app)
            self.rgba_f32 =  pixels.rgba.astype(np.float32)

          
          if self.use_matplot_lib:
            if self.use_cuda_interop:
              ftensor = self.ttensor.type(torch.float32)
              np_img_arr = ftensor.cpu().numpy()
              np_img_arr = np.reshape(np_img_arr, (self.height, self.width, 4))
              np_img_arr = np.flipud(np_img_arr)
            else:
              np_img_arr = pixels.rgba
              np_img_arr = np.reshape(np_img_arr, (self.height, self.width, 4))
              np_img_arr = np_img_arr * (1. / 255.)
              np_img_arr = np.flipud(np_img_arr)
          
            self.image.set_data(np_img_arr)
            self.ax.plot([0])
            self.plt.show()
            self.plt.pause(0.0001)
  
          self.viz.swap_buffer()
        

          #print("self.dof_pos[env_ids, 0].shape=",self.dof_pos[env_ids, 0].shape)
          #sq = self.dof_pos[env_ids, 0].squeeze()
          
          #print("sq.shape=",sq.shape)
          #this copy/reshaping is sub-optimal, need a person with some pytorch-fu
          
          if self.use_cuda_interop:
            ftensor = self.ttensor.type(torch.float32)*255.
          else:
            ftensor = torch.from_numpy(self.rgba_f32)
          self.ftensor2 = ftensor

          ftensor  = torch.reshape(ftensor, (self.height, self.width, 4))
          #ftensor = torch.flipud(ftensor)
          ftensor = ftensor.reshape(self.max_x, self.tile_width, self.max_x, self.tile_height, 4)
          ftensor = ftensor.swapaxes(1,2)
          ftensor = ftensor.reshape(self.max_x*self.max_x, self.tile_width*self.tile_height*4)
          ftensor = ftensor[:self.num_envs,]
            
          ftensor = ftensor.reshape(self.max_x*self.max_x, self.tile_width,self.tile_height,4)
          self.ftensor = ftensor

          #print("ftensor.shape=", ftensor.shape)
          #self.obs_buf = ftensor
        
          #self.gym.render_all_camera_sensors(self.sim)
          #self.gym.start_access_image_tensors(self.sim)
          #stk = self.camera_image_stack
          #print("stk=", stk)#this copy/reshaping is sub-optimal, need a person with some pytorch-fu
          
          if self.use_cuda_interop:
            ftensor = self.ttensor.type(torch.float32)*255.
          else:
            ftensor = torch.from_numpy(self.rgba_f32)
          self.ftensor_opengl = ftensor

          ftensor  = torch.reshape(ftensor, (self.height, self.width, 4))
          #ftensor = torch.flipud(ftensor)
          ftensor = ftensor.reshape(self.max_x, self.tile_width, self.max_x, self.tile_height, 4)
          ftensor = ftensor.swapaxes(1,2)
          ftensor = ftensor.reshape(self.max_x*self.max_x, self.tile_width*self.tile_height*4)
          ftensor = ftensor[:self.num_envs,]
            
          ftensor = ftensor.reshape(self.max_x*self.max_x, self.tile_width,self.tile_height,4)
          
          #tensor layout for PyTorch/RL, with image for each environment in contiguos layout
          self.ftensor = ftensor

          #print("ftensor.shape=", ftensor.shape)
          #self.obs_buf = ftensor
        
          #self.gym.render_all_camera_sensors(self.sim)
          #self.gym.start_access_image_tensors(self.sim)
          #stk = self.camera_image_stack
          #print("stk=", stk)
          #if stk > 1:
          #    # move the previous (stack-1) frames 1 step forward in the buffer
          #    self.obs_buf[:, :, :, (1) * self.camera_channels: (stk) * self.camera_channels]
          #    self.obs_buf[:, :, :, (0) * self.camera_channels: (stk-1) * self.camera_channels]
                
          #print("self.obs_buf.shape=",self.obs_buf.shape)
          #print("ftensor.shape=", ftensor.shape)
          #print("self.obs_buf.shape=", self.obs_buf.shape)
          #if stk > 1:
          #    # move the previous (stack-1) frames 1 step forward in the buffer
          #    self.obs_buf[:, :, :, (1) * self.camera_channels: (stk) * self.camera_channels]
          #    self.obs_buf[:, :, :, (0) * self.camera_channels: (stk-1) * self.camera_channels]
                
          #print("self.obs_buf.shape=",self.obs_buf.shape)
          #print("ftensor.shape=", ftensor.shape)
          #print("self.obs_buf.shape=", self.obs_buf.shape)
                    
frames = []                    
if __name__ == '__main__':

  use_tiled = True

  if use_tiled:
    cam = CartpoleTiled(num_envs = 400, camera_width = 64, camera_height = 64, enable_tiled = True, sim_spacing = 0., use_matplot_lib = False, use_headless = True)
    num_frames = 10
  else:
    cam = CartpoleTiled(num_envs = 400, camera_width = 64, camera_height = 64, enable_tiled = False, sim_spacing = 2., use_matplot_lib = False, use_headless = False)
    num_frames = 1000

  npz_data = np.load("cartpole_trans.npz")
  all_rb = npz_data['a']
  print("all_rb.shape=",all_rb.shape)
  total_frames = all_rb.shape[0]
  frame = 0
  for f in range (num_frames):
    rb = all_rb[frame].copy()
    frame += 1
    #print("frame=",frame)
    if frame == (total_frames-1):
        frame = 0
    rb.resize(cam.num_envs,21)
    #print("!!!!rb.shape=", rb.shape)
    cam.sync_transforms_cpu(rb)
    #for GPU sync of transforms, you could use Warp/CUDA
    #see for example 
    #https://github.com/erwincoumans/OmniIsaacGymEnvs/blob/main/omniisaacgymenvs/tasks/cartpole_tiled_camera.py#L49

    cam.update_observations(write_transforms=True)

    if use_tiled:    
      np_img_arr = cam.ftensor2.cpu().numpy()
      np_img_arr = np.reshape(np_img_arr, (cam.height, cam.width, 4))
      np_img_arr = np.flipud(np_img_arr)
      frame_img = np_img_arr[:,:,:3]
      frame_img = np.array(frame_img , dtype=np.uint8)
      frames.append(frame_img)
    else:
       import time
       time.sleep(1./60.)

  if use_tiled:
    print("\nPlease wait while writing animated png file.")
    write_apng("tiled_image.png", frames, delay=100)
