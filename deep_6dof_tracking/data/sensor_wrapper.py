from enum import Enum

import numpy as np
import cv2

# Kinect dependencies
try:
    from pykinect2.PyKinectV2 import _DepthSpacePoint
    from pykinect2 import PyKinectV2
    from pykinect2 import PyKinectRuntime
    from deep_6dof_tracking.utils.kinect import *
except:
    print('Cannot import kinect libs')
    print('Continuing without Kinect2 support')

# Helios dependencies
try:
    from arena_api.buffer import _Buffer
    from arena_api.system import system
    from deep_6dof_tracking.utils.helios import *
except:
    print('Cannot import arena_api')
    print('Continuing without Helios2 support')


class SensorType(Enum):
    KINECT2 = 1
    HELIOS2 = 2
    TRITON = 3

class SensorWrapper():
    def __init__(self, sensor_type):
        self.sensor_type = sensor_type
        self.binning = False

    def initialize(self, sensor_index=-1, binning=False, ptp=False, serial=None):
        '''
        Initializes the sensor

        Parameters:
            sensor_index (int): While streaming multiple sensors at the same time, this parameter
                is necessary to order the sensors and share the available bandwidth, starting with 0.
                The default value is -1, which means that there is no need to share ressources.
        '''
        # TODO: Throw errors when correct libs couldn't be imported
        self.ptp = ptp
        if self.sensor_type == SensorType.KINECT2:
            self.sensor = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                            PyKinectV2.FrameSourceTypes_Depth)
        elif self.sensor_type == SensorType.HELIOS2:
            self.sensor = self.create_devices_with_tries(serial=serial)[0]
            self.sensor_index = sensor_index
            validate_device(self.sensor)
            self.set_default_settings()
        elif self.sensor_type == SensorType.TRITON:
            self.sensor = self.create_devices_with_tries(serial=serial)[0]
            self.sensor_index = sensor_index
            self.binning = binning
            self.set_default_settings()
        else:
            raise TypeError('Sensor type should be Kinect or Helios2')
        
    def start_stream(self):
        if self.sensor_type in [SensorType.HELIOS2, SensorType.TRITON]:
            self.sensor.start_stream(100)

    def get_last_frame(self, color=None, depth=None, heatmap=None):
        '''
        Returns the last RGBD frame from the sensor. 
        
        Parameters:
            color (bool): Whether to return the color image. Default is True for Kinect2 and False for Helios2.
            depth (bool): Whether to return the depth image. Default is True for Kinect2 and False for Helios2.
            heatmap (bool): Whether to return the depth image as a heatmap. Default is True for Kinect2 and False for Helios2.

        Returns:
            tuple: (color, depth, heatmap) A tuple containing the color, depth and heatmap images if the corresponding parameters are set to True.
                If the parameters are set to False, will return None for the corresponding image.
        '''
        # Set default values based on sensor_type
        if color is None:
            color = self.sensor_type in [SensorType.TRITON, SensorType.KINECT2]
        if depth is None:
            depth = self.sensor_type in [SensorType.HELIOS2, SensorType.KINECT2]
        if heatmap is None:
            heatmap = depth
        if self.sensor_type == SensorType.KINECT2:
            # FIXME
            # This is weird... why do we get depth frame and reshape before calling
            # get_frame_in_color_space? Might have to rework that function
            depth_width, depth_height = self.sensor.depth_frame_desc.Width, self.sensor.depth_frame_desc.Height # Default: 512, 424
            depth_frame = self.sensor.get_last_depth_frame()
            depth_img = depth_frame.reshape(((depth_height, depth_width))).astype(np.uint16)
            color_img, depth_img = get_frame_in_color_space(self.sensor, _DepthSpacePoint , depth_img, return_aligned_image=True, show=False)
            if heatmap:
                heatmap = cv2.convertScaleAbs(depth_img, alpha=255/2500)
                heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
                return color_img, depth_img, heatmap
            # Verify this
            ret = ()
            ret += (color_img,) if color else (None,)
            ret += (depth_img,) if depth else (None,)
            ret += (heatmap,) if heatmap else (None,)
            return ret
        
        elif self.sensor_type == SensorType.HELIOS2:
            buffer_3d : _Buffer = self.sensor.get_buffer()
            depth_img = get_depth_image_from_buffer2(buffer_3d, self.scale_z)
            self.sensor.requeue_buffer(buffer_3d)
            if heatmap:
                heatmap = cv2.convertScaleAbs(depth_img, alpha=255/2500)
                heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
                return None, depth_img, heatmap
            return None, depth_img, None
        
        elif self.sensor_type == SensorType.TRITON:
            buffer_rgb : _Buffer = self.sensor.get_buffer()
            rgb_img = get_rgb_image_from_buffer(buffer_rgb, binning=True)
            self.sensor.requeue_buffer(buffer_rgb)
            return rgb_img, None, None
        
    def get_last_buffer(self):
        if self.sensor_type in [SensorType.HELIOS2, SensorType.TRITON]:
            buffer : _Buffer = self.sensor.get_buffer()
            return buffer
        else:
            raise TypeError('get_last_buffer is only available for Helios2 and Triton sensors')

    def close(self):
        if self.sensor_type == SensorType.KINECT2:
            self.sensor.close()
        else : 
            self.sensor.stop_stream()
            # restores initial node values
            # for param, value in self.initial_params.items():
            #     if value is not None:
            #         print(f'Restoring {param} to {value}')
            #         try:
            #             self.nodemap[param].value = value
            #         except Exception as e:
            #             print(f'Error restoring {param} to {value}: {e}')
            system.destroy_device(self.sensor)

    def create_devices_with_tries(self, serial=None):

        '''
        This function waits for the user to connect a device before raising
            an exception
        '''
        # tries variables
        tries = 0
        tries_max = 1
        sleep_time_secs = 10

        # only get devices for specified sensor type
        device_infos = system.device_infos
        print(device_infos)
        if self.sensor_type == SensorType.HELIOS2:
            device_infos = [d for d in device_infos if 'HTP' in d['model']]
        elif self.sensor_type == SensorType.TRITON:
            device_infos = [d for d in device_infos if 'TRI' in d['model']]
        if serial is not None:
            device_infos = [d for d in device_infos if serial in d['serial']]

        # Try creating devices
        while tries < tries_max:  # Wait for device for 60 seconds
            devices = system.create_device(device_infos)
            if not devices:
                print(
                    f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                    f'secs for a device to be connected!')
                for sec_count in range(sleep_time_secs):
                    time.sleep(1)
                    print(f'{sec_count + 1 } seconds passed ',
                        '.' * sec_count, end='\r')
                tries += 1
            else:
                print(f'Created {len(devices)} device(s)')
                for device in devices:
                    thread_id = f'''{device.nodemap['DeviceModelName'].value}''' \
                        f'''-{device.nodemap['DeviceSerialNumber'].value} |'''
                    print(f'{thread_id} Device created')
                
                return devices
        else:
            raise Exception(f'No device found! Please connect a device and run '
                            f'the example again.')

    def set_default_settings(self):
        # # Set settings for device's stream nodemap
        # tl_stream_nodemap = self.sensor.tl_stream_nodemap
        # tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        # tl_stream_nodemap['StreamPacketResendEnable'].value = True
        # tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"

        # # Set device settings
        # self.nodemap = self.sensor.nodemap
        # default_nodes = self.nodemap.get_node(['Width',
        #                                'Height',
        #                                'PixelFormat',
        #                                'DeviceStreamChannelPacketSize',
        #                                'GevSCPD',
        #                                'GevSCFTD',
        #                                'AcquisitionMode'])
        # self.initial_params = {
        #     'PixelFormat': default_nodes['PixelFormat'].value,
        #     'DeviceStreamChannelPacketSize': default_nodes['DeviceStreamChannelPacketSize'].value,
        #     'GevSCPD': default_nodes['GevSCPD'].value,
        #     'GevSCFTD': default_nodes['GevSCFTD'].value,
        #     'AcquisitionMode': default_nodes['AcquisitionMode'].value
        # }

        # # Set nodes' values
        # default_nodes['PixelFormat'].value = 'Coord3D_C16' if self.sensor_type == SensorType.HELIOS2 else 'BayerRG8'
        # default_nodes['DeviceStreamChannelPacketSize'].value = default_nodes['DeviceStreamChannelPacketSize'].max
        # default_nodes['AcquisitionMode'].value = 'Continuous'

        # if self.sensor_index != -1:
        #     val = 67000
        #     default_nodes['GevSCPD'].value = val
        #     default_nodes['GevSCFTD'].value = val * self.sensor_index

        #     self.initial_params['PtpEnable'] = self.nodemap['PtpEnable'].value
        #     self.nodemap['PtpEnable'].value = self.ptp
        #     self.initial_params['PtpSlaveOnly'] = self.nodemap['PtpSlaveOnly'].value
        #     self.nodemap['PtpSlaveOnly'].value = True if self.sensor_type == SensorType.TRITON else False

        # if self.sensor_type == SensorType.TRITON:
        #     self.initial_params['Width'] = default_nodes['Width'].value
        #     self.initial_params['Height'] = default_nodes['Height'].value

        #     default_nodes['Width'].value = default_nodes['Width'].max
        #     default_nodes['Height'].value = default_nodes['Height'].max

        #     self.initial_params['AcquisitionFrameRateEnable'] = self.nodemap['AcquisitionFrameRateEnable'].value
        #     self.initial_params['AcquisitionFrameRate'] = self.nodemap['AcquisitionFrameRate'].value
        #     # self.nodemap['AcquisitionFrameRate'].value = self.nodemap['AcquisitionFrameRate'].max
        #     # self.nodemap['AcquisitionFrameRateEnable'].value = True
        #     # time.sleep(3)
        #     # self.nodemap['AcquisitionFrameRate'].value = 20.0
            
        #     # idk if manual exposure/framerate will work
        #     self.initial_params['ExposureTime'] = self.nodemap['ExposureTime'].value
        #     self.initial_params['ExposureAuto'] = self.nodemap['ExposureAuto'].value

        #     # self.nodemap['ExposureAuto'].value = 'Off'
        #     # self.nodemap['ExposureTime'].value = 40000.0

        # if self.sensor_type == SensorType.HELIOS2:
        #     self.initial_params['Scan3dOperatingMode'] = self.nodemap['Scan3dOperatingMode'].value
        #     self.initial_params['Scan3dCoordinateSelector'] = self.nodemap['Scan3dCoordinateSelector'].value

        #     self.nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
        #     self.nodemap['Scan3dOperatingMode'].value = 'Distance8300mmMultiFreq'
        #     # Get Z coordinate scale in order to convert z values to mm
        #     self.scale_z = self.nodemap["Scan3dCoordinateScale"].value

        # if self.binning:
        #     self.initial_params['BinningSelector'] = self.nodemap['BinningSelector'].value
        #     self.initial_params['BinningHorizontalMode'] = self.nodemap['BinningHorizontalMode'].value
        #     self.initial_params['BinningVerticalMode'] = self.nodemap['BinningVerticalMode'].value
        #     self.initial_params['BinningHorizontal'] = self.nodemap['BinningHorizontal'].value
        #     self.initial_params['BinningVertical'] = self.nodemap['BinningVertical'].value

        #     default_nodes['PixelFormat'].value = 'BayerRG8'
        #     self.nodemap['BinningSelector'].value = 'Digital'
        #     self.nodemap['BinningHorizontal'].value = 2
        #     self.nodemap['BinningVertical'].value = 2
        #     self.nodemap['BinningHorizontalMode'].value = 'Sum'
        #     self.nodemap['BinningVerticalMode'].value = 'Sum'

        tl_stream_nodemap = self.sensor.tl_stream_nodemap
        self.nodemap = self.sensor.nodemap

        import yaml

        # Function to convert a list of parameters into a dictionary
        def list_to_dict(param_list):
            param_dict = {}
            for param in param_list:
                key, value = param.split('=')
                param_dict[key] = value
            return param_dict

        # Read the YAML file
        with open('C:/Users/renau/Documents/Uni/maitrise/source/deep_6dof_tracking/deep_6dof_tracking/data/sensor_wrapper_test.yml', 'r') as file:
            config = yaml.safe_load(file)

        # Convert lists of parameters into dictionaries
        dict1 = list_to_dict(config['nodemap'])
        dict2 = list_to_dict(config['tl_stream_nodemap'])

        # Print the dictionaries
        print("nodemap:")
        print(dict1)
        print("\ntl_stream_nodemap:")
        print(dict2)

        def is_numeric(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        def is_integer(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        for param, value in dict2.items():
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            tl_stream_nodemap[param].value = value

        for param, value in dict1.items():
            if is_integer(value):
                value = int(value)
            elif is_numeric(value):
                value = float(value)
            elif value == 'True':
                value = True
            elif value == 'False':
                value = False
            print(f"Setting {param} to {value}")
            self.nodemap[param].value = value

        self.scale_z = 0.25



    def setup_ptp(self):
        self.initial_params['AcquisitionStartMode'] = self.nodemap['AcquisitionStartMode'].value
        self.nodemap['AcquisitionStartMode'].value = 'PTPSync'
        print(self.nodemap['AcquisitionFrameRate'].max)
        self.nodemap['AcquisitionFrameRate'].value = self.nodemap['AcquisitionFrameRate'].max
        self.nodemap['PTPSyncFrameRate'].value = 10.0




