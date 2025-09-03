import time
import sys
import ctypes

import numpy as np
import cv2

from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.buffer import _Buffer


def validate_device(device):

	# validate if Scan3dCoordinateSelector node exists.
	# If not, it is (probably) not a Helios Camera running the example
	try:
		scan_3d_operating_mode_node = device. \
			nodemap['Scan3dOperatingMode'].value
	except (KeyError):
		print('Scan3dCoordinateSelector node is not found. \
			Please make sure that Helios device is used for the example.\n')
		sys.exit()

	# validate if Scan3dCoordinateOffset node exists.
	# If not, it is (probably) that Helios Camera has an old firmware
	try:
		scan_3d_coordinate_offset_node = device. \
			nodemap['Scan3dCoordinateOffset'].value
	except (KeyError):
		print('Scan3dCoordinateOffset node is not found. \
			Please update Helios firmware.\n')
		sys.exit()

	# check if Helios2 camera used for the example
	device_model_name_node = device.nodemap['DeviceModelName'].value
	if 'HLT' or 'HTP' in device_model_name_node:
		global isHelios2
		isHelios2 = True
		
def get_depth_image_from_buffer(buffer3d: _Buffer, scale_z):
	pdata_16bit = ctypes.cast(buffer3d.pdata, ctypes.POINTER(ctypes.c_int16))

	number_of_pixels = buffer3d.width * buffer3d.height

    # Data is of the form XYZI (where I is intensity) we only care about Z
	z = pdata_16bit[2:2+number_of_pixels*4:4]
	depth_array = np.ctypeslib.as_array(z, shape=(buffer3d.height, buffer3d.width))
	depth_array = depth_array * scale_z
	depth_array = depth_array.reshape((buffer3d.height, buffer3d.width))
	return depth_array

def get_depth_image_from_buffer2(buffer3d, scale_z):
	pdata_16bit = ctypes.cast(buffer3d.pdata, ctypes.POINTER(ctypes.c_int16))

	#number_of_pixels = buffer3d.width * buffer3d.height

    # Data is of the form XYZI (where I is intensity) we only care about Z
	#z = pdata_16bit[2:2+number_of_pixels*4:4]
	depth_array = np.ctypeslib.as_array(pdata_16bit, shape=(buffer3d.height, buffer3d.width))
	depth_array = depth_array * scale_z
	depth_array = depth_array.reshape((buffer3d.height, buffer3d.width))
	return depth_array

def get_images_from_buffer(buffer3d, scale_z):
	pdata_16bit = ctypes.cast(buffer3d.pdata, ctypes.POINTER(ctypes.c_int16))

	number_of_pixels = buffer3d.width * buffer3d.height

    # Data is of the form XYZI (where I is intensity) we only care about Z
	z = pdata_16bit[2:2+number_of_pixels*4:4]
	depth_array = np.ctypeslib.as_array(z, shape=(buffer3d.height, buffer3d.width))
	depth_array = depth_array * scale_z
	depth_array = depth_array.reshape((buffer3d.height, buffer3d.width))

	# Get intensity also
	intensity = pdata_16bit[3:3+number_of_pixels*4:4]
	intensity = np.ctypeslib.as_array(intensity, shape=(buffer3d.height, buffer3d.width))
	intensity = intensity.reshape((buffer3d.height, buffer3d.width))
	return depth_array, intensity

def get_rgb_image_from_buffer(buffer, binning=False):
	item = BufferFactory.copy(buffer)
	if binning:
		item = BufferFactory.convert(item, 'RGB8')
	
	buffer_bytes_per_pixel = int(len(item.data)/(item.width * item.height))
	"""
	Buffer data as cpointers can be accessed using buffer.pbytes
	"""
	array = (ctypes.c_ubyte * 3 * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
	"""
	Create a reshaped NumPy array to display using OpenCV
	"""
	npndarray = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width, buffer_bytes_per_pixel))
	# BufferFactory.destroy(item)
	return npndarray