'''OpenGL extension KHR.texture_compression_astc_hdr

This module customises the behaviour of the 
OpenGL.raw.GLES2.KHR.texture_compression_astc_hdr to provide a more 
Python-friendly API

Overview (from the spec)
	
	Adaptive Scalable Texture Compression (ASTC) is a new texture
	compression technology that offers unprecendented flexibility, while
	producing better or comparable results than existing texture
	compressions at all bit rates. It includes support for 2D and
	slice-based 3D textures, with low and high dynamic range, at bitrates
	from below 1 bit/pixel up to 8 bits/pixel in fine steps.
	
	The goal of these extensions is to support the full 2D profile of the
	ASTC texture compression specification, and allow construction of 3D
	textures from multiple compressed 2D slices.
	
	ASTC-compressed textures are handled in OpenGL ES and OpenGL by adding
	new supported formats to the existing commands for defining and updating
	compressed textures, and defining the interaction of the ASTC formats
	with each texture target.

The official definition of this extension is available here:
http://www.opengl.org/registry/specs/KHR/texture_compression_astc_hdr.txt
'''
from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.KHR.texture_compression_astc_hdr import *
from OpenGL.raw.GLES2.KHR.texture_compression_astc_hdr import _EXTENSION_NAME

def glInitTextureCompressionAstcHdrKHR():
    '''Return boolean indicating whether this extension is available'''
    from OpenGL import extensions
    return extensions.hasGLExtension( _EXTENSION_NAME )


### END AUTOGENERATED SECTION