import vapoursynth as vs
import mvsfunc as mvf
import numpy as np
import functools

# PALLADIAWA MOBIACIKAKA's FUNCTION
MODULE_NAME = 'pmsfunc'
  
core = vs.core


####################################
###		 successive filtering	 ###
####################################
def scsFilters(clip, filterings, prerequisites=None):
	funcname			= 'scsFilters'

	if not isinstance(filterings, list):
		raise TypeError(funcname + ': \"filterings\" must be a list!')

	if prerequisites is None:
		pass
	elif not isinstance(prerequisites, str):
		raise TypeError(funcname + ': \"prerequisites\" must be a string!')
	else:
		exec(prerequisites)

	for func, param in filterings:
		clip = eval(func)(clip, **param)

	return clip
####################################


####################################
###		 Rectangle   Filter	     ###
####################################
### It's recommended copying	 ###
### this function into your		 ###
### scripts.					 ###
####################################
''' Parameter description
'''
def RecFilters(clip, hbegin, vbegin, hend, vend, mrange, filterings, prerequisites=None):
	funcname			= "RecFilter"

	mask				= cMask(clip, hbegin, vbegin, hend, vend, mrange)
	cropclip			= core.std.Crop(clip, hbegin, clip.width-hend, vbegin, clip.height-vend)

	# filtered clip
	fclip = scsFilters(cropclip, filterings, prerequisites)
	fclip				= core.std.AddBorders(fclip, hbegin, clip.width-hend, vbegin, clip.height-vend)

	return core.std.MaskedMerge(clip, fclip, mask, planes=0)
####################################



####################################
#####        Create Mask       #####
####################################
''' Parameter description
		clip		- specified the clip to copy the fps and etc.
		hbegin		- horizontal offset
		vbegin		- vertical offset
		hend		- horizontal end
		vend		- vertical end
		mrange		- specify the frames you want to mask 
'''
def cMask(clip, hbegin, vbegin, hend, vend, mrange):
	funcname = "cMask"
	if not isinstance(clip, vs.VideoNode):
		raise TypeError(funcname + ': \"input\" must be a clip!')

	cfamily			= clip.format.color_family
	bitdepth		= clip.format.bits_per_sample
	cwidth			= clip.width
	cheight			= clip.height
	sIsRGB			= cfamily == vs.RGB
	sIsYUV			= cfamily == vs.YUV

	if sIsRGB:
		white		= [255, 255, 255]
	elif sIsYUV:
		white		= [(1<<bitdepth)-1, 1 << (bitdepth-1), 1 << (bitdepth-1)]
	else:
		raise TypeError(funcname + ': color family is not supported!')

	if not isinstance(hbegin, int):
		raise TypeError(funcname + ': \"hbegin\" must be a int!')
	if not isinstance(vbegin, int):
		raise TypeError(funcname + ': \"vbegin\" must be a int!')
	if not isinstance(hend, int):
		raise TypeError(funcname + ': \"hend\" must be a int!')
	if not isinstance(vend, int):
		raise TypeError(funcname + ': \"vend\" must be a int!')
	if not isinstance(mrange, list):
		raise TypeError(funcname + ': mrange must be a list!')
	elif len(mrange) != 2:
		raise ValueError(funcname + ': mrange must contain 2 values!')
	elif not isinstance(mrange[0], int) or not isinstance(mrange[1], int):
		raise TypeError(funcname + ': mrange should be a list of 2 integer!')
	elif mrange[1] <= mrange[0]:
		raise ValueError(funcname + ': the second int must greater than the first!')

	if hend <= hbegin:
		raise ValueError(funcname + ': hend must be greater than hbegin')
	if vend <= vbegin:
		raise ValueError(funcname + ': vend must be greater than vbegin')

	mwidth			= hend - hbegin
	mheight			= vend - vbegin
	# blank clip
	bclip			= core.std.BlankClip(clip)
	# mask clip
	mclip			= core.std.BlankClip(clip , width=mwidth, height=mheight, length=mrange[1]-mrange[0], color=white)
	mclip			= core.std.AddBorders(mclip, left=hbegin, right=cwidth-hend, top=vbegin, bottom=cheight-vend)

	return bclip[:mrange[0]]+mclip+bclip[mrange[1]:]
####################################


####################################
### rewrite of clip.set_output() ###
####################################
def pmf_output(clips, debug=None, torgb=None):
	funcName = "pmf_output"

	debug = False if debug is None else debug
	torgb = False if torgb is None else torgb

	if debug:
		for i in range(len(clips)):
			if not isinstance(clips[i], vs.VideoNode):
				raise TypeError(funcName + ': \"input should be a clip!\"')
			else:
				clips[i] = core.text.FrameNum(clips[i], 8)
				clips[i] = core.text.Text(clips[i], r"Clip"+str(i+1), alignment=5)
		res = core.std.Interleave(clips)
		if torgb:
			res = mvf.ToRGB(res, full=False, depth=8)
	else:
		res = clips[0]

	res.set_output()
####################################

####################################
###       based on kirsch       ####
###  return a strong edge mask  ####
####################################
def kirsch5x(clip, bias=None, divisor=None):
	funcName = 'kirsch5x'

	if not isinstance(clip, vs.VideoNode):
		raise TypeError(funcName + ': clip must be a VideoNode!')
		
	if divisor is None:
		divisor = 0.8
	elif not isinstance(divisor, float):
		raise TypeError(funcName + ': divisor must be a float!')
		
	if bias is None:
		bias = 0.0
	elif not isinstance(bias, float):
		raise TypeError(funcName + ': bias must be a float!')
	
	core = vs.get_core()
	matrix    = [[]*25]*8
	matrix[0] = [3]*10 + [-2]*15
	matrix[1] = [-2]*15 + [3]*10
	matrix[2] = ( [-2]*3 + [3]*2 ) * 5
	matrix[3] = ( [3]*2 + [-2]*3 ) * 5
	for i in range(1, 6):
		matrix[4] = matrix[4] + [-2]*i + [3]*(5-i)
	for i in range(4,-1,-1):
		matrix[5] = matrix[5] + [3]*i + [-2]*(5-i)
	for i in range(5, 0,-1):
		matrix[6] = matrix[6] + [-2]*i + [3]*(5-i)
	for i in range(0, 5):
		matrix[7] = matrix[7] + [3]*i + [-2]*(5-i)
		
	clips = []
	for i in range(8):
		clips.append(
		core.std.Convolution(clip, matrix[i], bias, divisor, planes=0))
		#core.std.Convolution(clip, matrix, bias, divisor, planes, saturate, mode)
	
	expr = ['x y max z max a max b max c max d max e max', ""]
	return core.std.Expr(clips, expr)
####################################


''' abandoned functions

###########################################
### DEPRECATED USE std.Binarize INSTEAD ###
###########################################
# RETURN A MASK WITH x>valve SET TO 255 WHILE x<valve SET TO 0
## set chroma to True keep origin else set chroma to 128
def ld_mask(clip, valve=None, planes=None):
	funcName = "ld_mask"
	core = vs.get_core()

	if valve is None:
		valve = 75
	elif not isinstance(valve, int):
		raise TypeError(funcName + ': \"valve\" must be a int!')
	elif valve < 0 or valve > 255:
		raise ValueError(funcName + ': \"valve\" out of range!')

	if planes is None:
		planes = [0]
	elif not isinstance(planes, list):
		raise TypeError(funcName + ': \"planes mus be a list of int!\"')

	matrix_5x5 = [1] * 25
	clip = core.std.Convolution(clip=clip, matrix=matrix_5x5, planes=planes)

	# if Y is None:
	# 	Y = True
	# elif not isinstance(Y, int):
	# 	raise TypeError(funcName + ': \"Y\" must be a bool!')

	# if Chroma is None:
	# 	Chroma = False
	# elif not isinstance(Chroma, int):
	# 	raise TypeError(funcName + ': \"Chroma\" must be a bool!')

	expstr = "x " + str(valve) + " > 255 0 ?"
	explist = ["x","128","128"]

	if 0 in planes:
		explist[0] = expstr
	if 1 in planes:
		explist[1] = "x"
	if 2 in planes:
		explist[2] = "x"
	
	return core.std.Expr(clip, explist)
###########################################

####################################
def stretchluma(clip, thrL, thrH, planes=None):	
	funcName = 'stretchluma'

	if planes is None:
		planes=0
	elif not isinstance(planes, list) or not isinstance(planes, int):
		raise TypeError('stretchluma: planes must be a list or int!')
	elif planes != 0:
		raise TypeError('stretchluma: chroma plane not supported yet!')

	core = vs.get_core()

	strL = 500
	strH = 65000
	width = thrH - thrL
	width_str = strH - strL
	# (x-thrL) * width_str / width + strL
	expr = "x "+str(thrL)+" - "+str(width_str)+" * "+str(width)+" / "+str(strL)+" +"

	def getL2Hmask(clip, thrL, thrH, planes):
		blankclip = core.std.Expr(clip, ["0", ""])
		mask_lowluma = core.std.Binarize(clip, threshold=thrL, planes=planes)
		mask_highluma = core.std.Binarize(clip, threshold=thrH, planes=planes)
		clip_Low2Max = core.std.MaskedMerge(blankclip, clip, mask_lowluma, planes=planes)
		clip_Low2High = core.std.MaskedMerge(clip_Low2Max, blankclip, mask_highluma, planes=planes)
		return clip_Low2High

	clip_stretch = core.std.Expr(getL2Hmask(clip, thrL, thrH, planes), [expr ,""])
	return clip_stretch
####################################

'''