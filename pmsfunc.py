
#%%
import vapoursynth as vs
import mvsfunc as mvf
import numpy as np
import functools
import subprocess
# from numba import jit

# PALLADIAWA MOBIACIKAKA's FUNCTION
MODULE_NAME = 'pmsfunc'
  
core = vs.core

# %%
# @jit
def FAMask(clip):
	def DarkStrengthen(mask, shift=None):
		bps         = mask.format.bits_per_sample
		shift		= 2 if shift is None else shift
		thr         = 1 << (bps-shift)
		peek		= 1 << bps
		scv			= "y "+str(peek)+" x - "+str(thr)+" / 2 pow * "
		exp			= scv + str(peek)+" < "+scv+str(peek-1)+" ?"
		ds			= core.std.Expr([clip, mask], [scv, ""])
		mmask		= core.std.MaskedMerge(ds, mask, clip, planes=0)
		return mmask

	def LightStrengthen(mask, shift=None):
		bps         = mask.format.bits_per_sample
		shift		= 2 if shift is None else shift
		peek		= 1 << bps
		thr         = peak - (1 << (bps-shift))
		# x > thr ? 
		scv			= "y "+str(peek)+" x - "+str(thr)+" / 2 pow * "
		exp			= scv + str(peek)+" < "+scv+str(peek-1)+" ?"
		ds			= core.std.Expr([clip, mask], [exp, ""])
		return mmask

	mask1 = core.std.Prewitt(clip, planes=0)
	mask2 = DarkStrengthen(mask1)
	# m3 = LightStrengthen(m1)

	mask1 = mask1.std.Maximum().std.Minimum()
	mask3 = core.std.Expr([mask2, mask1], ["x y - y 2 * +",""])
	# m21.set_output(0)
	
	return mask3

# %%
############################################################################################################
'''	Dark Field Strengthen Mask
	Parameters Desciption
		mode	{int}	:
		thr		{int}	:
		planes	{int}	: 
'''
############################################################################################################
# @jit
def DFSMask(clip, mode=None, thr=None, scale=None, planes=None):
	funcname = "DFSMask"

	if not isinstance(clip, vs.VideoNode):
		raise vs.Error(funcname+": \"clip\" must be a VideoNode!")
	# if not isinstance(mode, int):
	# 	raise vs.Error(funcname+": \"mode\" must be a integer!")
	# if not isinstance(thr, int):
	# 	raise vs.Error(funcname+": \"thr\" must be a integer!")
	# if not isinstance(scale, float):
	# 	raise vs.Error(funcname+": \"thr\" must be a float!")
	# if not isinstance(planes, list) or not isinstance(planes, int):
	# 	raise vs.Error(funcname+": \"planes\" must be a list!")

	mode		= 0 if mode is None else mode
	bps			= clip.format.bits_per_sample
	thr			= 1 << (bps-2) if thr is None else thr
	peek		= 1 << bps
	planes		= [0,1,2] if planes is None else planes
	planes		= [planes] if isinstance(planes, int) else planes

	if	 mode	== 0:
		mask = core.std.Prewitt(clip, planes=planes)
	elif mode	== 1:
		mask = core.std.Sobel(clip, planes=planes)
	elif mode	== 2:
		mask = core.tcanny.TCanny(clip, planes=planes, sigma=0.5, mode=1, op=1)
	else:
		raise vs.Error("\"mode\" should be in range 0 to 3!")

	scv			= "y "+str(peek)+" x - "+str(thr)+" / 2 pow * "
	exp			= scv + str(peek)+" < "+scv+str(peek-1)+" ?"
	expy		= exp if 0 in planes else ""
	expu		= exp if 1 in planes else ""
	expv		= exp if 2 in planes else ""

	ds			= core.std.Expr([clip, mask], [expy, expu, expv])
	mmask		= core.std.MaskedMerge(ds, mask, clip, planes=planes)
	return mmask
############################################################################################################

# %%
# @jit
def kirsch3x(clip):
	matrix = []
	matrix.append([-3]*6+[5]*3)
	matrix.append([5]*3+[-3]*6)
	matrix.append([-3,-3, 5]*3)
	matrix.append([ 5,-3,-3]*3)
	clips = [core.std.Convolution(clip, matrix[i], planes=0) for i in range(len(matrix))]
	expr = ['x y max z max a max', ""]
	return core.std.Expr(clips, expr)

#%%
####################################
###       based on kirsch       ####
###  return a strong edge mask  ####
####################################
# @jit
def kirsch5x(clip, bias=None, divisor=None):
	funcName = 'kirsch5x'

	if not isinstance(clip, vs.VideoNode):
		raise TypeError(funcName + ': clip must be a VideoNode!')
		
	if divisor is None:
		divisor = 1
	elif not isinstance(divisor, float):
		raise TypeError(funcName + ': divisor must be a float!')
		
	if bias is None:
		bias = 0.0
	elif not isinstance(bias, float):
		raise TypeError(funcName + ': bias must be a float!')

	matrix = []
	matrix.append([4]*10 + [-3]*15)
	matrix.append([-3]*15 + [4]*10)
	matrix.append(( [-3]*3 + [4]*2 ) * 5)
	matrix.append(( [4]*2 + [-3]*3 ) * 5)
	matrix.append([-3, 4, 4, 4, 4, -3, -3, 4, 4, 4, -3, -3, -3, 4, 4, -3, -3, -3, -3, 4, -3, -3, -3, -3, -3])
	matrix.append([4, 4, 4, 4, -3, 4, 4, 4, -3, -3, 4, 4, -3, -3, -3, 4, -3, -3, -3, -3, -3, -3, -3, -3, -3])
	matrix.append([-3, -3, -3, -3, -3, -3, -3, -3, -3, 4, -3, -3, -3, 4, 4, -3, -3, 4, 4, 4, -3, 4, 4, 4, 4])
	matrix.append([-3, -3, -3, -3, -3, 4, -3, -3, -3, -3, 4, 4, -3, -3, -3, 4, 4, 4, -3, -3, 4, 4, 4, 4, -3])

	clips = [core.std.Convolution(clip, matrix[i], bias, divisor, planes=0) for i in range(len(matrix))]
	expr = ['x y max z max a max b max c max d max e max', ""]
	# expr = ['x y max z max a max', ""]
	return core.std.Expr(clips, expr)
####################################

#%%
####################################
###       based on kirsch       ####
###  return a strong edge mask  ####
###  with some details.		    ####
####################################
# @jit
def kirsch5xMod(clip, bias=None, divisor=None, planes=None):
	funcName = 'kirsch5x'

	if not isinstance(clip, vs.VideoNode):
		raise TypeError(funcName + ': clip must be a VideoNode!')
		
	if divisor is None:
		divisor = 1
	elif not isinstance(divisor, float):
		raise TypeError(funcName + ': divisor must be a float!')
		
	if bias is None:
		bias = 0.0
	elif not isinstance(bias, float):
		raise TypeError(funcName + ': bias must be a float!')

	if isinstance(planes, int):
		planes = [planes]

	matrix = []
	matrix.append([3]*10 + [-2]*15)
	matrix.append([-2]*15 + [3]*10)
	matrix.append(( [-2]*3 + [3]*2 ) * 5)
	matrix.append(( [3]*2 + [-2]*3 ) * 5)
	matrix.append([-2, 3, 3, 3, 3, -2, -2, 3, 3, 3, -2, -2, -2, 3, 3, -2, -2, -2, -2, 3, -2, -2, -2, -2, -2])
	matrix.append([3, 3, 3, 3, -2, 3, 3, 3, -2, -2, 3, 3, -2, -2, -2, 3, -2, -2, -2, -2, -2, -2, -2, -2, -2])
	matrix.append([-2, -2, -2, -2, -2, -2, -2, -2, -2, 3, -2, -2, -2, 3, 3, -2, -2, 3, 3, 3, -2, 3, 3, 3, 3])
	matrix.append([-2, -2, -2, -2, -2, 3, -2, -2, -2, -2, 3, 3, -2, -2, -2, 3, 3, 3, -2, -2, 3, 3, 3, 3, -2])

	clips = [core.std.Convolution(clip, matrix[i], bias, divisor, planes=planes) for i in range(len(matrix))]
	exp = 'x y max z max a max b max c max d max e max'
	expy = exp if 0 in planes else ""
	expu = exp if 1 in planes else ""
	expv = exp if 2 in planes else ""
	return core.std.Expr(clips, [expy, expu, expv])
####################################

#%%
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

#%%
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


#%%
####################################
#####		Create Mask       #####
####################################
''' Parameter description
		clip		- specified the clip to copy the fps and etc.
		hbegin		- horizontal offset
		vbegin		- vertical offset
		hend		- horizontal end
		vend		- vertical end
		mrange		- specify the frames you want to mask 
'''
def cMask(	clip	: vs.VideoNode	, 
			hbegin	: int			, 
			vbegin	: int			, 
			hend	: int			,
			vend	: int			,
			invert	: bool = False	) -> vs.VideoNode :

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
		black		= [  0,   0,   0]
	elif sIsYUV:
		white		= [(1 << bitdepth)-1, 1 << (bitdepth-1), 1 << (bitdepth-1)]
		black		= [0, 1 << (bitdepth-1), 1 << (bitdepth-1)]
	else:
		raise TypeError(funcname + ': color family is not supported!')

	# if len(mrange) != 2:
	# 	raise ValueError(funcname + ': mrange must contain 2 values!')
	# elif not isinstance(mrange[0], int) or not isinstance(mrange[1], int):
	# 	raise TypeError(funcname + ': mrange should be a list of 2 integer!')
	# elif mrange[1] <= mrange[0]:
	# 	raise ValueError(funcname + ': the second int must greater than the first!')

	if hend <= hbegin:
		raise ValueError(funcname + ': hend must be greater than hbegin')
	if vend <= vbegin:
		raise ValueError(funcname + ': vend must be greater than vbegin')
	if hbegin % 2 != 0 or hend % 2 != 0 or vbegin % 2 != 0 or vend % 2 != 0:
		raise ValueError(funcname + ': scope parameters must be a even!')

	if invert == True:
		c1, c2 = white, black
	else:
		c1, c2 = black, white

	mwidth			= hend - hbegin
	mheight			= vend - vbegin
	# blank clip
	bclip			= core.std.BlankClip(clip)
	# mask clip
	mclip			= core.std.BlankClip(clip , width=mwidth, height=mheight, length=clip.num_frames, color=c2)
	mclip			= core.std.AddBorders(mclip, left=hbegin, right=cwidth-hend, top=vbegin, bottom=cheight-vend, color=c1)

	# return bclip[:mrange[0]]+mclip+bclip[mrange[1]:]
	return mclip
####################################


#%%
####################################
### rewrite of clip.set_output() ###
####################################
def pmf_output(	clips		: list			, 
				debug		: bool	= None	, 
				torgb		: bool	= None	) -> vs.VideoNode :
	funcName = "pmf_output"

	debug = True  if debug is None else debug
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


#%%
''' encode helper

'''
def encode(clip: vs.VideoNode, binary: str, output_file: str, **args) -> None:
	"""Stolen from lyfunc
	Args:
		clip (vs.VideoNode): Source filtered clip
		binary (str): Path to x264 binary.
		output_file (str): Path to the output file.
	"""
	cmd = [binary,
		   "--demuxer", "y4m",
		   "--frames", f"{clip.num_frames}",
		   "--sar", "1:1",
		   "--output-depth", "10",
		   "--output-csp", "i420",
		   "--colormatrix", "bt709",
		   "--colorprim", "bt709",
		   "--transfer", "bt709",
		   "--no-fast-pskip",
		   "--no-dct-decimate",
		   "--partitions", "all",
		   "-o", output_file,
		   "-"]
	for i, v in args.items():
		i = "--" + i if i[:2] != "--" else i
		i = i.replace("_", "-")
		if i in cmd:
			cmd[cmd.index(i)+ 1] = str(v)
		else:
			cmd.extend([i, str(v)])

	print("Encoder command: ", " ".join(cmd), "\n")
	process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
	clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
				print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
	process.communicate()

#%%

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