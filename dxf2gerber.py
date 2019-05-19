#!/usr/bin/env python


# used file specs:
# gerber - http://www.ucamco.com/Portals/0/Documents/Ucamco/RS-274X_Extended_Gerber_Format_Specification_201201.pdf
# dxf 2012 - http://images.autodesk.com/adsk/files/acad_dxf1.pdf

import re
import os
import sys
import glob
import math
import numpy
import codecs
from ttfquery import glyph,describe,ttffiles,glyphquery

epsilon = sys.float_info.epsilon

# introduce scaling option ( in relation to mm)
# e.g. autocad drawing in micron units -> use 0.001
scaling = 0.001 # values will be multiplied by 0.001


def perpendicular( a ) :
    return numpy.array([a[1],-a[0]])

def normalize(a):
    return a/numpy.linalg.norm(a)

def intersection(s1,e1,s2,e2):
    d = (s1[0]-e1[0])*(s2[1]-e2[1]) - (s1[1]-e1[1])*(s2[0]-e2[0])
    if math.fabs(d) < 2*epsilon: return None
    
    f1 = s1[0]*e1[1]-s1[1]*e1[0]
    f2 = s2[0]*e2[1]-s2[1]*e2[0]
    
    return ((s2-e2)*f1-(s1-e1)*f2)/d

def load_font(fontname,allow_fallback, ttf_registry = ttffiles.Registry()):
    try:                return describe.openFont("C:/Windows/Fonts/" + fontname)
    except IOError:     pass
    
    try:
        if 0 == len(ttf_registry.fonts):
            ttf_registry.scan()
        return ttf_registry.fontFile(os.path.splitext(fontname)[0])
    except KeyError:
        if allow_fallback:
            print "could not load font",fontname," so try to fall back to arial"
            return load_font("arial.ttf",False)
        else:
            raise IOError("could not load font",fontname," and no fallback is allowed->we are done")

def line_ray_in_x_direction_intersection(s_x,s_y,e_x,e_y,p_x,p_y,verbose=0):
    apply_parallel_correction   = False
    hit                         = False
    if min(e_y,s_y) <= p_y and p_y < max(e_y,s_y) and (p_x < s_x or p_x < e_x):
        dx = e_x - s_x
        dy = e_y - s_y
        if verbose: print "dx,dy",dx,dy,
        if abs(dy)<10**-12:#parallel to x direction
            apply_parallel_correction = True
            if verbose: print "correction"
        elif (p_x < s_x and p_x < e_x) or p_x-s_x < (p_y-s_y) * dx/dy:
            hit = True
            if verbose: print "hit"
        else:
            if verbose: print "miss"
    else:
        if verbose: print "outside/miss"
        
    return hit,apply_parallel_correction

def check_A_within_B(A,B):
    """checks if A lies within B
        using a ray casting approach"""
    #NOTE: direct hit through two lines is not recognised correctly
    #because both lines will add +1 for this case. Check for "going inside" has to be added.
    crossing_count              = 0
    p_x,p_y                     = A[0][1:3]
    last_stored_y               = None
    apply_parallel_correction   = False
    verbose                     = 0#2
    
    if verbose > 0: print "DEBUG: testing",len(A),"vs",len(B),"point",(p_x,p_y)
    for segment in B:
        type,s_x,s_y,e_x,e_y = segment[:5]
        if verbose > 1: print type,
        if type == "straight":#ray line intersection with infinite ray from (p_x,p_y) to (inf,p_y) of first point of A
            #print "DEBUG:",s_x,s_y,e_x,e_y,p_x,p_y
            hit,correct = line_ray_in_x_direction_intersection(s_x,s_y,e_x,e_y,p_x,p_y,verbose > 1)
            if hit:     crossing_count += 1
            if correct: apply_parallel_correction = correct
        elif type == "arc":#ray arc intersection with infinite ray from (p_x,p_y) to (p_x+1,p_y) of first point of A
            r_x,r_y,ccw = segment[5:]
            c_x     = s_x + r_x
            c_y     = s_y + r_y
            within      = (p_x-c_x)**2 + (p_y-c_y)**2 <= r_x**2 + r_y**2
            hit,correct = line_ray_in_x_direction_intersection(s_x,s_y,e_x,e_y,p_x,p_y,verbose > 1)
            
            if within:
                left = (e_x-s_x)*(p_y-s_y) - (e_y-s_y)*(p_x-s_x) > 0
                
                if (left == ccw) == hit:
                    crossing_count += 1
                
            elif hit:
                crossing_count += 1
        else:
            print "WARNING: unknown boundary type found (check_A_within_B)"
            
        delta_y = e_y - s_y
        if delta_y:
            if apply_parallel_correction:
                if not last_stored_y:
                    for i in range(len(B)-1,0,-1):
                        previous_y = B[i][4]-B[i][2]
                        if previous_y:
                            last_stored_y = previous_y
                            break

                if delta_y*last_stored_y < 0:#y direction is discontinuity
                    crossing_count += 1
                    if verbose > 1: print "DEBUG: corrected discontinuity",crossing_count
                apply_parallel_correction = False
            last_stored_y = delta_y

    if verbose > 0: print "DEBUG: crossing_count",crossing_count
    return crossing_count % 2 == 1

#######################################################################
# converting + translation
#######################################################################

def convert_circle(dict_values,default_thickness,autofill):
    layer   = dict_values[8][0][0]
    x       = float(dict_values[10][0][0])#*scaling
    y       = float(dict_values[20][0][0])#*scaling
    r       = float(dict_values[40][0][0])*2*scaling    #keep scaling here?
    device  = "C,%f*"%r if autofill else "C,%fX%f*"%(r,r-default_thickness)
    
    return [{"layer":layer,"device":device,"x":x,"y":y,"style":"D03"}],True

def convert_arc(dict_values,default_thickness):
    layer       = dict_values[8][0][0]
    center_x    = float(dict_values[10][0][0])#*scaling
    center_y    = float(dict_values[20][0][0])#*scaling
    thickness   = float(dict_values[39][0][0]) if 39 in dict_values else 0.0
    radius      = float(dict_values[40][0][0])#*scaling
    start_angle = float(dict_values[50][0][0])/180.0*math.pi
    end_angle   = float(dict_values[51][0][0])/180.0*math.pi
    
    rel_x       = radius * math.cos(start_angle)
    rel_y       = radius * math.sin(start_angle)
    start_x     = center_x + rel_x
    start_y     = center_y + rel_y
    end_x       = center_x + radius * math.cos(end_angle)
    end_y       = center_y + radius * math.sin(end_angle)
    
    objects = []
    device = "C,%f*"%(thickness if thickness else default_thickness)
    objects.append({"layer":layer,"command":"G75*"})
    objects.append({"layer":layer,"device":device,"mode":"G01","x":start_x,"y":start_y,"style":"D02"})
    objects.append({"layer":layer,"device":device,"mode":"G03","style":"D01","x":end_x,"y":end_y,"i":-rel_x,"j":-rel_y})
    objects.append({"layer":layer,"command":"G01*"})

    return objects,True

def convert_ellipse(dict_values,default_thickness,autofill):
    layer           = dict_values[8][0][0]
    thickness       = float(dict_values[39][0][0]) if 39 in dict_values else 0.0
    center_x        = float(dict_values[10][0][0])#*scaling
    center_y        = float(dict_values[20][0][0])#*scaling
    major_x         = float(dict_values[11][0][0])#*scaling
    major_y         = float(dict_values[21][0][0])#*scaling
    minor_to_major  = float(dict_values[40][0][0])# axis ratio, no scaling needed here?
    start_angle     = float(dict_values[41][0][0])
    end_angle       = float(dict_values[42][0][0])

    tilt        = math.atan(major_y/major_x) if major_x else math.copysign(math.pi/2,major_y)
    sampling    = int((end_angle - start_angle)*8/math.pi)#a full cycle will be sampled with 2pi/(pi/8)= 16 polylines
    device      = "C,%f*"%(thickness if thickness else default_thickness)
    major_len   = math.sqrt(major_x**2+major_y**2)
    minor_len   = (major_x**2+major_y**2) * (minor_to_major**2)
    objects     = []
    
    if autofill:
        objects.append({"layer":layer,"command":"G36*"})
    
    for step in range(sampling):
        t = (end_angle - start_angle) * step / sampling
        x = center_x + major_len*math.cos(t)*math.cos(tilt) - minor_len*math.sin(t)*math.sin(tilt)
        y = center_y + major_len*math.cos(t)*math.sin(tilt) - minor_len*math.sin(t)*math.cos(tilt)
        objects.append({"layer":layer,"device":device,"x":x,"y":y,"style":"D01"})
        
    objects[0]["style"] = "D02"
    
    if autofill:
        objects.append({"layer":layer,"command":"G37*"})
        
    return objects,True

def convert_line(dict_values,default_thickness):
    layer       = dict_values[8][0][0]
    thickness   = float(dict_values[39][0][0]) if 39 in dict_values else 0.0
    start_x     = float(dict_values[10][0][0])#*scaling
    start_y     = float(dict_values[20][0][0])#*scaling
    stop_x      = float(dict_values[11][0][0])#*scaling
    stop_y      = float(dict_values[21][0][0])#*scaling
    
    device = "C,%f*"%(thickness if thickness else default_thickness)
    return [{"layer":layer,"device":device,"x":start_x,"y":start_y,"style":"D02"},
            {"layer":layer,"device":device,"x":stop_x,"y":stop_y,"style":"D01"}],True

def convert_lwpolyline(dict_values,default_thickness,autofill):
    layer       = dict_values[8][0][0]
    count       = int(dict_values[90][0][0])
    closed      = int(dict_values[70][0][0]) == 1
    points      = []
    copy_bulge  = list(dict_values[42]) if 42 in dict_values else []

    for i in range(count-1,-1,-1):
        x,no_x          = dict_values[10][i]
        y,no_y          = dict_values[20][i]
        #x*=scaling#
        #y*=scaling#
        bulge,no_bulge  = copy_bulge.pop() if len(copy_bulge) and copy_bulge[-1][1] > no_x else (None,None)
        points.insert(0,(numpy.array((float(x),float(y))),bulge))
    
    if closed:
        count += 1
        points.append(points[0])
        
    if 43 in dict_values:   #need scaling here?
        v = float(dict_values[43][0][0])
        widths = [(v,v)]*count
    else:
        widths = map(lambda x,y:(float(x[0]),float(y[0])),dict_values[40],dict_values[41])
        
    lines = list()
    for i in range(count-1): 
        start,start_bulge   = points[i]
        end,end_bulge       = points[i+1]
        w1,w2               = widths[i]
        
        ortholine           = perpendicular(end - start)
        ortholine_l         = numpy.linalg.norm(ortholine)
        
        if start_bulge:
            opening_angle   = math.atan(float(start_bulge))*4
        
            rel_start   = (end - start)/2 - ortholine / (2 * math.tan(opening_angle/2))
            rel_end     = start - end + rel_start

            radius      = numpy.linalg.norm(rel_start)
            
            start1      = start - rel_start * 0.5 * w1 / radius
            start2      = start + rel_start * 0.5 * w1 / radius
            end1        = end - rel_end * 0.5 * w2 / radius
            end2        = end + rel_end * 0.5 * w2 / radius
        elif ortholine_l:
            start1      = start - ortholine * 0.5 * w1 / ortholine_l
            start2      = start + ortholine * 0.5 * w1 / ortholine_l
            end1        = end - ortholine * 0.5 * w2 / ortholine_l
            end2        = end + ortholine * 0.5 * w2 / ortholine_l
            
            #no arc but thick line-> intersection test needed
            if w1 and w2 and i > 0 and not points[i-1][1]:
                start3,end3,start4,end4 = lines[-1][:4]
                ret1 = intersection(start1,end1,start3,end3)
                ret2 = intersection(start2,end2,start4,end4)
                
                if ret1 != None: 
                    start1 = ret1
                    lines[-1][1] = ret1
                if ret2 != None: 
                    start2 = ret2
                    lines[-1][3] = ret2
        else:
            start1      = start
            start2      = start
            end1        = end
            end2        = end
            
        lines.append([start1,end1,start2,end2,ortholine])
        
    objects = []
    for i in range(count-1):
        start,start_bulge   = points[i]
        end,end_bulge       = points[i+1]
        start_width         = widths[i][0]
        end_width           = widths[i][1]
        contours            = start_width!=0 or end_width!=0
        device              = ("C,%f*"%start_width) if start_width else "C,%f*"%default_thickness

        start1,end1,start2,end2,ortholine   = lines[i]
        ortholine_l                         = numpy.linalg.norm(ortholine)
        
        if start_bulge:
            opening_angle   = math.atan(float(start_bulge))*4
            mode            = "G03" if float(start_bulge) > 0 else "G02"
            inv_mode        = "G02" if float(start_bulge) > 0 else "G03"
            
            objects.append({"layer":layer,"command":"G75*"})
            if contours:
                ortho1      = perpendicular(end1 - start1)
                ortho2      = perpendicular(end2 - start2)
                cycle1      = -(end1 - start1)/2 - ortho1 / (2 * math.tan(opening_angle/2))
                cycle2      = (end2 - start2)/2 - ortho2 / (2 * math.tan(opening_angle/2))
                
                objects.append({"layer":layer,"command":"G36*"})
                objects.append({"layer":layer,"device":"C,0.001*","mode":"G01","x":start1[0],"y":start1[1],"style":"D02"})
                objects.append({"layer":layer,"device":"C,0.001*","mode":"G01","x":start2[0],"y":start2[1],"style":"D01"})
                objects.append({"layer":layer,"device":"C,0.001*","mode":mode,"style":"D01","x":end2[0],"y":end2[1],"i":cycle2[0],"j":cycle2[1]})
                objects.append({"layer":layer,"device":"C,0.001*","mode":"G01","x":end1[0],  "y":end1[1],  "style":"D01"})
                objects.append({"layer":layer,"device":"C,0.001*","mode":inv_mode,"style":"D01","x":start1[0],"y":start1[1],"i":cycle1[0],"j":cycle1[1]})
                objects.append({"layer":layer,"command":"G37*"})
            else:#circular arc with little round ends but who cares?
                rel = -(start - end)/2 - ortholine / (2 * math.tan(opening_angle/2))
                objects.append({"layer":layer,"device":device,"mode":"G01","x":start[0],"y":start[1],"style":"D02"})
                objects.append({"layer":layer,"device":device,"mode":mode,"style":"D01","x":end[0],"y":end[1],"i":rel[0],"j":rel[1]})
            objects.append({"layer":layer,"command":"G01*"})
        elif ortholine_l > 0:
            if contours:#using a rectangluar approach
                start1,end1,start2,end2 = lines[i][:4]                   

                objects.append({"layer":layer,"command":"G36*"})
                objects.append({"layer":layer,"device":"C,0.001*","x":start1[0],"y":start1[1],"style":"D02"})
                objects.append({"layer":layer,"device":"C,0.001*","x":start2[0],"y":start2[1],"style":"D01"})
                objects.append({"layer":layer,"device":"C,0.001*","x":end2[0],  "y":end2[1],  "style":"D01"})
                objects.append({"layer":layer,"device":"C,0.001*","x":end1[0],  "y":end1[1],  "style":"D01"})
                objects.append({"layer":layer,"device":"C,0.001*","x":start1[0],"y":start1[1],"style":"D01"})
                objects.append({"layer":layer,"command":"G37*"})
            else:#straight line with little round ends but who cares?
                if not ortholine_l:
                    print "warning: found a thick polygon line with a length of %d. cannot handle that correctly=>fallback to simple line."%ortholine_l
                    #print i,count,points[i],points[i+1]
                objects.append({"layer":layer,"device":device,"x":start[0],"y":start[1],"style":"D02"})
                objects.append({"layer":layer,"device":device,"x":end[0],  "y":end[1],  "style":"D01"})
                
    #NOTE: try really closed poly and remove doubled lines
    if autofill and closed:
        objects = [o for o in objects if o.get("command") not in ("G36*","G37*")]
        first = True
        for o in objects:
            if "style" in o:
                o["style"] = "D02" if first else "D01"
                first = False
        objects.insert(0,{"layer":layer,"command":"G36*"})
        objects.append({"layer":layer,"command":"G37*"})
    return objects, True

def convert_hatch(dict_values):
    pattern_name    = dict_values[2][0][0]
    layer           = dict_values[8][0][0]
    solid           = int(dict_values[70][0][0]) == 1
    hatchstyle      = int(dict_values[75][0][0])
    loops           = int(dict_values[91][0][0])
    negative        = int(dict_values[75][0][0]) == 1
    seedpoints      = int(dict_values[98][0][0])
    boundaries      = list()
    ret             = list()
    
    print "INFO: the current implementation does NOT support intersections between hatch boundaries."
    if not solid:
        print "INFO: currently only solid fill is supported so the hatch pattern is ignored."
        
    #remove Elevation point
    dict_values[10].pop(0)
    dict_values[20].pop(0)
        
    #parse boundary paths
    for type_flag,no in dict_values[92]:
        type_flag   = int(type_flag)
        external    = type_flag & 1 != 0
        polyline    = type_flag & 2 != 0
        derived     = type_flag & 4 != 0
        textbox     = type_flag & 8 != 0
        outermost   = type_flag & 16 != 0
        points      = list()
        
        #print external,polyline,derived,derived,textbox,outermost
        
        if polyline:
            has_bulge       = bool(dict_values[72].pop(0)[0])
            is_closed       = bool(dict_values[73].pop(0)[0])
            vertex_count    = int(dict_values[93].pop(0)[0])
            
            print "WARNING: polylines in hatches are untested because they never occurred to me. If you have an example please send it to me." 
            print "DEBUG: vertex_count",vertex_count
            for i in range(vertex_count):
                start_x     = float(dict_values[10].pop(0)[0])
                start_y     = float(dict_values[20].pop(0)[0])
                bulge       = float(dict_values[42].pop(0)[0]) if has_bulge else 0
                end_x       = float(dict_values[10][0][0])#note: careful-is this value for the last entry correct or from another boundary?
                end_y       = float(dict_values[20][0][0])
                
                print "DEBUG: polyline start-end",start_x,start_y,end_x,end_y
                if bulge:
                    opening_angle   = math.atan(float(bulge))*4
                    ccw             = "G03" if float(bulge) > 0 else "G02"
                    ortholine_x     = end_y - start_y
                    ortholine_y     = -(end_x - start_x)
                    rel_x           = -(start_x - end_x)/2 - ortholine_x / (2 * math.tan(opening_angle/2))
                    rel_y           = -(start_y - end_y)/2 - ortholine_y / (2 * math.tan(opening_angle/2))

                    points.append(("arc",start_x,start_y,end_x,end_y,rel_x,rel_y,ccw))
                else:
                    points.append(("straight",start_x,start_y,end_x,end_y))
        else:
            edge_count   = int(dict_values[93].pop(0)[0])
            #print "edge_count",edge_count
            
            for i in range(edge_count):
                edgetype    = int(dict_values[72].pop(0)[0])
                #print "edgetype",edgetype
                if edgetype == 1:#line
                    start_x     = float(dict_values[10].pop(0)[0])
                    start_y     = float(dict_values[20].pop(0)[0])
                    end_x       = float(dict_values[11].pop(0)[0])
                    end_y       = float(dict_values[21].pop(0)[0])
                    
                    points.append(("straight",start_x,start_y,end_x,end_y))
                    
                elif edgetype == 2: #circular arc
                    center_x    = float(dict_values[10].pop(0)[0])
                    center_y    = float(dict_values[20].pop(0)[0])
                    radius      = float(dict_values[40].pop(0)[0])
                    start_angle = (float(dict_values[50].pop(0)[0]))/180.0*math.pi
                    end_angle   = (float(dict_values[51].pop(0)[0]))/180.0*math.pi
                    ccw         = int(dict_values[73].pop(0)[0]) == 1

                    rel_x       = radius * math.cos(start_angle)
                    rel_y       = radius * math.sin(start_angle) * (1 if ccw else -1)
                    start_x     = center_x + rel_x
                    start_y     = center_y + rel_y
                    end_x       = center_x + radius * math.cos(end_angle)
                    end_y       = center_y + radius * math.sin(end_angle) * (1 if ccw else -1)
                    
                    points.append(("arc",start_x,start_y,end_x,end_y,-rel_x,-rel_y,ccw))
                    
                #elif edgetype == 3: #elliptic arc
                #elif edgetype == 4: #spline
                else:
                    print "WARNING: unhandled edgetype", edgetype, ".Try to ignore it but mostelike it will fail."
                    next_no = dict_values[72][1]
                    for key,lists in dict_values.items():
                        dict_values[key] = [x for x in lists if x[1] > next_no]
                        
        if not len(points) or abs(points[-1][3]-points[0][1]) >10**-12 or abs(points[-1][4]-points[0][2]) >10**-12:
            print "WARNING: not closed objects are NOT supported within a hatch. Excluding: ",points
        elif external or hatchstyle == 0 or (outermost and hatchstyle == 1):
            boundaries.append(points)
        
        #num_source_boundarys = int(dict_values[97].pop(0)[0])
        #for i in range(num_source_boundarys):
        #    print "ref: ", dict_values[330].pop(0)[0]
    #building dependency/lies within tree
    if not len(boundaries):
        print "WARNING: empty hatch found."
        return [],True

    roots   = [(boundaries.pop(),list())]
    while len(boundaries):
        branch      = roots
        points      = boundaries.pop()
        searching   = True
        children    = list()
        
        while searching:
            searching = False
            for other,subobj in branch[:]:
                if check_A_within_B(points,other):#our boundary lies within other
                    branch      = subobj
                    searching   = True
                    #print len(points),"lies within",len(other),"(1)"
                    break
                elif check_A_within_B(other,points):#other lies within our boundary
                    branch.remove((other,subobj))
                    children.append((other,subobj))
                    #print len(other),"lies within",len(points),"(2)"
            if not searching:
                branch.append((points,children))

    darkfield   = True
    drawlist    = roots
    while len(drawlist):
        new_drawlist = list()

        #print "DEBUG: drawing",len(drawlist),"with parity", darkfield
        for points,subobjs in drawlist:
            new_drawlist.extend(subobjs)
        
            last_x      = None
            last_y      = None
            last_parity = darkfield
            
            ret.append({"layer":layer,"command":"G36*"})
            for p in points:
                type,start_x,start_y,end_x,end_y = p[:5]
                if p[0] == "straight":
                    if not last_x or abs(last_x-start_x)>10**-12 or abs(last_y-start_y)>10**-12:
                        ret.append({"layer":layer,"device":"C,0.001*","x":start_x,"y":start_y,"style":"D02"})
                    ret.append({"layer":layer,"device":"C,0.001*","x":end_x,"y":end_y,"style":"D01"})
                else:#arc
                    rel_x,rel_y,ccw = p[5:]
                    
                    ret.append({"layer":layer,"command":"G75*"})
                    if not last_x or abs(last_x-start_x)>10**-12 or abs(last_y-start_y)>10**-12:
                        ret.append({"layer":layer,"device":"C,0.001*","mode":"G01","x":start_x,"y":start_y,"style":"D02"})
                    ret.append({"layer":layer,"device":"C,0.001*","mode":"G03" if ccw else "G02","style":"D01","x":end_x,"y":end_y,"i":rel_x,"j":rel_y})
                    ret.append({"layer":layer,"command":"G01*"})
                last_x = end_x
                last_y = end_y
            ret.append({"layer":layer,"command":"G37*"})
            
        darkfield   = not darkfield
        drawlist    = new_drawlist
        ret.append({"layer":layer,"command":"%LPD*%" if darkfield else "%LPC*%"})
        
    ret.append({"layer":layer,"command":"%LPD*%"})
    return ret,False

def convert_text(dict_values, fonts,mtext):
    layer       = dict_values[8][0][0]
    text        = ("".join(dict_values[3][0][0]) if 3 in dict_values else "") + dict_values[1][0][0]
    xoffset     = float(dict_values[10][0][0])
    yoffset     = float(dict_values[20][0][0])
    height      = float(dict_values[40][0][0])
    rotation    = float(dict_values[50][0][0])/180.0*math.pi if 50 in dict_values else 0
    font        = fonts[dict_values[7][0][0].lower() if 7 in dict_values else "standard"]

    if mtext:
        boxwidth    = float(dict_values[41][0][0]) if 41 in dict_values else 1
        x_scale     = 1
        attachment  = int(dict_values[71][0][0]) if 71 in dict_values else 1
        
        text        = re.sub("\\\\[^;]*;", "", text)
        text        = re.sub("([^\\\\])\\\\P", "\g<1>\\n", text)
        spacing_fac = float(dict_values[44][0][0]) if 44 in dict_values else 1
                
    else:
        boxwidth    = 1#None raised error
        x_scale     = float(dict_values[41][0][0]) if 41 in dict_values else 1
        xalign      = float(dict_values[11][0][0]) if 11 in dict_values else None#only used for Horizontal text justification typ == 3
        yalign      = float(dict_values[21][0][0]) if 11 in dict_values else None#only used for Horizontal text justification typ == 3
        drawdir     = int(dict_values[71][0][0]) if 71 in dict_values else 0#mirroring in X (=2) or Y (=4) direction
        h_text_just = int(dict_values[72][0][0]) if 72 in dict_values else 0#0=left 1=center 2=right 3=aligned
        v_text_just = int(dict_values[73][0][0]) if 73 in dict_values else 3#0=baseline 1=bottom 2=middle 3=top
        spacing_fac = 1

    for number,name in [(51,"'oblique angle'")]:
        if number in dict_values:
            print "INFO: sorry but option ",name, "(",number,") is not supported yet"

    cheight     = float(glyphquery.charHeight(font)) / 1.32#magic number
    objects     = []
    boxwidth    /= x_scale * height / cheight
    charoff     = 0
    lineno      = 1
    lines       = [[]]
        
    for n, letter in enumerate(text):
        glname  = glyphquery.glyphName(font,letter)
        glwidth = glyphquery.width(font,glname)
        
        print(glname,glwidth,letter, charoff)
        
        if letter == "\n" or (boxwidth and charoff + glwidth > boxwidth):
            lines.append(list())
            #charoff         = 0 # temprarily deactivated, sets charoff always to 0 else...
            lineno          += 1

        
        if letter != "\n":
            boundaries  = [cheight*10,cheight*10,0,0]
            for c in glyph.Glyph(glname).calculateContours(font):
                outlines    = glyph.decomposeOutline(c,2)
                first       = True
                lines[-1].append({"layer":layer,"command":"G36*"})
                for l in outlines:
                    org_x = (l[0]+charoff) * x_scale * height / cheight
                    org_y = l[1] * height / cheight
                    x = org_x * math.cos(rotation) - org_y * math.sin(rotation) + xoffset
                    y = org_x * math.sin(rotation) + org_y * math.cos(rotation) + yoffset
    
                    # changes polarity -> clear holes in fonts
                    if first and (boundaries[0] < l[0] and l[0] < boundaries[2] and boundaries[1] < l[1] and l[1] < boundaries[3]):
                        lines[-1].pop()#pop the last G36; the gerber file is not read correctly if LPC comes within a G36-G37 Block
                        lines[-1].append({"layer":layer,"command":"%LPC*%"})
                        lines[-1].append({"layer":layer,"command":"G36*"})
                    
                    boundaries[0] = min(boundaries[0],l[0])
                    boundaries[1] = min(boundaries[1],l[1])
                    boundaries[2] = max(boundaries[2],l[0])
                    boundaries[3] = max(boundaries[3],l[1])
                    
                    #x*=scaling#
                    #y*=scaling#
                    
                    lines[-1].append({"layer":layer,"device":"C,0.001*","x":x,"y":y,"style":"D02" if first else "D01"})
                    first = False
                lines[-1].append({"layer":layer,"command":"G37*"})
            lines[-1].append({"layer":layer,"command":"%LPD*%"})
            charoff += glwidth
            
        #print(lines)#


    #respect the alignment (only affects MTEXT. For TEXT it is already considered
    if mtext:
        spacing     = height/3*5*spacing_fac
        textwidth   = charoff * x_scale * height / cheight
        vertical    = int((attachment-1)/3) #0=oben 1=mitte 2=unten
        horizontal  = (attachment-1)%3 #0=links 1=mitte 2=rechts
        
        if horizontal == 0:     shift_x = 0#links
        elif horizontal == 1:   shift_x = -textwidth / 2#mitte
        elif horizontal == 2:   shift_x = -textwidth#rechts
        
        lineno = 0
        for line_objects in lines:
            if vertical == 0:       shift_y = -height -(lineno-1)*spacing#oben
            elif vertical == 1:     shift_y = (len(lines)-lineno)*spacing/ 2#mitte
            elif vertical == 2:     shift_y = (len(lines)-lineno)*spacing#unten
        
            lineno += 1
            for o in line_objects:
                if "x" in o: o["x"] += shift_x
                if "y" in o: o["y"] += shift_y

    for line_objects in lines:
        objects.extend(line_objects)

    return objects,True

def handle_insert(dict_values,default_thickness,autofill,convert,blocks):
    block_name  = dict_values[2][0][0]
    layer       = dict_values[8][0][0]
    xoffset     = float(dict_values[10][0][0])
    yoffset     = float(dict_values[20][0][0])
    xscale      = float(dict_values[41][0][0]) if 41 in dict_values else 1
    yscale      = float(dict_values[42][0][0]) if 42 in dict_values else 1
    rotation    = (float(dict_values[50][0][0])/180*math.pi) if 50 in dict_values else 0#math.pi+
    ref         = dict_values[2][0][0]
    switch_ccw  = xscale*yscale < 0
    
    #print dict_values
    objects = []
    for obj in blocks[ref]:
        obj = dict(obj)
        if "x" in obj:
            x = xscale*obj["x"]
            y = yscale*obj["y"]
            obj["x"] = (x*math.cos(rotation) - y*math.sin(rotation)) + xoffset
            obj["y"] = (x*math.sin(rotation) + y*math.cos(rotation)) + yoffset
        if "i" in obj:
            i = xscale*obj["i"]
            j = yscale*obj["j"]
            obj["i"] = (i*math.cos(rotation) - j*math.sin(rotation))
            obj["j"] = (i*math.sin(rotation) + j*math.cos(rotation))
        
        if block_name[0] != "*":
            obj["layer"] = layer
        
        if switch_ccw and "mode" in obj:
            if   obj["mode"] == "G03": obj["mode"] = "G02"
            elif obj["mode"] == "G02": obj["mode"] = "G03"
        objects.append(obj)         
    #"""
    return objects,True

#######################################################################
# parsing+writing
#######################################################################

def collect_by_commands(filename):
    number      = None
    commands    = []
    counter     = 0
    with codecs.open(filename,"r",encoding='cp1252') as file:#utf-8-sig
        for line in file:
            line = line.replace("\n","").replace("\r","")
            if number is not None:
                if number not in commands[-1]: commands[-1][number] = []
                commands[-1][number].append((line,counter))
                number  = None
                counter += 1
            else:#if len(line) == 3:
                try:
                    number = int(line)
                    if number == 999:
                        number = None
                    elif number == 0:
                        commands.append(dict())
                except ValueError:
                    pass
    return commands

def convert_dxf2gerber(filename):
    autofill = len(sys.argv) >= 3 and sys.argv[2] == "autofill"
    print "INFO: converting",filename, " with autofill ", "enabled" if autofill else "disabled"
    commands = collect_by_commands(filename)#"2010-01-07 Maske v2.dxf")#
    layers  = dict()
    blocks  = dict()
    fonts   = dict()
    
    #workaround for polylines
    open_poly = None
    for c in list(commands):
        if c[0][0][0] == "POLYLINE":
            open_poly       = c
            open_poly[10]   = list()
            open_poly[20]   = list()
            open_poly[43]   = c[39] if 39 in c else [(0.01,)]
        elif c[0][0][0] == "VERTEX" and open_poly:
            open_poly[10].append(c[10][0])
            open_poly[20].append(c[20][0])
            if 40 in c: open_poly[40].append(c[40][0])
            if 41 in c: open_poly[41].append(c[41][0])
            if 42 in c: open_poly[42].append(c[42][0])
            commands.remove(c)
        elif c[0][0][0] == "SEQEND" and open_poly:
            open_poly[90] = [(len(open_poly[10]),None)]
            commands.remove(c)
    
    convert = {'TEXT':          lambda x,y,z: convert_text(x,fonts,False),
               'MTEXT':         lambda x,y,z: convert_text(x,fonts,True),
               'LWPOLYLINE':    convert_lwpolyline,
               'POLYLINE':      convert_lwpolyline,
               'HATCH':         lambda x,y,z: convert_hatch(x),
               'CIRCLE':        convert_circle,
               'LINE':          lambda x,y,z: convert_line(x,y),
               'ARC':           lambda x,y,z: convert_arc(x,y),
               'ELLIPSE':       convert_ellipse,
               'INSERT':        lambda x,y,z: handle_insert(x,y,z,convert,blocks),}
               
    current_section = None
    for com in commands:
        name = com[0][0][0]
        if name == "SECTION":
            current_section = com[2][0][0]
        elif name == "ENDSEC":
            current_section = None
        elif name == "STYLE":
            fontname            = com[3][0][0]
            stylename           = com[2][0][0].lower()
            fonts[stylename]    = load_font(fontname,True)
        elif current_section == "BLOCKS":
            if name == "BLOCK":
                current_block           = com[2][0][0]
                blocks[current_block]   = list()
            elif name == "ENDBLK":
                current_block = None
            else:
                    name = com[0][0][0]
                    if name == "HATCH":
                        print "found a HATCH within a INSERT which can cause malfunctional drawing (elements can be erased)"
                    if name not in convert:
                        print "ignored unknown command: ", name
                    else:
                        child_objects,append = convert[name](com,0.01,autofill)
            
                        if append:  blocks[current_block] = blocks[current_block] + child_objects
                        else:       blocks[current_block] = child_objects + blocks[current_block]
        elif current_section == "ENTITIES":
            if name not in convert:
                print "WARNING: ignored unknown command: ", name
            else:
                print "INFO: converting", name
                objects,append = convert[name](com,0.01,autofill)
                counter = 0
                for o in objects:
                    if o["layer"] not in layers:    layers[o["layer"]] = [o]
                    elif append:                    layers[o["layer"]].append(o)
                    else:                           layers[o["layer"]].insert(counter,o)
                    counter += 1
            
    #write gerber
    for l,objects in layers.items():
        devices = {obj["device"]:None for obj in objects if "device" in obj}
        with open(l+".gbr","w") as file:
            file.write("%FSLAX36Y36*%\n")
            file.write("%MOMM*%\n")
            file.write("%LN"+ l + "*%\n")
        
            counter = 10
            for d in devices.keys():
                file.write("%ADD" + str(counter) + d + "%\n")
                devices[d] = counter
                counter += 1
                
            current_device = None
            for obj in objects:
                if "command" in obj:
                    file.write(obj["command"] +"\n")
                else:
                    if obj["device"] != current_device:
                        file.write("G54D%d*\n"%devices[obj["device"]])
                        current_device = obj["device"]
                        
                    line = ""
                    
                    if "mode" in obj:   line += obj["mode"]
                    if "x" in obj:      line += "X"+str(int(obj["x"] * 1000000*scaling))    #introduce scaling only here is easier...
                    if "y" in obj:      line += "Y"+str(int(obj["y"] * 1000000*scaling))
                    if "i" in obj:      line += "I"+str(int(obj["i"] * 1000000))
                    if "j" in obj:      line += "J"+str(int(obj["j"] * 1000000))
                    if "style" in obj:  line += obj["style"]
                    
                    file.write(line+"*\n")
            file.write("M02*\n")
    print "done!"
            
print sys.argv
if 2 <= len(sys.argv):    
    convert_dxf2gerber(sys.argv[1])
else:
    for filename in glob.glob("*.dxf"):
        convert_dxf2gerber(filename)