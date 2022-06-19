#!/usr/bin/env python
import numpy as np
from utils.globals import *

class Experience(object):
    "A single experience."

    def __init__(self, x_pc, y_pc, th_pc, x_m, y_m, facing_rad, view_cell):
        "Initializes the Experience."

        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.x_m = x_m
        self.y_m = y_m
        self.facing_rad = facing_rad
        self.view_cell = view_cell
        self.links = []

    def link_to(self, target, accum_delta_x, accum_delta_y, accum_delta_facing):
        "Creates a link between this experience and a taget one."

        d = np.sqrt(accum_delta_x**2 + accum_delta_y**2)
        heading_rad = signed_delta_rad(self.facing_rad, np.arctan2(accum_delta_y, accum_delta_x))
        facing_rad = signed_delta_rad(self.facing_rad, accum_delta_facing)
        link = ExperienceLink(self, target, facing_rad, d, heading_rad)
        self.links.append(link)

class ExperienceLink(object):
    "A representation of connection between experiences."

    def __init__(self, target, facing_rad, d, heading_rad):
        "Initializes the link."

        self.target = target
        self.facing_rad = facing_rad
        self.d = d
        self.heading_rad = heading_rad

class ExperienceMap(object):
    "Experience Map module."

    def __init__(self):
        "Initializes the experience map."

        self.size = 0
        self.exps = []
        
        self.current_exp = None
        self.current_view_cell = None

        self.accum_delta_x = 0
        self.accum_delta_y = 0
        self.accum_delta_facing = np.pi/2

        self.history = []

    def _create_exp(self, x_pc, y_pc, th_pc, view_cell):
        "Creates a new Experience object."

        self.size += 1
        x_m = self.accum_delta_x
        y_m = self.accum_delta_y
        facing_rad = clip_rad_180(self.accum_delta_facing)

        if self.current_exp is not None:
            x_m += self.current_exp.x_m
            y_m += self.current_exp.y_m

        exp = Experience(x_pc, y_pc, th_pc, x_m, y_m, facing_rad, view_cell)

        if self.current_exp is not None:
            self.current_exp.link_to(exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing)

        self.exps.append(exp)
        view_cell.exps.append(exp)

        return exp

    def __call__(self, view_cell, vtrans, vrot, x_pc, y_pc, th_pc):
        "Run an interaction of the experience map."

        # integrate the delta x, y, facing
        self.accum_delta_facing = clip_rad_180(self.accum_delta_facing + vrot)
        self.accum_delta_x += vtrans*np.cos(self.accum_delta_facing)
        self.accum_delta_y += vtrans*np.sin(self.accum_delta_facing)

        if self.current_exp is None:
            delta_pc = 0
        else:
            delta_pc = np.sqrt(
                min_delta(self.current_exp.x_pc, x_pc, PC_DIM_XY)**2 + \
                min_delta(self.current_exp.y_pc, y_pc, PC_DIM_XY)**2 + \
                min_delta(self.current_exp.th_pc, th_pc, PC_DIM_TH)**2
            )

        # if the vt is new or the pc x,y,th has changed enough create a new
        # experience
        adjust_map = False
        if len(view_cell.exps) == 0 or delta_pc > EXP_DELTA_PC_THRESHOLD:
            exp = self._create_exp(x_pc, y_pc, th_pc, view_cell)

            self.current_exp = exp
            self.accum_delta_x = 0
            self.accum_delta_y = 0
            self.accum_delta_facing = self.current_exp.facing_rad

        # if the vt has changed (but isn't new) search for the matching exp
        elif view_cell != self.current_exp.view_cell:

            # find the exp associated with the current vt and that is under the
            # threshold distance to the centre of pose cell activity
            # if multiple exps are under the threshold then don't match (to reduce
            # hash collisions)
            adjust_map = True
            matched_exp = None

            delta_pcs = []
            n_candidate_matches = 0
            for (i, e) in enumerate(view_cell.exps):
                delta_pc = np.sqrt(
                    min_delta(e.x_pc, x_pc, PC_DIM_XY)**2 + \
                    min_delta(e.y_pc, y_pc, PC_DIM_XY)**2 + \
                    min_delta(e.th_pc, th_pc, PC_DIM_TH)**2
                )
                delta_pcs.append(delta_pc)

                if delta_pc < EXP_DELTA_PC_THRESHOLD:
                    n_candidate_matches += 1

            if n_candidate_matches > 1:
                pass

            else:
                min_delta_id = np.argmin(delta_pcs)
                min_delta_val = delta_pcs[min_delta_id]

                if min_delta_val < EXP_DELTA_PC_THRESHOLD:
                    matched_exp = view_cell.exps[min_delta_id]

                    # see if the prev exp already has a link to the current exp
                    link_exists = False
                    for linked_exp in [l.target for l in self.current_exp.links]:
                        if linked_exp == matched_exp:
                            link_exists = True

                    if not link_exists:
                        self.current_exp.link_to(matched_exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing)

                if matched_exp is None:
                    matched_exp = self._create_exp(x_pc, y_pc, th_pc, view_cell)

                self.current_exp = matched_exp
                self.accum_delta_x = 0
                self.accum_delta_y = 0
                self.accum_delta_facing = self.current_exp.facing_rad

        self.history.append(self.current_exp)

        if not adjust_map:
            return


        # Iteratively update the experience map with the new information     
        for i in range(0, EXP_LOOPS):
            for e0 in self.exps:
                for l in e0.links:
                    # e0 is the experience under consideration
                    # e1 is an experience linked from e0
                    # l is the link object which contains additoinal heading
                    # info

                    e1 = l.target
                    
                    # correction factor
                    cf = EXP_CORRECTION
                    
                    # work out where exp0 thinks exp1 (x,y) should be based on 
                    # the stored link information
                    lx = e0.x_m + l.d * np.cos(e0.facing_rad + l.heading_rad)
                    ly = e0.y_m + l.d * np.sin(e0.facing_rad + l.heading_rad)

                    # correct e0 and e1 (x,y) by equal but opposite amounts
                    # a 0.5 correction parameter means that e0 and e1 will be 
                    # fully corrected based on e0's link information
                    e0.x_m = e0.x_m + (e1.x_m - lx) * cf
                    e0.y_m = e0.y_m + (e1.y_m - ly) * cf
                    e1.x_m = e1.x_m - (e1.x_m - lx) * cf
                    e1.y_m = e1.y_m - (e1.y_m - ly) * cf

                    # determine the angle between where e0 thinks e1's facing
                    # should be based on the link information
                    df = signed_delta_rad(e0.facing_rad + l.facing_rad, e1.facing_rad)

                    # correct e0 and e1 facing by equal but opposite amounts
                    # a 0.5 correction parameter means that e0 and e1 will be 
                    # fully corrected based on e0's link information           
                    e0.facing_rad = clip_rad_180(e0.facing_rad + df * cf)
                    e1.facing_rad = clip_rad_180(e1.facing_rad - df * cf)
    
        return
