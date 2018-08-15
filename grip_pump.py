"""
Prototype attentional focus grip task.

Cognition, Control, and Action Lab - Dr. Taraz Lee
Tyler A and Sean A

Behavior:
    The task asks the patient to squeeze two grip sensors such that each squeeze
    increases the L (left) and R (right) force indicators for the trial.
    The goal is to update L and R in the allotted time so that
    L^2 + R^2 = Z results in a target Z value.
Feedback:
    During the trial, the patient receives online visual feedback about the value
    of the left and right grip forces. After the trial, they receive feedback
    specifying the error as distance from target in window units.
Conditions:
    Internal vs. External focus instructions (varied by block)
    Target grip force Z value (varied by trial)
"""

# LIBRARIES
# math
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import math
from math import ceil
import numpy as np
from numpy.random import random, randint, normal, shuffle, uniform
import pandas as pd

# system
import os
import sys

# stimulus-response
import psychopy
from psychopy import visual, core, data, event
from psychopy.hardware import joystick


# GRIP SENSOR OBJECT
class Grips:
    """
    Manages joystick input via pyglet and provides interface for calculating
    user input value.
    """
    def __init__(self):
        """
        REQUIRES: visual.Window object is initialized
        MODIFIES: joystick.backend, nJoys, sensors, nAxes
        EFFECTS:  Initializes joystick using pyglet joystick backend. Sets
                  default max grip values.
        """
        joystick.backend = 'pyglet'
        
        self.nJoys = joystick.getNumJoysticks()
        
        JOYSTICK_ID  = 0 # not sure what this is for
        self.sensors = joystick.Joystick(JOYSTICK_ID)
        self.nAxes   = self.sensors.getNumAxes()
    
        # default max values
        self.left_max  = 1
        self.right_max = 1
    
        # Window units transformation factor
        self.WINDOW_TO_GRIPFORCE = 100
    
    def calibrate_left(self, max):
        # REQUIRES: 0 < max < 2
        # MODIFIES: left_max
        # EFFECTS:  updates subj's max left force
        
        self.left_max = max
        return
    
    def calibrate_right(self, max):
        # REQUIRES: 0 < max < 2
        # MODIFIES: right_max
        # EFFECTS:  updates subj's max right force
        
        self.right_max = max
        return

    def get_left(self):
        # EFFECTS: returns activation of left grip normalized by left-hand max
        #          and transposed into window units with preferred gradation
        # NOTE:    grip sensor input range is [-1, 1]
        
        raw = self.sensors.getY()
        force = ((raw + 1) / self.left_max) / self.WINDOW_TO_GRIPFORCE
        return force, raw

    def get_right(self):
        # EFFECTS: returns activation of right grip normalized by right-hand max
        #          and transposed into window units with preferred gradation
        # NOTE:    grip sensor input range is [-1, 1]
        
        raw = self.sensors.getX()
        force = ((raw + 1) / self.right_max) / self.WINDOW_TO_GRIPFORCE
        return force, raw

    #def get_score(self):
    #    """
    #    EFFECTS: Calculates subject response score using the equation
    #             specified in the docstring and subject's max grip strengths.
    #    """
    #
    #    # NOTE: - 1 at end is to transform score to psychopy window units
    #    score = (self.get_left()[0] ** 2) + (self.get_right()[0] ** 2) - 1
    #    return score


# STIMULI MANAGER OBJECT
class Stimuli:
    """
    Displays all stimuli, instructions, and feedback. Hosts all graphics
    necessary for those things.
    """
    
    def __init__(self):
        """
        Initializes psychopy window.
        """
        WINDOW_SIZE = 500 # pixels
        
        self.win = visual.Window(size=[WINDOW_SIZE, WINDOW_SIZE], fullscr=False,
                                 screen=0,
                                 allowGUI=True, allowStencil=False,
                                 monitor='testMonitor', color="black",
                                 colorSpace='rgb', blendMode='avg',
                                 useFBO=True, winType='pyglet')
    
    # GRAPHICS BUILDING
    def build_fixationpoint(self):
    
        self.fixation = visual.TextStim(self.win, pos=[0,0], text="+")
    
        return

    def build_trial_graphics(self):
        
        self.build_fixationpoint()
        
        #TRIAL_POLE_WIDTH  = 0.1
        #TRIAL_POLE_HEIGHT = 2  # the entire window
        #self.trial_pole = visual.Rect(self.win, width=TRIAL_POLE_WIDTH,
        #                              height=TRIAL_POLE_HEIGHT, pos=[0,0],
        #                              fillColor="grey", lineColor="grey")
                                      
        TRIAL_TARGET_WIDTH  = 0.75
        TRIAL_TARGET_HEIGHT = 0.3
        self.trial_target = visual.Rect(self.win, width=TRIAL_TARGET_WIDTH,
                                        height=TRIAL_TARGET_HEIGHT, pos=[0,0],
                                        fillColor="yellow", lineColor="black")
        
        # online grip feedback bars
        TRIAL_BARS_WIDTH = 0.1
        self.TRIAL_L_BAR_XPOS = -0.9
        self.TRIAL_R_BAR_XPOS = 0.9
        
        self.trial_L_bar = visual.Rect(self.win, width=TRIAL_BARS_WIDTH,
                                       height=0, pos=[self.TRIAL_L_BAR_XPOS, 0],
                                       fillColor="orange")
        self.trial_R_bar = visual.Rect(self.win, width=TRIAL_BARS_WIDTH,
                                       height=0, pos=[self.TRIAL_R_BAR_XPOS, 0],
                                       fillColor="orange")

        TRIAL_RESPONSE_WIDTH  = 0.5
        TRIAL_RESPONSE_HEIGHT = 0.2
        self.trial_response = visual.Rect(self.win, width=TRIAL_RESPONSE_WIDTH,
                                          height=TRIAL_RESPONSE_HEIGHT,
                                          pos=[0,-1], fillColor="blue")
                                        
        self.TRIAL_FEEDBACK_XPOS = 0.5
        
        self.trial_feedback_hit  = visual.TextStim(self.win, text="Hit!",
                                                   color="green",
                                                   pos=[self.TRIAL_FEEDBACK_XPOS,0])
        self.trial_feedback_miss = visual.TextStim(self.win, text="Miss!",
                                                   color="red",
                                                   pos=[self.TRIAL_FEEDBACK_XPOS,0])
                                                   
        self.TRIAL_POINTS_XPOS = -0.5
        
        self.trial_points = visual.TextStim(self.win, text="+",
                                            pos=[self.TRIAL_POINTS_XPOS, 0])
                                                   
        TRIAL_TIMER_XPOS = -0.75
        TRIAL_TIMER_YPOS = 0.75
        self.trial_timer_text = visual.TextStim(self.win, text="",
                                                pos=[TRIAL_TIMER_XPOS,
                                                     TRIAL_TIMER_YPOS])
        
        return
    
    def build_start_instructions(self):
        # Builds start-of-experiment instructions to window buffer.
        
        INSTRUCTIONS = "[INSERT INTRODUCTION INSTRUCTIONS HERE]"
        
        self.exp_instructions = visual.TextStim(self.win, text=INSTRUCTIONS)
        return
    
    def build_calibration_instructions(self):
        # Builds calibration instructions to the window buffer.
        INSTR_L = "CALIBRATION: Using your LEFT hand only, squeeze the LEFT\
                   grip as hard as you can."
        INSTR_R = "CALIBRATION: Using your RIGHT hand only, squeeze the RIGHT\
                   grip as hard as you can."
    
        self.calibrate_L_instr = visual.TextStim(self.win, text=INSTR_L)
        self.calibrate_R_instr = visual.TextStim(self.win, text=INSTR_R)

        # A rectangle to display online grip force input.
        CALIB_RECT_YPOS = -0.5
        self.CALIB_RECT_WIDTH = 0.1

        self.calibration_rectangle = visual.Rect(self.win, fillColor="orange",
                                                 pos=[0, CALIB_RECT_YPOS],
                                                 size=[self.CALIB_RECT_WIDTH,0])
        
        return
    
    def build_train_instructions(self):
        """
        MODIFIES: self
        EFFECTS:  Instantiates text for train phase instructions.
        """
        
        INSTRUCTIONS = "[INSERT TRAINING INSTRUCTIONS HERE]"
    
        self.train_instructions = visual.TextStim(self.win, text=INSTRUCTIONS)
    
        return
    
    def build_test_instructions_I(self):
        """
        MODIFIES: self
        EFFECTS:  Instantiates text for Internal test phase instructions.
        """
    
        INSTRUCTIONS = "[FOCUS ON YOUR HANDS TO PERFORM WELL ON THE TASK]"
    
        self.test_instructions_I = visual.TextStim(self.win, text=INSTRUCTIONS)
    
        return
    
    def build_test_instructions_E(self):
        """
        MODIFIES: self
        EFFECTS:  Instantiates text for External test phase instructions.
        """
    
        INSTRUCTIONS = "[FOCUS ON THE BARS/TARGET POSITION TO PERFORM WELL ON\
                        THE TASK]"
    
        self.test_instructions_E = visual.TextStim(self.win, text=INSTRUCTIONS)
    
        return
    
    def build_block_instructions(self):
        """
        MODIFIES: self
        EFFECTS:  Instantiates text for block instructions.
        NOTE: Text for these instructions is determined by run_[phase] methods
        """
        self.block_instructions = visual.TextStim(self.win, text="[INSERT BLOCK\
                                                                  INSTRUCTIONS]")
        
        return
    
    def build_point_total_feedback(self):
        """
        MODIFIES: self
        EFFECTS:  Instantiates text for point totals to be displayed at the end
                  of each block.
        """
        POINTS_TEXT_YPOS = 0.5 # window units
        
        self.point_total_text = visual.TextStim(self.win, text="[points!]",
                                                pos=[0, POINTS_TEXT_YPOS],
                                                color='green')
        
        return
    
    def build_focus_feedback(self):
        """
        MODIFIES: self
        EFFECTS:  Instantiates text for focus feedback, to be displayed after each
                  failed trial.
        """
        self.NUM_FOCUS_INSTRUCTIONS = 2
        
        self.INTERNAL_FEEDBACK = ["Try squeezing your hands better",
                                  "Really focus on your fingers!"]
        self.EXTERNAL_FEEDBACK = ["Pay attention to how far you are away from the target",
                                  "Eyes on the prize! Focus on the target"]
                                  
        self.focus_feedback = visual.TextStim(self.win)
    
        return
    
    # GRAPHICS DISPLAY INTERFACE
    def disp_start(self):
        # EFFECTS: Displays start-of-experiment instructions for five seconds.
        
        INSTR_READ_TIME = 5 # seconds
        
        self.build_start_instructions()
        
        self.exp_instructions.draw(win=self.win)
        self.win.flip()
        
        psychopy.clock.wait(INSTR_READ_TIME)
    
        return
    
    def disp_test_instructions(self, condition):
        """
        REQUIRES: self.test_instructions_I/E are built
        MODIFIES: self
        EFFECTS:  Displays the test phase block instructions for given condition
        """
    
        INSTR_READ_TIME = 3 # seconds
    
        if condition == 'I':
            self.test_instructions_I.draw()
        else:
            self.test_instructions_E.draw()

        self.win.flip()
        psychopy.clock.wait(INSTR_READ_TIME)
                
        return

    def disp_fixation(self, time_interval):
        """
        REQUIRES: self.fixation is built, time_interval is float or int
        MODIFIES: self
        EFFECTS:  displays the fixation dot for the specified interval.
        """
        
        self.fixation.draw()
        self.win.flip()
        psychopy.clock.wait(time_interval)
        
        return

    def disp_trial(self, target_ypos, online=True):
        """
        REQUIRES: -1 < target_ypos < 1, trial graphics built
        MODIFIES: self
        EFFECTS:  displays graphics for the trial (inviting subject response)
        """

        # online grip feedback and counter
        if online:
            self.trial_L_bar.setAutoDraw(True)
            self.trial_R_bar.setAutoDraw(True)
        
        self.trial_timer_text.setAutoDraw(True)
        
        # target
        #self.trial_pole.setAutoDraw(True)
        self.trial_target.pos = [0, target_ypos]
        self.trial_target.setAutoDraw(True)

        return

    def hide_trial_graphics(self):
        """
        REQUIRES: trial graphics built
        MODIFIES: self
        EFFECTS:  halts drawing of trial graphics (removes them from the window)
        """

        #self.trial_pole.setAutoDraw(False)
        self.trial_L_bar.setAutoDraw(False)
        self.trial_R_bar.setAutoDraw(False)
        self.trial_timer_text.setAutoDraw(False)
        self.trial_response.setAutoDraw(False)
        self.trial_target.setAutoDraw(False)

        return

    def disp_trial_feedback(self, points, correct, trial_response_ypos, time_interval):
        """
        REQUIRES: correct is a bool, -1 < trial_response_ypos < 1,
                  time_interval > 0
        MODIFIES: self
        EFFECTS:  Displays "hit" or "miss" based on correctness of trial response
        """

        # Fixme: display points for trial

        if correct:
            # display Hit!
            self.trial_feedback_hit.pos = [self.TRIAL_FEEDBACK_XPOS,
                                           trial_response_ypos]
            self.trial_feedback_hit.draw()
        else:
            # display Miss!
            self.trial_feedback_miss.pos = [self.TRIAL_FEEDBACK_XPOS,
                                            trial_response_ypos]
            self.trial_feedback_miss.draw()

        
        # display points for trial (no points during training)
        if points != 0:
            points = '+' + str(points)
            self.trial_points.text = points
            self.trial_points.pos = [self.TRIAL_POINTS_XPOS, trial_response_ypos]
            self.trial_points.draw()
        
        self.win.flip()
        psychopy.clock.wait(time_interval)

        return
    
    def disp_focus_feedback(self, focus, instr_index, time_interval):
        """
        REQUIRES: 0 <= instr_index < len(self.INTERNAL_FEEDBACK)
                  focus == 'I' or 'E'
        MODIFIES: self
        EFFECTS:  Displays focus instructions-type feedback to subj.
        NOTE:     Up to Experiment to counterbalance which instructions are 
                  displayed, and when focus instructions are given.
        """

        if focus == 'I':
            self.focus_feedback.text = self.INTERNAL_FEEDBACK[instr_index]
        else:
            self.focus_feedback.text = self.EXTERNAL_FEEDBACK[instr_index]
        
        self.focus_feedback.draw()
        psychopy.clock.wait(time_interval)
        
        return

    def disp_block_feedback(self, points_cum, time_interval):
        """
        MODIFIES: self
        EFFECTS:  Displays total points so far.
        """

        text = "You gained %s points this block!" % points_cum
        self.point_total_text.text = text
        self.point_total_text.draw()
        self.win.flip()
        psychopy.clock.wait(time_interval)

        return


# EXPERIMENT OBJECT
class Experiment:
    """
    Manages subject responses, stimuli, and trial organization.
    """
    def __init__(self, subj_num, session_name, data_dir):
        """
        MODIFIES: self...
        EFFECTS:  initializes stimuli object (and window), Grips object, ...
        """
        
        # data collection
        self.subj_id = subj_num
        SEPARATOR = '/'
        self.SESSION_FILE = data_dir + SEPARATOR + 'data/%s_%s_output.tsv'\
                            % (session_name, subj_num)
        self.TEMP_FILE = data_dir + SEPARATOR + 'temp_outputs/%s_%s_temp_output.tsv'\
                         % (session_name, subj_num)
        
        self.stimuli = Stimuli() # opens psychopy window
        self.grips = Grips()
        self.init_data_lists()
    
    def init_data_lists(self):
        # EFFECTS: creates lists to hold trial data values (until they are
        #          packaged in a pd.DataFrame)
        # NOTE: subj_id, gripLmax, and gripRmax remain constant during experiment
        
        self.trial_nums    = []
        self.focuses       = []
        self.phases        = []
        self.blocks        = []
        self.accuracies    = []
        self.grip_scores   = []
        self.targets       = []
        self.gripLraws     = []
        self.gripRraws     = []
        self.gripLnormds   = []
        self.gripRnormds   = []
        self.points_gained = []
    
        return
    
    def calibrate_grip_L(self, counter_text, instr_time, cali_time):
        """
        REQUIRES: counter_text is visual.TextStim, calibration stimuli are built
        MODIFIES: stimuli, grips
        EFFECTS:  prompts subject to follow calibration procedure using stimuli
                  and updates L grip accordingly. Uses provided instruction
                  read time and calibration time intervals.
        """
        
        # show L instructions
        self.stimuli.calibrate_L_instr.draw()
        self.stimuli.win.flip()
        psychopy.clock.wait(instr_time)
    
        # prepare timer and graphics
        timer = psychopy.clock.CountdownTimer(cali_time)
        self.stimuli.calibration_rectangle.setAutoDraw(True)
        counter_text.setAutoDraw(True)
    
        # solicit max grip force L
        while timer.getTime() > 0:
            max_left = self.grips.sensors.getY() + 1
                
            # fixme: debug
            print(max_left)
            
            # display online rectangle for input
            # fixme: Debug
            #rect_height = (0 if max_left < 0 else max_left - 1)
            self.stimuli.calibration_rectangle.size = [self.stimuli.CALIB_RECT_WIDTH,
                                                       max_left]
            counter_text.text = str(int(timer.getTime()) + 1)
            
            # refresh window
            self.stimuli.win.flip()
            
        self.grips.calibrate_left(max_left)
        
        counter_text.setAutoDraw(False)
        self.stimuli.calibration_rectangle.setAutoDraw(False)

        #FIXME: debug
        print("left max: " + str(self.grips.left_max))
            
        return self.grips.left_max
    
    def calibrate_grip_R(self, counter_text, instr_time, cali_time):
        """
        REQUIRES: counter_text is visual.TextStim, calibration stimuli are built
        MODIFIES: stimuli, grips
        EFFECTS:  prompts subject to follow calibration procedure using stimuli
                  and updates R grip accordingly. Uses provided instruction
                  read time and calibration time intervals.
        """
    
        # show R instructions
        self.stimuli.calibrate_R_instr.draw()
        self.stimuli.win.flip()
        psychopy.clock.wait(instr_time)
    
        # prepare timer and graphics
        timer = psychopy.clock.CountdownTimer(cali_time)
        self.stimuli.calibration_rectangle.setAutoDraw(True)
        counter_text.setAutoDraw(True)
    
        # solicit max grip force R
        while timer.getTime() > 0:
            max_right = self.grips.sensors.getX() + 1
            
            # fixme: debug
            print(max_right)
            
            # display online rectangle for input
            # fixme: debug
            #rect_height = (0 if max_right < 0 else max_right - 1)
            self.stimuli.calibration_rectangle.size = [self.stimuli.CALIB_RECT_WIDTH,
                                                       max_right]
            counter_text.text = str(int(timer.getTime()) + 1)
            
            # refresh window
            self.stimuli.win.flip()
        
        self.grips.calibrate_right(max_right)
        
        counter_text.setAutoDraw(False)
        self.stimuli.calibration_rectangle.setAutoDraw(False)

        #FIXME: debug
        print("right max: " + str(self.grips.right_max))
            
        return self.grips.right_max
    
    def calibrate_grips(self):
        """
        MODIFIES: stimuli, grips
        EFFECTS:  prompts subject to follow calibration procedure and updates
                  grips accordingly
        """
        
        CALIBRATION_INTERVAL = 3 # seconds
        INSTR_READ_TIME      = 3
        # text to show countdown timer
        countdown_text = visual.TextStim(self.stimuli.win, text="")

        self.stimuli.build_calibration_instructions()
        
        self.calibrate_grip_L(countdown_text, INSTR_READ_TIME, CALIBRATION_INTERVAL)
        self.calibrate_grip_R(countdown_text, INSTR_READ_TIME, CALIBRATION_INTERVAL)
            
        return

    def calc_force_score(self, Lforce, Rforce):
        """
        REQUIRES: Lforce, Rforce are numbers
        EFFECTS:  Returns force score for given bimanual force inputs using
                  equation specified in docstring.
        """
            
        # NOTE: - 1 at end is to transform score to psychopy window units
        score = (Lforce ** 2) + (Rforce ** 2) - 1
        return score

    def get_subj_response(self, countdown_timer):
        """
        REQUIRES: countdown_timer is psychopy.clock.CountdownTimer
                  Trial graphics are built (and showing)
        MODIFIES: stimuli, grips
        EFFECTS:  Updates trial graphics as subj responds using grip sensors.
                  Returns calculated response score of trial.
        """
        
        # initialize force totals
        Lforce_total = 0
        Rforce_total = 0
        Lraw_total   = 0
        Rraw_total   = 0

        while countdown_timer.getTime() > 0:
            timer_text = str(int(countdown_timer.getTime()) + 1)
            self.stimuli.trial_timer_text.text = timer_text
            
            # get grip sensor data
            Lforce, Lraw = self.grips.get_left()
            Rforce, Rraw = self.grips.get_right()
            
            #fixme: record online bimanual grip activation path for each trial?
            
            # calculate force sums
            Lforce_total += Lforce
            Lraw_total   += Lraw
            Rforce_total += Rforce
            Rraw_total   += Rraw
            
            #fixme: Debug
            print "left RAW: %s, right RAW: %s" % (self.grips.sensors.getY(),
                                                   self.grips.sensors.getX())
            print "left input: %s, right input: %s" % (Lforce, Rforce)

            # update left grip online feedback bar
            self.stimuli.trial_L_bar.height = Lforce_total
            self.stimuli.trial_L_bar.pos = [self.stimuli.TRIAL_L_BAR_XPOS,
                                            (-1 + Lforce_total / 2)]
            # update right "
            self.stimuli.trial_R_bar.height = Rforce_total
            self.stimuli.trial_R_bar.pos = [self.stimuli.TRIAL_R_BAR_XPOS,
                                            (-1 + Rforce_total / 2)]
                                            
            trial_response = self.calc_force_score(Lforce_total, Rforce_total)
            self.stimuli.win.flip()
        
            self.check_escaped()

        return trial_response, Lforce_total, Lraw_total, Rforce_total, Rraw_total
    
    def log_trial(self, phase, block, trial_num, focus, target, correct, 
                  gripLraw, gripLnormd, gripRraw, gripRnormd, grip_score, points):
        """
        REQUIRES: 
        MODIFIES: self, pwd/temp_data
        EFFECTS:  Adds all relevant data of trial to subj_data. Writes all data
                  so far to a temp data file in case program crashes.
        """
    
        # fill in trial condition data
        self.trial_nums.append(trial_num)
        self.focuses.append(focus)
        self.phases.append(phase)
        self.blocks.append(block)
        self.targets.append(target)

        # fill in trial response data
        self.grip_scores.append(grip_score)
        self.gripLraws.append(gripLraw)
        self.gripRraws.append(gripRraw)
        self.gripLnormds.append(gripLnormd)
        self.gripRnormds.append(gripRnormd)
        self.accuracies.append(correct)
        self.points_gained.append(points)

        # create temp file in case of crash
        temp_data = {'subject': [self.subj_id] * len(self.grip_scores),
                     'trial_num': self.trial_nums,
                     'focus': self.focuses,
                     'phase': self.phases,
                     'block': self.blocks,
                     'accuracy': self.accuracies,
                     'grip_score': self.grip_scores,
                     'points': self.points_gained,
                     'target': self.targets,
                     'gripLraw': self.gripLraws,
                     'gripRraw': self.gripRraws,
                     'gripLnormd': self.gripLnormds,
                     'gripRnormd': self.gripRnormds,
                     'gripLmax': [self.grips.left_max] * len(self.grip_scores),
                     'gripRmax': [self.grips.right_max] * len(self.grip_scores)}

        temp_DF = pd.DataFrame(temp_data)
        temp_DF.to_csv(self.TEMP_FILE, sep='\t')
    
        return

    def run_trial(self, phase, block, trial_num, focus, target_ypos):
        """
        REQUIRES: -1 < target_ypos < 1, focus is either 'I' or 'E'
        MODIFIES: self (self.stimuli, self.grips, etc.)
        EFFECTS:  Runs a single trial and records trial data. 
                  FIXME: support for focus conditions
        """
        
        FIXATION_LIMIT = 2 # seconds
        TRIAL_LIMIT    = 3
        FEEDBACK_LIMIT = 3
        
        ACCURACY_THRESHOLD = 0.1 # "transformed grip space" (window) units
        
        # FIXATION AND TRIAL GRAPHICS
        self.stimuli.disp_fixation(FIXATION_LIMIT)
        self.stimuli.disp_trial(target_ypos, online=(phase == 'train'))
        
        timer = psychopy.clock.CountdownTimer(TRIAL_LIMIT)
        
        # SUBJ RESPONSE
        trial_response, Lforce, Lraw, Rforce, Rraw = self.get_subj_response(timer)
    
        # DATA LOGGING
        # compute accuracy
        distance = abs(trial_response - target_ypos)
        accurate = (1 if distance <= ACCURACY_THRESHOLD else 0)
        
        # update subj point total, no negative points
        if phase == 'test' and (1 - distance) > 0:
            points = int((1 - distance) * 100)
            self.subj_points += points
        else:
            points = 0
        
        # log data (publish temp results file)
        self.log_trial(phase, block, trial_num, focus, target_ypos, accurate,
                       Lraw, Lforce, Rraw, Rforce, trial_response, points)

        #fixme: debug
        print('\n' + "trial response: " + str(trial_response))
        print("target: " + str(target_ypos))
        print("accuracy: " + str(accurate))

        # FEEDBACK
        self.stimuli.trial_timer_text.setAutoDraw(False)
        self.stimuli.trial_response.setAutoDraw(True)
        self.stimuli.trial_response.pos = [0, trial_response]
        
        self.stimuli.disp_trial_feedback(points, accurate, trial_response,
                                         FEEDBACK_LIMIT)
        self.stimuli.hide_trial_graphics()
    
        return distance

    def run_block(self, phase, block, focus):
        """
        REQUIRES: phase is 'train' or 'test', block > 0, focus is 'I' or 'E'
                  block_instructions is built
        MODIFIES: self (basically... all of it.)
        EFFECTS:  Runs a single block of trials with phase constraints with
                  randomized target condition order.
        """
        BLOCK_INTRO_TIME = 2 # seconds
        END_BLOCK_FEEDBACK_TIME = 3 # seconds
        
        # fixme: How many trials per block?
        TRIAL_TARGETS = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        NUM_TARGET_REPEATS = 1
        
        self.stimuli.build_trial_graphics()
        self.stimuli.build_point_total_feedback()
        
        # create randomized conditions list
        trial_targets_order = data.TrialHandler(TRIAL_TARGETS,
                                                NUM_TARGET_REPEATS,
                                                method='random')
        
        self.stimuli.block_instructions.text = "Block #" + str(block)
        self.stimuli.block_instructions.draw()
        self.stimuli.win.flip()
        psychopy.clock.wait(BLOCK_INTRO_TIME)
        
        trial_counter = 1
        for target in trial_targets_order:
            self.run_trial(phase, block, trial_counter, focus, target)
            # display focus-specific advice every 2 trials
            if trial_counter % 2 == 0 and phase == 'test':
                instruction = trial_counter % self.stimuli.NUM_FOCUS_INSTRUCTIONS
                self.stimuli.disp_focus_feedback(focus, instruction,
                                                 END_BLOCK_FEEDBACK_TIME)
            trial_counter += 1
        
        # display points at end of block in test phase
        if phase == 'test':
            total = 0
            for trial_points in self.points_gained:
                total += trial_points
            self.stimuli.disp_block_feedback(total, END_BLOCK_FEEDBACK_TIME)
        
        return
    
    def run_training(self):
        """
        REQUIRES: self.grips are calibrated
        MODIFIES: self (basically... all of it.)
        EFFECTS:  Runs the training section of the experiment, consisting of
                  1 block of 14 trials.
        """
    
        INSTR_READ_TIME = 3 # seconds
        
        # show training instructions
        self.stimuli.build_train_instructions()
        self.stimuli.train_instructions.draw()
        self.stimuli.win.flip()
        psychopy.clock.wait(INSTR_READ_TIME)
        
        self.stimuli.build_block_instructions()
    
        self.run_block('train', 1, 'E') #fixme: focus condition
    
        return
    
    def run_test(self):
        """
        REQUIRES: self.grips are calibrated
        MODIFIES: self (basically... all of it.)
        EFFECTS:  Runs the test section of the experiment, currently consisting
                  of 1 Internal block and 1 External block
        """
        CONDITIONS = ['I', 'E']
        INSTR_READ_TIME = 3 # seconds
        
        self.subj_points = 0
    
        # show test instructions
        self.stimuli.build_test_instructions_E()
        self.stimuli.build_test_instructions_I()
        self.stimuli.build_focus_feedback()
    
        condition_order = data.TrialHandler(CONDITIONS, 1, method='random')
    
        current_block = 1
        for focus in condition_order:
            self.stimuli.disp_test_instructions(focus)
            self.run_block('test', current_block, focus)
            current_block += 1
            
        return
    
    def check_escaped(self):
        # Checks if participant pressed escape. Quits the experiment (while
        # ensuring all data has been logged) if yes. FIXME: implement
        if event.getKeys(keyList=["escape"]):
            self.stimuli.win.close()
            core.quit()
            
        return 1
        
    def write_data(self):
        """
        REQUIRES: more than one trial has occurred successfully
        MODIFIES: grip/data/
        EFFECTS:  writes all data from experiment to tsv.
        """
        
        # create dataframe
        data = {'subject': [self.subj_id] * len(self.grip_scores),
                'trial_num': self.trial_nums,
                'focus': self.focuses,
                'phase': self.phases,
                'block': self.blocks,
                'accuracy': self.accuracies,
                'grip_score': self.grip_scores,
                'points': self.points_gained,
                'target': self.targets,
                'gripLraw': self.gripLraws,
                'gripRraw': self.gripRraws,
                'gripLnormd': self.gripLnormds,
                'gripRnormd': self.gripRnormds,
                'gripLmax': [self.grips.left_max] * len(self.grip_scores),
                'gripRmax': [self.grips.right_max] * len(self.grip_scores)}
                
        subj_data = pd.DataFrame(data)
        subj_data.to_csv(self.SESSION_FILE, sep='\t')
        
        return

    def start(self):
        """
        MODIFIES: stimuli
        EFFECTS:  presents start-of-experiment instructions to the window
                  (includes time delay for reading). Coordinates grip sensor
                  calibration.
        """

        self.stimuli.disp_start()

        self.calibrate_grips()
        
        self.stimuli.build_block_instructions()
        self.run_training()
        self.run_test()
        
        return

    def end(self):
        """
        MODIFIES: stimuli, ...
        EFFECTS:  Ends experiment peacefully by ensuring all data is logged,
                  closing the window, and closing psychopy application.
        """

        self.write_data()
        self.stimuli.win.close()
        core.quit()

        return


def main():
                       
    # PATH INFORMATION
    # FIXME: need correct working directory!
    DATA_DIR = "/Users/seanpaul/Desktop/Box Sync/grip"

    # SUBJECT INFORMATION
    SESSION_NAME = 'bimanual_grip_pump'
    SUB_ID = raw_input("Please enter the participant's ID number: ")

    # run experiment
    focus_task = Experiment(SUB_ID, SESSION_NAME, DATA_DIR)

    focus_task.start()
    focus_task.end()

    return 0

# run the program
if __name__ == '__main__':
    main()
