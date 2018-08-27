"""
Prototype attentional focus grip task.

Cognition, Control, and Action Lab - Dr. Taraz Lee
Tyler A and Sean A

Created using the PsychoPy Python library (1.90.2):

    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi:10.3389/neuro.11.010.2008

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
import math
from math import ceil
import numpy as np
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
        #self.left_max  = 1
        #self.right_max = 1
    
        # Window units transformation factor
        #self.WINDOW_TO_GRIPFORCE = 100
    
    def set_left_max(self, max):
        """
        REQUIRES: 0 < max < 2
        MODIFIES: left_max
        EFFECTS:  updates subj's max left force
        """
        
        self.left_max = max
        return
    
    def set_right_max(self, max):
        """
        REQUIRES: 0 < max < 2
        MODIFIES: right_max
        EFFECTS:  updates subj's max right force
        """
        
        self.right_max = max
        return

    def get_left(self):
        """
        EFFECTS: Returns activation of left grip
        NOTE:    grip sensor input range is [-1, 1]
        """
        
        return self.sensors.getY() + 1

    def get_right(self):
        """
        EFFECTS: returns activation of right grip
        NOTE:    grip sensor input range is [-1, 1]
        """
        return self.sensors.getX() + 1

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
        WINDOW_SIZE = 700 # pixels
        
        self.win = visual.Window(size=[WINDOW_SIZE, WINDOW_SIZE], fullscr=False,
                                 screen=0, checkTiming=True,
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
                                      
        TRIAL_TARGET_FILENAME = "target.png"
        TRIAL_TARGET_SIZE = 0.4
        self.trial_target = visual.ImageStim(self.win, image=TRIAL_TARGET_FILENAME,
                                             pos=[0,0], size=TRIAL_TARGET_SIZE)
        
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

        TRIAL_RESPONSE_RADIUS = 0.05
        self.trial_response = visual.Circle(self.win, radius=TRIAL_RESPONSE_RADIUS,
                                            pos=[0,-1], fillColor="blue")
                                        
        self.TRIAL_FEEDBACK_XPOS = 0.6
        
        self.trial_feedback_hit  = visual.TextStim(self.win, text="Hit!",
                                                   color="green",
                                                   pos=[self.TRIAL_FEEDBACK_XPOS,0])
        self.trial_feedback_miss = visual.TextStim(self.win, text="Miss!",
                                                   color="red",
                                                   pos=[self.TRIAL_FEEDBACK_XPOS,0])
        self.trial_feedback_outside = visual.TextStim(self.win, text="Out of bounds!",
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
                                                color='yellow')
        
        return
    
    def build_focus_feedback(self):
        """
        MODIFIES: self
        EFFECTS:  Instantiates text for focus feedback, to be displayed after each
                  failed trial.
        NOTE:     AFTER DATA COLLECTION STARTED, do not mess with order of these
                  instructions in their lists! Needed for data analysis.
                  
        """
        
        self.INTERNAL_FEEDBACK = ["Try squeezing your hands better",
                                  "Really focus on your fingers!",
                                  "Pay attention to your hand/arm muscles",
                                  "Control your grip strength effectively"]
        self.EXTERNAL_FEEDBACK = ["Pay attention to how far you are away from the target",
                                  "Eyes on the prize! Focus on the target",
                                  "Note the distance between the target and your\
                                   response.",
                                  "Attend to the height of the target"]
                                  
        # check if number of instructions are same for each focus condition
        WARNING_MSG = "WARNING! There isn't an equal number of internal vs external\
                       focus instructions. Please correct this in\
                       Stimuli.build_focus_feedback!"
        assert len(self.INTERNAL_FEEDBACK) == len(self.EXTERNAL_FEEDBACK), WARNING_MSG
        
        self.NUM_FOCUS_INSTRUCTIONS = len(self.INTERNAL_FEEDBACK)
                                  
        self.focus_feedback = visual.TextStim(self.win)
    
        return
    
    # GRAPHICS DISPLAY INTERFACE
    def disp_start(self):
        """
        REQUIRES: self.exp_instructions are built
        MODIFIES: self
        EFFECTS: Displays start-of-experiment instructions
        """
        
        self.build_start_instructions()
        
        self.exp_instructions.draw(win=self.win)
        self.win.flip()
    
        return
    
    def disp_test_instructions(self, condition):
        """
        REQUIRES: self.test_instructions_I/E are built
        MODIFIES: self
        EFFECTS:  Displays the test phase block instructions for given condition
        """
    
        if condition == 'I':
            self.test_instructions_I.draw()
        else:
            self.test_instructions_E.draw()

        self.win.flip()
                
        return

    def disp_block_instructions(self, block_num):
        """
        REQUIRES: block_instructions are built
        MODIFIES: self
        EFFECTS:  Displays the block # to the screen.
        """
    
        self.block_instructions.text = "Block #" + str(block_num)
        self.block_instructions.draw()
        self.win.flip()
    
        return

    def disp_fixation(self):
        """
        REQUIRES: self.fixation is built, time_interval is float or int
        MODIFIES: self
        EFFECTS:  displays the fixation dot for the specified interval.
        """
        
        self.fixation.draw()
        self.win.flip()
        
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
            
    def disp_trial_response_animation(self, trial_response_ypos, time_interval):
        """
        REQUIRES: -1 < target_ypos < 1, 0 < time_interval, self.trial_response
                  is built and .autoDraw enabled!
        MODIFIES: self
        EFFECTS:  Animates the response circle sliding across the screen to
                  response_ypos location.
        NOTE:     'escape' keypress NOT enabled during this animation.
        """
    
        self.trial_response.setAutoDraw(True)
        
        # calculate smoothest window step interval
        y_increment = (self.win.monitorFramePeriod / time_interval) \
                       * (trial_response_ypos + 1)
        
        if trial_response_ypos == -1:
            # score is zero; don't move target
            self.trial_response.pos = [0, -1]
            self.win.flip()
            return
        
        # animate subject response
        response_circle_ypos = -1 # bottom of screen
        while response_circle_ypos < trial_response_ypos:
            self.trial_response.pos = [0, response_circle_ypos]
            response_circle_ypos += y_increment
            self.win.flip()
        
        # make sure response graphic ends up exactly where it's supposed to be
        self.trial_response.pos = [0, trial_response_ypos]
        self.win.flip()

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

    def disp_trial_hitmiss(self, correct, trial_response_ypos):
        """
        REQUIRES: correct is a bool, -1 < trial_response_ypos < 1
        MODIFIES: self
        EFFECTS:  Displays "hit" or "miss" based on correctness of trial response
        """

        if correct:
            # display Hit!
            self.trial_feedback_hit.pos = [self.TRIAL_FEEDBACK_XPOS,
                                           trial_response_ypos]
            self.trial_feedback_hit.draw()
        elif trial_response_ypos > 1 or trial_response_ypos == -1:
            # above top of window or didn't meet grip threshold
            self.trial_feedback_outside.draw()
        else:
            # display Miss!
            self.trial_feedback_miss.pos = [self.TRIAL_FEEDBACK_XPOS,
                                            trial_response_ypos]
            self.trial_feedback_miss.draw()
        
        self.win.flip()

        return

    def disp_trial_points(self, points, trial_response_ypos):
        """
        REQUIRES: -1 < trial_response_ypos < 1, points >= 0
        MODIFIES: self
        EFFECTS:  Displays points earned in this trial
        """
        
        # display out of bounds
        if trial_response_ypos > 1 or trial_response_ypos == -1:
            # above top of window or didn't meet grip threshold
            self.trial_feedback_outside.draw()
        
        # display points for trial (no points during training)
        if points != 0:
            points = '+' + str(points)
            self.trial_points.text = points
            self.trial_points.pos = [self.TRIAL_POINTS_XPOS, trial_response_ypos]
            self.trial_points.draw()
        
        self.win.flip()
    
        return
    
    def disp_focus_feedback(self, focus, instr_index):
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
        self.win.flip()
        
        return

    def disp_block_feedback(self, points_cum):
        """
        REQUIRES: point_total_text is built
        MODIFIES: self
        EFFECTS:  Displays total points so far.
        """

        text = "You gained %s points this block!" % points_cum
        self.point_total_text.text = text
        self.point_total_text.draw()
        self.win.flip()

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
        # files for online grip force (no temp file, overwrite each trial)
        online_filename_left = 'data/online/%s_%s_online_left.tsv'\
                                % (session_name, subj_num)
        online_filename_right = 'data/online/%s_%s_online_right.tsv'\
                                % (session_name, subj_num)
        self.ONLINE_FILE_LEFT = data_dir + SEPARATOR + online_filename_left
        self.ONLINE_FILE_RIGHT = data_dir + SEPARATOR + online_filename_right
        
        self.stimuli = Stimuli() # opens psychopy window
        self.grips = Grips()
        self.init_data_lists()
    
    def init_data_lists(self):
        """
        EFFECTS: creates lists to hold trial data values (until they are
                 packaged in a pd.DataFrame)
        NOTE:    subj_id, gripLmax, gripRmax, and beta factor remain constant
                 during experiment
        """
        
        # standard trial data
        self.trial_nums    = []
        self.focuses       = []
        self.instrs        = []
        self.phases        = []
        self.blocks        = []
        self.accuracies    = []
        self.out_of_bounds = []
        self.grip_scores   = []
        self.targets       = []
        self.gripLtotals   = []
        self.gripRtotals   = []
        self.raw_scores    = []
        self.points_gained = []
        
        # online grip force data
        self.gripLonline = {}
        self.gripRonline = {}
        # keep track of which trial we're measuring (see get_subj_response)
        self.trial_iterator = 1
    
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
        self.wait(instr_time)
    
        # prepare timer and graphics
        timer = psychopy.clock.CountdownTimer(cali_time)
        self.stimuli.calibration_rectangle.setAutoDraw(True)
        counter_text.setAutoDraw(True)
    
        # solicit total grip force L over calibration time
        left_total = 0
        while timer.getTime() > 0:
            left_total += self.grips.get_left()
                
            # fixme: debug
            print(left_total)
            
            # display online rectangle for input
            self.stimuli.calibration_rectangle.size = [self.stimuli.CALIB_RECT_WIDTH,
                                                       self.grips.get_left()]
            counter_text.text = str(int(timer.getTime()) + 1)
            
            # refresh window
            self.stimuli.win.flip()
            self.check_escaped()
            
        self.grips.set_left_max(left_total)
        
        counter_text.setAutoDraw(False)
        self.stimuli.calibration_rectangle.setAutoDraw(False)

        #FIXME: debug
        print("left total max: " + str(self.grips.left_max))
            
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
        self.wait(instr_time)
    
        # prepare timer and graphics
        timer = psychopy.clock.CountdownTimer(cali_time)
        self.stimuli.calibration_rectangle.setAutoDraw(True)
        counter_text.setAutoDraw(True)
    
        # solicit max grip force R
        right_total = 0
        while timer.getTime() > 0:
            right_total += self.grips.get_right()
            
            # fixme: debug
            print(right_total)
            
            # display online rectangle for input
            self.stimuli.calibration_rectangle.size = [self.stimuli.CALIB_RECT_WIDTH,
                                                       self.grips.get_right()]
            counter_text.text = str(int(timer.getTime()) + 1)
            
            # refresh window
            self.stimuli.win.flip()
            self.check_escaped()
        
        self.grips.set_right_max(right_total)
        
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
        
        CALIBRATION_INTERVAL = 3 # MUST BE SAME AS TRIAL_LIMIT
        INSTR_READ_TIME      = 3
        # text to show countdown timer
        countdown_text = visual.TextStim(self.stimuli.win, text="")

        self.stimuli.build_calibration_instructions()
        
        self.calibrate_grip_L(countdown_text, INSTR_READ_TIME, CALIBRATION_INTERVAL)
        self.calibrate_grip_R(countdown_text, INSTR_READ_TIME, CALIBRATION_INTERVAL)
        
        # calculate beta normalization factor
        # 2 (top of window) / 0.5 (% of max grip in one hand) = 4
        self.beta = 4 / min(self.grips.right_max, self.grips.left_max)
            
        return

    def calc_force_score(self, Ltotal, Rtotal):
        """
        REQUIRES: Lforce, Rforce are numbers
        EFFECTS:  Returns force score in window units!
                  for given bimanual force inputs using
                  equation specified in docstring.
        """
        THRESHOLD_R = self.grips.right_max * 0.01
        THRESHOLD_L = self.grips.left_max * 0.01
        
        force_sum = Ltotal + Rtotal
        
        # NOTE: -1 to transform score to psychopy window units
        if Ltotal <= THRESHOLD_L or Rtotal <= THRESHOLD_R:
            score = -1 # bottom of screen
        else:
            score = -1 + (Ltotal + Rtotal) * self.beta

        return score, force_sum

    def get_subj_response(self, countdown_timer):
        """
        REQUIRES: countdown_timer is psychopy.clock.CountdownTimer
                  Trial graphics are built (and showing)
        MODIFIES: stimuli, grips
        EFFECTS:  Updates trial graphics as subj responds using grip sensors.
                  Returns calculated response score of trial.
        """
        
        # initialize force totals and online lists
        Lforce_total = 0
        Rforce_total = 0
        self.gripLonline[(self.trial_iterator, 'force')] = []
        self.gripRonline[(self.trial_iterator, 'force')] = []
        
        # start recording frame intervals
        self.stimuli.win.recordFrameIntervals = True
        frame_ints_len = len(self.stimuli.win.frameIntervals)

        while countdown_timer.getTime() > 0:
            timer_text = str(int(countdown_timer.getTime()) + 1)
            self.stimuli.trial_timer_text.text = timer_text
            
            # get grip sensor data
            Lforce = self.grips.get_left()
            Rforce = self.grips.get_right()
            
            #record online bimanual grip activation path for this trial frame
            self.gripLonline[(self.trial_iterator, 'force')].append(Lforce)
            self.gripRonline[(self.trial_iterator, 'force')].append(Rforce)
            
            # calculate force sums
            Lforce_total += Lforce
            Rforce_total += Rforce
            
            #fixme: Debug
            print "left input: %s, right input: %s" % (Lforce, Rforce)

            # update left grip online feedback bar
            Lheight = Lforce_total / self.grips.left_max
            self.stimuli.trial_L_bar.height = Lheight
            self.stimuli.trial_L_bar.pos = [self.stimuli.TRIAL_L_BAR_XPOS,
                                            (-1 + Lheight / 2)]
            # update right "
            Rheight = Rforce_total / self.grips.right_max
            self.stimuli.trial_R_bar.height = Rheight
            self.stimuli.trial_R_bar.pos = [self.stimuli.TRIAL_R_BAR_XPOS,
                                            (-1 + Rheight / 2)]

            self.stimuli.win.flip()
            self.check_escaped()
        
        # note frame intervals for trial and stop recording
        self.stimuli.win.recordFrameIntervals = False
        self.frame_ints = self.stimuli.win.frameIntervals[frame_ints_len:]
        # calculate time stamps and add to online data
        self.frame_ints = list(np.cumsum(self.frame_ints))
        self.gripLonline[(self.trial_iterator, 'time')] = self.frame_ints
        self.gripRonline[(self.trial_iterator, 'time')] = self.frame_ints
        
        trial_response, raw_response = self.calc_force_score(Lforce_total,
                                                             Rforce_total)
        
        # increment trial iterator (absolute trial index)
        self.trial_iterator += 1

        return trial_response, Lforce_total, Rforce_total, raw_response
    
    def log_trial(self, phase, block, trial_num, focus, target, correct,
                  out_of_bounds, raw_response, gripLtotal, gripRtotal, grip_score,
                  points, instr_index):
        """
        REQUIRES: ...0 <= instr_index < len(Stimuli.NUM_FOCUS_INSTRUCTIONS),
                  target is in window units
        MODIFIES: self, pwd/temp_data
        EFFECTS:  Adds all relevant data of trial to subj_data. Writes all data
                  so far to a temp data file in case program crashes.
        """
    
        # fill in trial condition data
        self.trial_nums.append(trial_num)
        self.focuses.append(focus)
        self.phases.append(phase)
        self.blocks.append(block)
        self.targets.append(target + 1) # convert window units to grip units
        
        instr_index = (np.nan if phase == 'train' else focus + str(instr_index))
        self.instrs.append(instr_index)

        # fill in trial response data
        #   convert window units [-1,1] to grip units [0,2]
        self.grip_scores.append(grip_score + 1)
        self.raw_scores.append(raw_response)
        self.gripLtotals.append(gripLtotal)
        self.gripRtotals.append(gripRtotal)
        self.accuracies.append(correct)
        self.points_gained.append(points)
        self.out_of_bounds.append(out_of_bounds)

        # create temp file in case of crash
        temp_data = {'subject': [self.subj_id] * len(self.grip_scores),
                     'trial_num': self.trial_nums,
                     'focus': self.focuses,
                     'instr_last_seen': self.instrs,
                     'phase': self.phases,
                     'block': self.blocks,
                     'accuracy': self.accuracies,
                     'out_of_bounds': self.out_of_bounds,
                     'grip_score': self.grip_scores,
                     'points': self.points_gained,
                     'target': self.targets,
                     'raw_score': self.raw_scores,
                     'gripLtotal': self.gripLtotals,
                     'gripRtotal': self.gripRtotals,
                     'gripLmax': [self.grips.left_max] * len(self.grip_scores),
                     'gripRmax': [self.grips.right_max] * len(self.grip_scores),
                     'beta_factor': [self.beta] * len(self.grip_scores),
                     'frame_duration': [self.stimuli.win.monitorFramePeriod]\
                                        * len(self.grip_scores)}

        temp_DF = pd.DataFrame(temp_data)
        temp_DF.to_csv(self.TEMP_FILE, sep='\t')
        
        self.log_online_forces()
    
        return
                                        
    def log_online_forces(self):
        """
        REQUIRES: at least one trial has been run (len(self.gripLonline) > 0)
        MODIFIES: pwd/data/online, self.gripLonline, self.gripRonline
        EFFECTS:  Writes all current online grip force data to two csv's, one
                  for left and one for right.
        """

        # Correct trials with not the same length (prevent pandas errors)
        # (NOTE: this means that framerate is changing over course of experiment)
        lengths_L = set([len(trial) for trial in self.gripLonline.values()])
        lengths_R = set([len(trial) for trial in self.gripRonline.values()])
        
        if len(lengths_L) > 1:
            # get every trial list to same length so pandas is happy
            for key in self.gripLonline.keys():
                while len(self.gripLonline[key]) < max(lengths_L):
                    self.gripLonline[key].append(np.nan)

        if len(lengths_R) > 1:
            # do the same for R
            for key in self.gripRonline.keys():
                while len(self.gripRonline[key]) < max(lengths_R):
                    self.gripRonline[key].append(np.nan)

        # write online force files
        L_online_DF = pd.DataFrame(self.gripLonline)
        R_online_DF = pd.DataFrame(self.gripRonline)
        L_online_DF.to_csv(self.ONLINE_FILE_LEFT, sep='\t')
        R_online_DF.to_csv(self.ONLINE_FILE_RIGHT, sep='\t')

        return

    def run_train_trial(self, block, trial_num, target_ypos):
        """
        REQUIRES: -1 < target_ypos < 1
        MODIFIES: self (self.stimuli, self.grips, etc.)
        EFFECTS:  Runs a single trial for use in the training phase.
        """
        FIXATION_LIMIT = 2 # seconds
        TRIAL_LIMIT    = 3
        FEEDBACK_LIMIT = 3
        ANIMATION_TIME = 0.25
    
        ACCURACY_THRESHOLD = 0.15 # "transformed grip space" (window) units
    
        # FIXATION AND TRIAL GRAPHICS
        self.stimuli.disp_fixation()
        self.wait(FIXATION_LIMIT)
        # only show online feedback bars in first 5 trials
        early_trial = (True if trial_num < 5 and block == 1 else False)
        self.stimuli.disp_trial(target_ypos, online=early_trial)
    
        timer = psychopy.clock.CountdownTimer(TRIAL_LIMIT)
    
        # SUBJ RESPONSE
        trial_response, Ltotal, Rtotal, raw_score = self.get_subj_response(timer)
    
        # DATA LOGGING
        # compute accuracy
        distance = abs(trial_response - target_ypos)
        accurate = (1 if distance <= ACCURACY_THRESHOLD else 0)
        out_of_bounds = (1 if trial_response == -1 or trial_response > 1 else 0)
    
        # log data (publish temp results file)
        self.log_trial('train', block, trial_num, np.nan, target_ypos, accurate,
                       out_of_bounds, raw_score, Ltotal, Rtotal, trial_response,
                       np.nan, np.nan)
    
        #fixme: debug
        print('\n' + "trial response: " + str(trial_response))
        print("target: " + str(target_ypos))
        print("accuracy: " + str(accurate))
    
        # FEEDBACK
        self.stimuli.trial_timer_text.setAutoDraw(False)
        
        self.stimuli.disp_trial_response_animation(trial_response, ANIMATION_TIME)
        self.stimuli.disp_trial_hitmiss(accurate, trial_response)
        self.wait(FEEDBACK_LIMIT)
        
        self.stimuli.hide_trial_graphics()
                                         
        return distance

    def run_test_trial(self, block, trial_num, focus, target_ypos, instr_index):
        """
        REQUIRES: -1 < target_ypos < 1, focus is either 'I' or 'E',
                  0 <= instr_index < len(Stimuli.NUM_FOCUS_INSTRUCTIONS)
        MODIFIES: self (self.stimuli, self.grips, etc.)
        EFFECTS:  Runs a single trial with given phase and focus conditions
                  and records all relevant trial data.
        """
        
        FIXATION_LIMIT = 2 # seconds
        TRIAL_LIMIT    = 3
        FEEDBACK_LIMIT = 3
        ANIMATION_TIME = 0.25
        
        ACCURACY_THRESHOLD = 0.15 # "transformed grip space" (window) units
        
        # FIXATION AND TRIAL GRAPHICS
        self.stimuli.disp_fixation()
        self.wait(FIXATION_LIMIT)
        self.stimuli.disp_trial(target_ypos, online=False)
        
        timer = psychopy.clock.CountdownTimer(TRIAL_LIMIT)
        
        # SUBJ RESPONSE
        trial_response, Ltotal, Rtotal, raw_score = self.get_subj_response(timer)
    
        # DATA LOGGING
        # compute accuracy
        distance = abs(trial_response - target_ypos)
        accurate = (1 if distance <= ACCURACY_THRESHOLD else 0)
        out_of_bounds = (1 if trial_response == -1 or trial_response > 1 else 0)
        
        # update subj point total, no negative points
        if (1 - distance) > 0 and not out_of_bounds:
            points = int((1 - distance) * 100)
            self.subj_points += points
        else:
            points = 0
        
        # log data (publish temp results file)
        self.log_trial('test', block, trial_num, focus, target_ypos, accurate,
                       out_of_bounds, raw_score, Ltotal, Rtotal, trial_response,
                       points, instr_index)

        #fixme: debug
        print('\n' + "trial response: " + str(trial_response))
        print("target: " + str(target_ypos))
        print("accuracy: " + str(accurate))

        # FEEDBACK
        self.stimuli.trial_timer_text.setAutoDraw(False)
        
        self.stimuli.disp_trial_response_animation(trial_response, ANIMATION_TIME)
        self.stimuli.disp_trial_points(points, trial_response)
        self.wait(FEEDBACK_LIMIT)
        
        self.stimuli.hide_trial_graphics()
    
        return points

    def run_train_block(self, block):
        """
        REQUIRES: block > 0, block_instructions is built
        MODIFIES: self (basically... all of it.)
        EFFECTS: Runs a single block of trials for use during train phase.
                 Randomizes target condition order.
        """
        BLOCK_INTRO_TIME        = 2 # seconds
        END_BLOCK_FEEDBACK_TIME = 3
    
        # fixme: How many trials per train block?
        # Note: Trial targets are in window units! Don't pick a trial target
        # height above 0.9, that's too close to the top of the screen.
        TRIAL_TARGETS = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        NUM_TARGET_REPEATS = 1
        
        self.stimuli.build_trial_graphics()
    
        # create randomized conditions list
        trial_targets_order = data.TrialHandler(TRIAL_TARGETS,
                                                NUM_TARGET_REPEATS,
                                                method='random')
    
        self.stimuli.disp_block_instructions(block)
        self.wait(BLOCK_INTRO_TIME)
    
        trial_counter = 1
        for target in trial_targets_order:
            self.run_train_trial(block, trial_counter, target)
            trial_counter += 1
        
        return
    
    def run_test_block(self, block, focus):
        """
        REQUIRES: block > 0, focus is 'I' or 'E'
                  block_instructions is built
        MODIFIES: self (basically... all of it.)
        EFFECTS:  Runs a single test block of trials with
                  randomized target condition order.
        """
        
        BLOCK_INTRO_TIME = 2 # seconds
        END_BLOCK_FEEDBACK_TIME = 3 # seconds
        
        # fixme: How many trials per block?
        # Note: Trial targets are in window units! Don't pick a trial target
        # height above 0.9, that's too close to the top of the screen.
        TRIAL_TARGETS = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        NUM_TARGET_REPEATS = 1
        
        self.stimuli.build_trial_graphics()
        self.stimuli.build_point_total_feedback()
        
        # create randomized conditions list
        trial_targets_order = data.TrialHandler(TRIAL_TARGETS,
                                                NUM_TARGET_REPEATS,
                                                method='random')
                                                
        # create randomized focus instructions order
        focus_instr_order = data.TrialHandler(range(self.stimuli.NUM_FOCUS_INSTRUCTIONS),
                                              NUM_TARGET_REPEATS,
                                              method='random')
        
        self.stimuli.disp_block_instructions(block)
        self.wait(BLOCK_INTRO_TIME)
        
        points_total = 0
        trial_counter = 1
        # first trial(s) haven't seen focus instr feedback yet
        instr_index = np.nan
        for target in trial_targets_order:
            points = self.run_test_trial(block, trial_counter, focus, target,
                                         instr_index)
                                         
            # display focus-specific advice every 2 trials
            if trial_counter % 2 == 0:
                instr_index = focus_instr_order.next()
                self.stimuli.disp_focus_feedback(focus, instr_index)
                self.wait(END_BLOCK_FEEDBACK_TIME)
            
            points_total += points
            trial_counter += 1
        
        # display points at end of block in test phase
        self.stimuli.disp_block_feedback(points_total)
        self.wait(END_BLOCK_FEEDBACK_TIME)
        
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
        self.wait(INSTR_READ_TIME)
        
        self.stimuli.build_block_instructions()
    
        self.run_train_block(1)
    
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
            self.wait(INSTR_READ_TIME)
            
            self.run_test_block(current_block, focus)
            current_block += 1
            
        return
    
    def check_escaped(self):
        # Checks if participant pressed escape. Quits the experiment (while
        # ensuring all data has been logged) if yes. FIXME: implement
        if event.getKeys(keyList=["escape"]):
            self.stimuli.win.close()
            core.quit()
            
        return 1
    
    def wait(self, time_interval):
        """
        REQUIRES: time_interval > 0
        MODIFIES: self
        EFFECTS:  Uses psychopy countdown timer to wait the given time interval
                  while giving the subject a chance to press escape and quit.
        """
        timer = psychopy.clock.CountdownTimer(time_interval)
        while timer.getTime() > 0:
            self.check_escaped()

        return
    
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
                'instr_last_seen': self.instrs,
                'block': self.blocks,
                'accuracy': self.accuracies,
                'out_of_bounds': self.out_of_bounds,
                'grip_score': self.grip_scores,
                'points': self.points_gained,
                'target': self.targets,
                'raw_score': self.raw_scores,
                'gripLtotal': self.gripLtotals,
                'gripRtotal': self.gripRtotals,
                'gripLmax': [self.grips.left_max] * len(self.grip_scores),
                'gripRmax': [self.grips.right_max] * len(self.grip_scores),
                'beta_factor': [self.beta] * len(self.grip_scores),
                'frame_duration': [self.stimuli.win.monitorFramePeriod]\
                                   * len(self.grip_scores)}
                
        subj_data = pd.DataFrame(data)
        subj_data.to_csv(self.SESSION_FILE, sep='\t')
        
        # online grip force files already written
        
        return

    def start(self):
        """
        MODIFIES: stimuli
        EFFECTS:  presents start-of-experiment instructions to the window
                  (includes time delay for reading). Coordinates grip sensor
                  calibration.
        """
        INSTR_READ_TIME = 5 # seconds

        self.stimuli.disp_start()
        self.wait(INSTR_READ_TIME)

        self.calibrate_grips()
        
        # fixme: should block instr be built here?
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
    DATA_DIR = "C:/Users/seanpaul/Desktop/Box Sync/grip"
    os.chdir(DATA_DIR) # fixme: directory check

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
