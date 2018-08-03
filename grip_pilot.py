"""
Prototype attentional focus grip task.

Cognition, Control, and Action Lab - Dr. Taraz Lee
Tyler A and Sean A
Behavior:
    The task asks the patient to squeeze two grip sensors such that the force
    L of the left hand and R of the right hand in the equation
    L^2 + R^2 = Z result in a target Z value.
Feedback:
    During the trial, the patient receives online visual feedback about the value
    of the left and right grip forces. After the trial, they receive feedback
    specifying the error as distance from target in window units
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
        # fixme: debug
        print(self.nJoys)
        
        JOYSTICK_ID  = 0 # not sure what this is for
        self.sensors = joystick.Joystick(JOYSTICK_ID)
        self.nAxes   = self.sensors.getNumAxes()
    
        # default max values
        self.left_max  = 1
        self.right_max = 1
    
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
        # NOTE:    grip sensor input range is [-1, 1]
        
        raw = self.sensors.getY()
        force = (raw + 1) / self.left_max
        return force, raw

    def get_right(self):
        # EFFECTS: returns activation of right grip normalized by right-hand max
        # NOTE:    grip sensor input range is [-1, 1]
        
        raw = self.sensors.getX()
        force = (raw + 1) / self.right_max
        return force, raw

    def get_score(self):
        """
        EFFECTS: Calculates subject response score using the equation 
                 specified in the docstring and subject's max grip strengths.
        """
    
        # NOTE: - 1 at end is to transform score to psychopy window units
        score = (self.get_left()[0] ** 2) + (self.get_right()[0] ** 2) - 1
        return score


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
        
        TRIAL_POLE_WIDTH  = 0.1
        TRIAL_POLE_HEIGHT = 2  # the entire window
        self.trial_pole = visual.Rect(self.win, width=TRIAL_POLE_WIDTH,
                                      height=TRIAL_POLE_HEIGHT, pos=[0,0],
                                      fillColor="grey", lineColor="grey")
                                      
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
                                                   
        TRIAL_TIMER_XPOS = -0.75
        TRIAL_TIMER_YPOS = 0.75
        self.trial_timer_text = visual.TextStim(self.win, text="",
                                                pos=[TRIAL_TIMER_XPOS,
                                                     TRIAL_TIMER_YPOS])
        
        return
    
    def build_start_instructions(self):
        # Builds start-of-experiment instructions to window buffer.
        
        instructions = "Welcome! You have a grip force sensor in each hand.\
                        After we calibrate each, you will squeeze both to control\
                        a cannon. In each trial, squeeze both sensors to control\
                        the power of the shot. After 3 seconds, the cannon will \
                        fire, and you will see where your shot ended up."
        
        self.exp_instructions = visual.TextStim(self.win, text=instructions)
        return
    
    def build_calibration_instructions(self):
        # Builds calibration instructions to the window buffer.
        instr_L = "Using your LEFT hand only, squeeze the LEFT grip as\
                   hard as you can."
        instr_R = "Using your RIGHT hand only, squeeze the RIGHT grip as\
                   hard as you can."
    
        self.calibrate_L_instr = visual.TextStim(self.win, text=instr_L)
        self.calibrate_R_instr = visual.TextStim(self.win, text=instr_R)

        # A rectangle to display online grip force input.
        CALIB_RECT_YPOS = -0.5
        self.CALIB_RECT_WIDTH = 0.1

        self.calibration_rectangle = visual.Rect(self.win, fillColor="orange",
                                                 pos=[0, CALIB_RECT_YPOS],
                                                 size=[self.CALIB_RECT_WIDTH,0])
        
        return
    
    def build_block_instructions(self):
        return
    
    # GRAPHICS DISPLAY INTERFACE
    def clear(self):
        # FIXME: turn every graphic's autoDraw to false?
        return
        
    def disp_start(self):
        # EFFECTS: Displays start-of-experiment instructions for five seconds.
        
        INSTR_READ_TIME = 5 # seconds
        
        self.build_start_instructions()
        
        self.exp_instructions.draw(win=self.win)
        self.win.flip()
        
        psychopy.clock.wait(INSTR_READ_TIME)
    
        return

    def disp_fixation(self, time_interval):
        # REQUIRES: self.fixation is built, time_interval is float or int
        # MODIFIES: self
        # EFFECTS:  displays the fixation dot for the specified interval.
        
        self.fixation.draw()
        self.win.flip()
        psychopy.clock.wait(time_interval)
        
        return

    def disp_trial(self, target_ypos):
        # REQUIRES: -1 < target_ypos < 1, trial graphics built
        # MODIFIES: self
        # EFFECTS:  displays graphics for the trial (inviting subject response)

        # online grip feedback and counter
        self.trial_pole.setAutoDraw(True)
        self.trial_L_bar.setAutoDraw(True)
        self.trial_R_bar.setAutoDraw(True)
        self.trial_timer_text.setAutoDraw(True)
        
        # target
        self.trial_target.pos = [0, target_ypos]
        self.trial_target.setAutoDraw(True)

        return

    def hide_trial_graphics(self):
        # REQUIRES: trial graphics built
        # MODIFIES: self
        # EFFECTS:  halts drawing of trial graphics (removes them from the window)

        self.trial_pole.setAutoDraw(False)
        self.trial_L_bar.setAutoDraw(False)
        self.trial_R_bar.setAutoDraw(False)
        self.trial_timer_text.setAutoDraw(False)
        self.trial_response.setAutoDraw(False)
        self.trial_target.setAutoDraw(False)

        return

    def disp_trial_feedback(self, correct, trial_response_ypos, time_interval):
        """
        REQUIRES: correct is a bool, -1 < trial_response_ypos < 1,
                  time_interval > 0
        MODIFIES: self
        EFFECTS:  Displays "hit" or "miss" based on correctness of trial response
        """
        
        if correct:
            # display Hit!
            self.trial_feedback_hit.pos = [self.TRIAL_FEEDBACK_XPOS,
                                           trial_response_ypos]
            self.trial_feedback_hit.draw()
            self.win.flip()
            psychopy.clock.wait(time_interval)
        else:
            # display Miss!
            self.trial_feedback_miss.pos = [self.TRIAL_FEEDBACK_XPOS,
                                            trial_response_ypos]
            self.trial_feedback_miss.draw()
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
        
        #DATA_COLS = ['subject','trial_num','focus','phase','block','accuracy',
        #             'grip_score','target','gripLraw','gripRraw','gripLnormd',
        #             'gripRnormd','gripLmax','gripRmax']
        
        #self.subj_data = pd.DataFrame(columns=DATA_COLS)
    
    def init_data_lists(self):
        # EFFECTS: creates lists to hold trial data values (until they are
        #          packaged in a pd.DataFrame)
        # NOTE: subj_id, gripLmax, and gripRmax remain constant during experiment
        
        self.trial_nums = []
        self.focuses = []
        self.phases = []
        self.blocks = []
        self.accuracies = []
        self.grip_scores = []
        self.targets = []
        self.gripLraws = []
        self.gripRraws = []
        self.gripLnormds = []
        self.gripRnormds = []
    
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
            max_left = self.grips.get_left()[0]
                
            # fixme: debug
            print(max_left)
            
            # display online rectangle for input
            self.stimuli.calibration_rectangle.size = [self.stimuli.CALIB_RECT_WIDTH,
                                                       max_left]
            counter_text.text = str(ceil(timer.getTime()))
            
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
            max_right = self.grips.get_right()[0]
            
            # fixme: debug
            print(max_right)
            
            # display online rectangle for input
            self.stimuli.calibration_rectangle.size = [self.stimuli.CALIB_RECT_WIDTH,
                                                       max_right]
            counter_text.text = str(ceil(timer.getTime()))
            
            # refresh window
            self.stimuli.win.flip()
        
        self.grips.calibrate_right(max_right)
        
        counter_text.setAutoDraw(False)
        self.stimuli.calibration_rectangle.setAutoDraw(False)

        #FIXME: debug
        print("right max: " + str(self.grips.right_max))
            
        return self.grips.right_max
    
    def calibrate_grips(self):
        # MODIFIES: stimuli, grips
        # EFFECTS:  prompts subject to follow calibration procedure and updates
        #           grips accordingly
        
        CALIBRATION_INTERVAL = 3 # seconds
        INSTR_READ_TIME      = 3
        # text to show countdown timer
        countdown_text = visual.TextStim(self.stimuli.win, text="")

        self.stimuli.build_calibration_instructions()
        
        self.calibrate_grip_L(countdown_text, INSTR_READ_TIME, CALIBRATION_INTERVAL)
        self.calibrate_grip_R(countdown_text, INSTR_READ_TIME, CALIBRATION_INTERVAL)
            
        return

    def get_subj_response(self, countdown_timer):
        """
        REQUIRES: countdown_timer is psychopy.clock.CountdownTimer
                  Trial graphics are built (and showing)
        MODIFIES: stimuli, grips
        EFFECTS:  Updates trial graphics as subj responds using grip sensors.
                  Returns calculated response score of trial.
        """
        while countdown_timer.getTime() > 0:
            self.stimuli.trial_timer_text.text = str(ceil(countdown_timer.getTime()))
            
            # get grip sensor data
            Lforce, Lraw = self.grips.get_left()
            Rforce, Rraw = self.grips.get_right()

            # update left grip online feedback bar
            self.stimuli.trial_L_bar.height = Lforce
            self.stimuli.trial_L_bar.pos = [self.stimuli.TRIAL_L_BAR_XPOS,
                                            (-1 + Lforce / 2)]
            # update right "
            self.stimuli.trial_R_bar.height = Rforce
            self.stimuli.trial_R_bar.pos = [self.stimuli.TRIAL_R_BAR_XPOS,
                                            (-1 + Rforce / 2)]
                                            
            trial_response = self.grips.get_score()
            self.stimuli.win.flip()

        return trial_response, Lforce, Lraw, Rforce, Rraw
    
    def log_trial(self, phase, block, trial_num, focus, target, correct, 
                  gripLraw, gripLnormd, gripRraw, gripRnormd, grip_score):
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

        # create temp file in case of crash
        temp_data = {'subject': [self.subj_id] * len(self.grip_scores),
                     'trial_num': self.trial_nums,
                     'focus': self.focuses,
                     'phase': self.phases,
                     'block': self.blocks,
                     'accuracy': self.accuracies,
                     'grip_score': self.grip_scores,
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
        
        FIXATION_LIMIT = 3 # seconds
        TRIAL_LIMIT    = 5
        FEEDBACK_LIMIT = 3
        
        ACCURACY_THRESHOLD = 0.1 # "transformed grip space" units
        
        RESPONSE_ANIMATION_STEP = 0.01 # window units
        RESPONSE_ANIMATION_STEP_INTERVAL = 0.002 # seconds
        
        # FIXATION AND TRIAL GRAPHICS
        self.stimuli.disp_fixation(FIXATION_LIMIT)
        self.stimuli.disp_trial(target_ypos)
        
        timer = psychopy.clock.CountdownTimer(TRIAL_LIMIT)
        
        # SUBJ RESPONSE
        trial_response, Lforce, Lraw, Rforce, Rraw = self.get_subj_response(timer)
    
        # DATA LOGGING
        # compute accuracy
        accurate = abs(trial_response - target_ypos) <= ACCURACY_THRESHOLD
        # log data (publish temp results file)
        self.log_trial(phase, block, trial_num, focus, target_ypos, accurate,
                       Lraw, Lforce, Rraw, Rforce, trial_response)

        #fixme: implement/debug
        print('\n' + "trial response: " + str(trial_response))
        print("target: " + str(target_ypos))
        print("accuracy: " + str(accurate))

        # FEEDBACK
        self.stimuli.trial_timer_text.setAutoDraw(False)
        self.stimuli.trial_response.setAutoDraw(True)
        
        self.stimuli.trial_response.pos = [0, trial_response]

        """
        # fixme: it looks a little pixelated...
        # animate subj response
        responsebox_ypos = -1 # bottom of screen
        while responsebox_ypos < trial_response:
            self.stimuli.trial_response.pos = [0, responsebox_ypos]
            responsebox_ypos += RESPONSE_ANIMATION_STEP
            #psychopy.clock.wait(RESPONSE_ANIMATION_STEP_INTERVAL)
            self.stimuli.win.flip()
        """
        
        self.stimuli.disp_trial_feedback(accurate, trial_response, FEEDBACK_LIMIT)
        
        self.stimuli.hide_trial_graphics()
    
        return accurate

    def run_block(self, phase, block, focus):
        """
        REQUIRES: phase is 'train' or 'test', block > 0, focus is 'I' or 'E'
        MODIFIES: self (basically... all of it.)
        EFFECTS:  Runs a single block of trials with phase constraints in a
                  counterbalanced design.
        """
        
        # fixme: phase support
        # fixme: ensure target conditions are accessible to subj's grip strength?
        TRIAL_TARGETS = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        NUM_TARGET_REPEATS = 2
        
        self.stimuli.build_trial_graphics()
        
        # create randomized conditions list
        trial_targets_order = data.TrialHandler(TRIAL_TARGETS,
                                                NUM_TARGET_REPEATS,
                                                method='random')
        
        # fixme: implement block intro graphics
        
        trial_counter = 1
        for target in trial_targets_order:
            self.run_trial(phase, block, trial_counter, focus, target)
            trial_counter += 1
        
        return
    
    def check_escaped(self):
        # Checks if participant pressed escape. Quits the experiment (while
        # ensuring all data has been logged) if yes. FIXME: implement
        if event.getKeys(keyList=["escape"]):
            core.quit()
            win.close()
            
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
        
        # fixme: one block
        self.run_block("train", 1, 'E')
        
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
    DATA_DIR = "/Users/seanpaul/Box Sync/grip"

    # SUBJECT INFORMATION
    SESSION_NAME = 'bimanual_grip_task'
    SUB_ID = raw_input("Please enter the participant's ID number: ")

    # run experiment
    focus_task = Experiment(SUB_ID, SESSION_NAME, DATA_DIR)

    focus_task.start()
    focus_task.end()

    return 0

# run the program
if __name__ == '__main__':
    main()
