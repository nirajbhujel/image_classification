
import time


class Timer(object):
    """A simple timer."""
    def __init__(self, average=False,):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.
        self.average = average

    def tic(self, tic_once=False):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()
        self.tic_once = tic_once

    def toc(self, ): # use average if using tic every step, for single tic use diff ( later is useful for measuring loading data in the for iterator)
        self.diff = time.time() - self.start_time
    
        self.calls += 1

        if self.tic_once: # tic only once before the begining of for loop
            self.total_time = self.diff

        else: # tic every step
            self.total_time += self.diff

        if self.average:
            self.duration = self.total_time / self.calls

        else:
            if self.tic_once: # measure from start from the for loop
                self.duration = self.diff/self.calls

            else:
                self.duration = self.diff # divide by self.calls if single tic (at the begining of the for loop) 
            # self.duration = self.diff # divide by self.calls if tic every iteration 

    @property
    def fps(self, ):
        if self.total_time==0:
            return 0
        return self.calls/self.total_time
        
    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.
