import time

class BenchmarkTimer(object):
    def __init__(self):
        self.start_timer_step = 0
        self.end_timer_step = 100001
        self.cur_step = 0
        self.total_time = 0.0
        self.step_start = 0.0

    def set_start_step(self, step):
        self.start_timer_step = step

    def time_begin(self):
        self.cur_step += 1
        if self.cur_step > self.start_timer_step:
            self.step_start = time.time()

    def time_end(self):
        if self.cur_step > self.start_timer_step:
            end = time.time()
            self.total_time += end - self.step_start

    def time_per_step(self):
        if self.cur_step <= self.start_timer_step:
            return 0.0
        return self.total_time / (self.cur_step - self.start_timer_step)