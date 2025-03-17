
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.utils.file_util import *
from src.litmusbayes.prog.inst import *




class Strategy(object):
    def __init__(self):
        self.dif_litmus = None
        self.dif_state = None
        self.dif_exe =None
        self.dif_ra =None
        self.litmus_dir = ''

        self.thread_events_exe = []
        self.thread_events_to_id_map = {}


    def clear(self):
        self.dif_litmus = None
        self.dif_state = None
        self.dif_exe = None
        self.dif_ra = None
        self.litmus_dir = ''


        self.thread_events_exe = []
        self.thread_events_to_id_map = {}

    def set(self, dif_litmus, dif_state, dif_exe, dif_ra, litmus_dir):
        self.dif_litmus = dif_litmus
        self.dif_state = dif_state
        self.dif_exe = dif_exe
        self.dif_ra = dif_ra
        self.litmus_dir = litmus_dir

        self.thread_events_to_id_map = {event: i for i, event in enumerate(self.dif_exe)}

        for thread_id in range(self.dif_litmus.n_threads):
            self.thread_events_exe.append([event for event in self.dif_exe if event.pid == thread_id])

    def litmus_transform(self):
        assert self.dif_litmus is not None
        assert self.dif_state is not None
        assert self.dif_exe is not None
        assert self.dif_ra is not None

        pass

    def get_litmus_and_delete_file(self,litmus_file_path):
        content = read_file(litmus_file_path)
        litmus = parse_litmus(content)
        rm_file(litmus_file_path)

        return litmus
