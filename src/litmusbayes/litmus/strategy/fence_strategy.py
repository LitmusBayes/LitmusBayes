

from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.litmus.strategy.strategy import Strategy
from src.litmusbayes.prog import FenceInst
from src.litmusbayes.litmus.litmus_changer import InjectPoint, InjectList

class FenceStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.add_fence_list = InjectList()


    def clear(self):
        self.__init__()
        self.add_fence_list = InjectList()

    def can_inject_fence(self, event_i, event_j):

        thread_events = self.thread_events_exe[event_i.pid]
        for event in thread_events:
            if event.inst.idx == event_j.idx:
                break
            if event.inst.idx > event_j.idx:
                return False, -2

        for event in reversed(thread_events):
            if event.inst.idx == event_i.idx:
                break
            if event.inst.idx < event_i.idx:
                return False, -2

        # print(event_i,event_j,self.dif_ra.fence(event_i,event_j))
        if self.dif_ra.fence(event_i, event_j):
            return False, -2

        inject_idx = event_i.idx
        for event in thread_events:
            if event.inst.idx == event_i.idx:
                break
            if event.inst.idx > inject_idx:
                inject_idx = event.inst.idx
        return True, inject_idx


    def find_inject_point(self):

        # print('thread_events_exe:')
        for thread_events in self.thread_events_exe:
            # print(thread_events)
            for i in range(len(thread_events)-1):
                event_i = thread_events[i]
                event_j = thread_events[i+1]
                flag, inject_idx =self.can_inject_fence(event_i, event_j)
                if not flag:
                    continue
                # print(event_i, event_j)
                if self.thread_events_to_id_map[event_i] +1 < self.thread_events_to_id_map[event_j]:
                    if self.dif_ra.fence(event_i, event_j):
                        continue

                    fence_inst = FenceInst('fence')
                    fence_inst.idx = -1
                    self.add_fence_list.add_inject_point(InjectPoint(event_i.pid, inject_idx, fence_inst))







    def litmus_transform(self):
        super().litmus_transform()


        # print('======fence mutate before======')
        self.find_inject_point()

        # for inject_point in self.add_fence_list.inject_list:
        #     print('add fence in:', inject_point)

        litmus_file_path = self.dif_litmus.mutate_new_litmus(self.add_fence_list, self.litmus_dir)
        litmus = self.get_litmus_and_delete_file(litmus_file_path)
        # print('======fence mutate after======')


        return litmus, self.add_fence_list








