

from src.litmusbayes.analysis import Event
from src.litmusbayes.litmus.litmus_changer import InjectPoint, InjectList
from src.litmusbayes.litmus.strategy.strategy import Strategy
from src.litmusbayes.prog import LoadInst, AmoInst
from src.litmusbayes.prog.types import EType


class DelayStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.delay_start_list = InjectList()

    def clear(self):
        self.__init__()
        self.delay_start_list = InjectList()

    def delay_start_event(self):

        for thread_id in range(self.dif_litmus.n_threads):
            thread = self.thread_events_exe[thread_id]

            if thread[0].inst.idx == 0:
                continue

            start_id,start_event =thread[0].inst.idx,thread[0]
            for event in thread:
                if start_id > event.idx:
                    start_id = event.idx
                    start_event = event
            #TODO: for AMO or Lw/Sw imm
            inst = LoadInst('lw', 'x0', str(start_event.inst.inst.rs1), 0)
            inst.idx = -1
            inject_point = InjectPoint(thread_id, -1, inst)
            self.delay_start_list.add_inject_point(inject_point)


    def litmus_transform(self):
        super().litmus_transform()

        # print('======delay mutate before======')

        self.delay_start_event()
        # for inject_point in self.delay_start_list.inject_list:
        #     print('add delay in:',inject_point)

        litmus_file_path = self.dif_litmus.mutate_new_litmus(self.delay_start_list, self.litmus_dir)
        litmus = self.get_litmus_and_delete_file(litmus_file_path)
        # print('======delay mutate after======')


        return litmus, self.delay_start_list

