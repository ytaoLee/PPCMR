"""Read-only Windows process working-set counters for the local benchmark.

The harness combines owner, dealer and client state. Its process memory must
not be relabelled as isolated client/dealer memory. Individual lifetime peaks
are not a simultaneous aggregate peak; sampled sums are reported separately.
"""
import ctypes
import os
import threading
from time import perf_counter


class Counters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[
        (n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize',
        'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]


def process_memory(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    api=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    api.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    handle=kernel.OpenProcess(0x410,False,pid)
    if not handle:raise OSError(ctypes.get_last_error(),'OpenProcess failed')
    try:
        c=Counters();c.cb=ctypes.sizeof(c)
        if not api.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):
            raise OSError(ctypes.get_last_error(),'GetProcessMemoryInfo failed')
        return {'working_set_bytes':int(c.WorkingSetSize),
                'lifetime_peak_working_set_bytes':int(c.PeakWorkingSetSize),
                'private_commit_bytes':int(c.PrivateUsage)}
    finally:kernel.CloseHandle(handle)


class MemorySampler:
    def __init__(self):
        self.stage='publication';self.pids={os.getpid():'trusted_harness'}
        self.peaks={};self.stages={};self.samples=0
        self.stop=threading.Event();self.thread=threading.Thread(target=self._run,daemon=True)
        self.start=perf_counter()

    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();self.sample()

    def sample(self):
        total=0;live=0
        for pid,role in list(self.pids.items()):
            try:values=process_memory(pid)
            except OSError:continue
            previous=self.peaks.setdefault(str(pid),{'role':role,'pid':pid})
            for name,value in values.items():previous[name]=max(previous.get(name,0),value)
            total+=values['working_set_bytes'];live+=1
        key=self.stage
        stage=self.stages.setdefault(key,{'samples':0,'max_sampled_sum_working_set_bytes':0,'max_live_processes':0})
        stage['samples']+=1;stage['max_sampled_sum_working_set_bytes']=max(stage['max_sampled_sum_working_set_bytes'],total)
        stage['max_live_processes']=max(stage['max_live_processes'],live)
        self.samples+=1

    def _run(self):
        while not self.stop.wait(.02):self.sample()

    def register_workers(self,engine,label):
        processes=engine.processes if hasattr(engine,'processes') else [engine.process]
        for i,p in enumerate(processes):self.pids[p.pid]=f'{label}_worker_{i}'
        self.sample()

    def report(self):
        return {'sampler_period_seconds':.02,'samples':self.samples,
            'process_counters':self.peaks,'stages':self.stages,
            'definition':'Windows working set; native lifetime peak per process. Sampled sums are 20ms observations and may miss transient aggregate peaks. Worker registration follows construction, so their native peaks include setup but sampled aggregate setup does not.',
            'harness_scope':'One trusted process combines owner, client and dealer; no isolated client/dealer RSS measurement.'}
