"""Inspect or gracefully interrupt one verified Windows JShi console.

Run as a separate helper: never detach the caller's console. Refuse to signal
when unrelated processes share the target console.
"""
import argparse
import ctypes
import json
import os
import time


def main(pid, allowed, signal):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.FreeConsole()
    if not kernel.AttachConsole(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        kernel.SetConsoleCtrlHandler(None,True)
        ids=(ctypes.c_uint32*256)()
        count=kernel.GetConsoleProcessList(ids,len(ids))
        if count<1 or count>len(ids):
            raise RuntimeError('Cannot enumerate the complete target console')
        members=set(ids[:count])
        unexpected=members-set(allowed)-{pid,os.getpid()}
        if unexpected:
            raise RuntimeError(f'Refusing to signal unrelated console processes: {sorted(unexpected)}')
        result={'target':pid,'other_verified_members':sorted(members-{os.getpid()}),'signal_sent':False}
        if signal:
            if not kernel.GenerateConsoleCtrlEvent(0,0):
                raise ctypes.WinError(ctypes.get_last_error())
            result['signal_sent']=True
            time.sleep(.5)
        # stdout is the helper's redirected pipe; no control input is sent.
        print(json.dumps(result),flush=True)
    finally:
        kernel.FreeConsole()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--allowed',type=int,nargs='*',default=[])
    p.add_argument('--signal',action='store_true')
    a=p.parse_args()
    main(a.pid,a.allowed,a.signal)
