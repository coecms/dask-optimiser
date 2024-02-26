
import asyncio
import weakref
import psutil
import os
import fcntl
import signal
import uuid
from typing import Dict, Any, Callable, List, Optional, Tuple
from distributed.client import Client
from distributed import Worker, LocalCluster
from distributed.diagnostics.plugin import WorkerPlugin
from distributed.core import rpc
from distributed.worker import logger

worker_override_params: Dict[str,Any]={"memory_limit":0,"nthreads":1}
### Treat the params below as analogies of each other
worker_analogy_params: Dict[str,str]={"threads_per_worker":"nthreads"}
### An option 'k' is affected by other options in the list v, if a user
### has specified one of those options, assume they're happy with 
### whatever dask has given them for k
worker_dependent_params: Dict[str,List[str]]={"nthreads":["n_workers"],"memory_limit":[]}

### Worker Plugin definition
class BindWorker(WorkerPlugin):
    def __init__(self, lock_suffix: str, slots_requested: Optional[Tuple[int,int]]):
        ### Main worker process is never bound
        self.slots_requested = None
        self.slots_available = None
        self.lock_suffix = lock_suffix
        self.lock_fn = None
        self.slots_requested = slots_requested

        self.lock_file_fd=None
        self.default_signal_handlers={}

    def handle_signal(self,signum,frame):
        if self.lock_file_fd is not None:
            fcntl.lockf(self.lock_file_fd,fcntl.LOCK_UN)
            self.lock_file_fd = None

    def register_signal_handler(self):
        for sig in range(1, signal.NSIG):
            try:
                self.default_signal_handlers[sig] = signal.signal(sig,self.handle_signal)
            except OSError:
                ### Handle signals we aren't allowed to modify
                pass
            

    def deregister_signal_handler(self):
        for sig in range(1, signal.NSIG):
            try:
                signal.signal(sig,self.default_signal_handlers[sig])
            except KeyError:
                pass

    
    def setup(self,worker: Worker):

        ### Do nothing if we're not in PBS
        if not os.getenv("PBS_NCPUS"):
            return

        self.worker = worker
        ### Do we need to derive slots_requested?
        if not self.slots_requested:
            self._derive_slots_requested()

        if not self.slots_available:
            self.slots_available=os.sched_getaffinity(os.getppid())

        ### User has requested overcommitting, do not bind
        if len(self.slots_available) < self.slots_requested[0] * self.slots_requested[1]:
            return
        
        ### Maybe we've already been bound?
        if self.slots_available != os.sched_getaffinity(0):
            return
        
        if not self.lock_fn:
            self.lock_fn = os.getenv('PBS_JOBFS','/tmp') + '/' + self.lock_suffix
        ### Take a lock (or wait for the lock to be available)
        f=open(self.lock_fn,'w')
        self.lock_file_fd=f.fileno()
        fcntl.lockf(self.lock_file_fd,fcntl.LOCK_EX)
        self.register_signal_handler()

        ### If we're undercommitting, we may want to assign more slots
        ### per worker than the number of threads (but not bind to them)
        slots_per_worker = self._reserved_slots_per_worker()

        ### Inspect our sibling processes and figure out where our
        ### next available slots are
        self.worker.pid=os.getpid()
        taken_slots=set()
        siblings=[ w.pid for w in psutil.Process(os.getppid()).children() if w.pid != self.worker.pid ]

        slot_list = sorted(list(self.slots_available))

        for s in siblings:
            sibling_affinity = os.sched_getaffinity(s)
            if sibling_affinity != self.slots_available:
                ### When figuring out taken slots, assume we're taking the nearest (slots_per_worker - nthreads) cores after
                ### nthreads cores returned by the getaffinity call
                start = slot_list.index(min(sibling_affinity))
                end = start + slots_per_worker
                for i in range(start,end):
                    taken_slots.add(slot_list[i])

        ### Bind to the number of requested threads
        os.sched_setaffinity(0,set(sorted(list(self.slots_available - taken_slots))[:self.worker.state.nthreads]))
 
        ### Release the lock
        fcntl.lockf(f.fileno(),fcntl.LOCK_UN)
        self.lock_file_fd=None
        self.deregister_signal_handler()
        f.close()
    
    def _reserved_slots_per_worker(self):
        ### If we've been launched via PBSCluster, used the data from derive_slots_requested
        if self.slots_requested[0] < len(self.slots_available):
            ### Undercommitting
            return len(self.slots_available) // self.slots_requested[0]
        return 1

    def _derive_slots_requested(self):
        ### An ugly hack assuming we've been launched by a dask worker from e.g. a PBSCluster
        nthreads=1
        nprocs=1
        cmdline=psutil.Process(os.getppid()).cmdline()
        for i,arg in enumerate(cmdline):
            if arg=="--nthreads":
                nthreads=int(cmdline[i+1])
            elif arg=="--nworkers":
                nprocs=int(cmdline[i+1])
        
        self.slots_requested=(nprocs, nthreads)

def override_worker_opts(opts: Dict[str,Any],client_args: Dict[str,Any]):
    out = opts
    for k,v in worker_override_params.items():
        if k in out and k not in client_args and not any([v in client_args for v in worker_dependent_params[k] ]):
            out[k] = v
    return out

def will_modify(client_args: Dict[str,Any], worker_spec: Dict[str,Any], override_params: Dict[str,Any]):

    to_modify=[]
    for k,v in override_params.items():
        if k in client_args:
            to_modify.append(False)
            continue
        try:
            if v==worker_spec[k]:
                to_modify.append(False)
                continue
        except KeyError:
            pass
        to_modify.append(True)
    return any(to_modify)


async def _wrap_awaitable(aw: Callable):
    return await aw

async def dask_setup(dask_client: Client):

    ### Do nothing if we're not in PBS
    if not os.getenv("PBS_NCPUS"):
        return

    nworkers=None

    ### This bit is only for local clusters:
    if isinstance(dask_client.cluster,LocalCluster):

        ### If we've been passed a LocalCluster object, not much we can
        ### do at this point unless we figure out how to pull out the
        ### users intention.
        if dask_client._start_arg or "address" in dask_client._startup_kwargs:
            return

        ### Definitely a LocalCluster, which means we've already
        ### launched workers. Get their details now
        slots=sum(i.nthreads for i in dask_client.cluster.workers.values())

        ### Try not to use 'hidden' attributes
        client_args=dask_client._startup_kwargs.copy()
        for k,v in worker_analogy_params.items():
            if k in client_args:
                client_args[v]=client_args[k]
                del client_args[k]

        new_spec=dask_client.cluster.new_spec.copy()
        worker_spec=dask_client.cluster.worker_spec.copy()

        if will_modify(client_args,new_spec.get("options",{}),worker_override_params):
            print("Modifying workers")

            ### Set the new worker spec
            cls,opts=new_spec["cls"], new_spec.get("options",{})
            opts=opts.copy()
            opts = override_worker_opts(opts,client_args)
            dask_client.cluster.new_spec={"cls":cls,"options":opts}

            ### Would have been nice
            #await dask_client.cluster.scale(0)
            #print(dask_client.cluster.new_spec)
            #await dask_client.cluster.scale(workers_to_launch)

            ### Copied from dask.distributed.deploy.spec.py l 365-388
            workers=[]
            my_worker_spec={}

            for k,v in worker_spec.items():
                cls,opts=v["cls"], v.get("options",{})
                opts=opts.copy()
                opts = override_worker_opts(opts,client_args)
                my_worker_spec[k]={"cls":v["cls"],"options":opts}
                if "name" not in opts:
                    opts["name"]=k

                worker = cls(
                    getattr(dask_client.cluster.scheduler,"contact_address",None) 
                    or dask_client.scheduler.addr,
                    **opts,
                )
                workers.append(worker)

            ### Do we need more workers? ONLY EVALUATE THIS IF THE USER DID NOT SPECIFY nthreads or n_workers
            if "nthreads" not in client_args and "n_workers" not in client_args and len(workers) < slots:
                cls, opts = new_spec["cls"], new_spec.get("options",{})
                opts = opts.copy()
                opts = override_worker_opts(opts,client_args)
                for i in range(len(workers),slots):
                    i_opts=opts.copy()
                    my_worker_spec[i]={"cls":cls,"options":i_opts}
                    if "name" not in opts:
                        i_opts["name"]=i
                    worker = cls(
                        getattr(dask_client.cluster.scheduler,"contact_address",None)
                        or dask_client.scheduler.addr,
                        **i_opts,
                    )
                    workers.append(worker)

            if dask_client.cluster.workers:
                await asyncio.wait(
                                [asyncio.create_task(w.close()) for w in dask_client.cluster.workers.values()]
                            )

            if workers:
                await asyncio.wait(
                    [asyncio.create_task(_wrap_awaitable(w)) for w in workers]
                        )
                for w in workers:
                    w._cluster = weakref.ref(dask_client.cluster)
                    await w  # for tornado gen.coroutine support

            dask_client.cluster.workers={ w.name:w for w in workers }
            dask_client.cluster.worker_spec=my_worker_spec

        nworkers = ( len(dask_client.cluster.workers), min(i.nthreads for i in dask_client.cluster.workers.values()) )

    ### Register worker plugin after we've finalised workers
    plugin = BindWorker(str(uuid.uuid1()),nworkers)
    await dask_client.upload_file(__file__)
    await dask_client.register_worker_plugin(plugin)
    