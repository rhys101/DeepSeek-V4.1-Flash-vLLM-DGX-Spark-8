"""Reconsider a full-batch hint after overlapped completion releases a slot."""
import logging
import os

def install(module):
    if os.environ.get('SGLANG8_RETRY_FREE_ADMISSION') != '1':return
    cls=module.Scheduler
    original=cls.get_new_batch_prefill
    def get_new_batch_prefill(self,running_batch):
        if (
            self.spec_algorithm.is_dspark()
            and running_batch.batch_is_full
            and self.waiting_queue
            and self.chunked_req is None
            and self.get_num_allocatable_reqs(len(running_batch.reqs),running_batch=running_batch)>0
        ):
            # This flag is an admission shortcut, not an allocation. The original
            # prefill adder still applies every request, token and memory limit.
            running_batch.batch_is_full=False
            if not getattr(self,'_spark8_admission_reported',False):
                logging.getLogger(__name__).warning('SGLANG8_FREE_SLOT_ADMISSION_RETRY: rechecking a queued request after a slot became available')
                self._spark8_admission_reported=True
        return original(self,running_batch)
    cls.get_new_batch_prefill=get_new_batch_prefill
