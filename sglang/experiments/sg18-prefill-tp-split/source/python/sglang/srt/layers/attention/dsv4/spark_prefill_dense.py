"""TP8 query-row partition for the traced native SM121 FP4 prefill indexer.

The native scorer, ragged top-k and FP4 quantization remain the serving ones.
Source masks travel as lossless block IDs; no floating-point collective is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import os
import torch
import torch.nn.functional as F

logger=logging.getLogger(__name__)

def partition(rows,world,rank,tile=128):
    if rows<0 or world<1 or not 0<=rank<world or tile<1:
        raise ValueError((rows,world,rank,tile))
    extent=((rows+world*tile-1)//(world*tile))*tile
    start=rank*extent
    return start,min(rows,start+extent),extent

def candidate_block_ids(logits,lens,topk_blocks,block_size):
    scores=F.pad(logits,(0,-logits.shape[-1]%block_size),value=-torch.inf)
    scores=scores.unflatten(-1,(-1,block_size)).amax(-1)
    scores=scores.masked_fill(torch.arange(scores.shape[-1],device=logits.device)==(lens-1)//block_size,torch.inf)
    top=scores.topk(min(topk_blocks,scores.shape[-1]),dim=-1)
    return top.indices.masked_fill(~(top.values>-torch.inf),-1).to(torch.int32)

def candidate_mask(ids,width,block_size):
    blocks=(width+block_size-1)//block_size
    cols=ids.long().masked_fill(ids<0,blocks)
    keep=torch.zeros((ids.shape[0],blocks+1),dtype=torch.bool,device=ids.device).scatter_(1,cols,True)
    return keep[:,:blocks].repeat_interleave(block_size,dim=-1)[:,:width]

def local_selections(indexer,q_fp4,q_sf,k_fp4,k_sf,weights,ks,lens,
                     lengths,q_lengths,world,rank,scorer,topk,mask_topk,consume):
    rows=q_fp4.shape[0]
    start,end,extent=partition(rows,world,rank)
    source=indexer.is_candidate_source
    blocks=min(indexer.candidate_topk_blocks,(max(lengths)+indexer.candidate_block_size-1)//indexer.candidate_block_size) if source else 0
    packed=torch.full((extent,indexer.index_topk+blocks),-1,dtype=torch.int32,device=q_fp4.device)
    if start>=end:
        return packed
    local=slice(start,end)
    local_lens=lens[local];local_ks=ks[local]
    logits=scorer((q_fp4[local],q_sf[local]),(k_fp4,k_sf),weights[local],local_ks,local_ks+local_lens,((max(lengths)+3)//4)*4)
    offset=0
    for request,(width,count) in enumerate(zip(lengths,q_lengths)):
        lo=max(start,offset);hi=min(end,offset+count)
        if width and lo<hi:
            take=slice(lo-start,hi-start);scores=logits[take,:width]
            if source:
                row_lens=local_lens[take,None]
                scores.masked_fill_(torch.arange(width,device=logits.device)[None,:]>=row_lens,-torch.inf)
                # Preserve the baseline's copy bound inside the local partition.
                step=max(1,(1<<30)//(width*4))
                nblocks=min(indexer.candidate_topk_blocks,(width+indexer.candidate_block_size-1)//indexer.candidate_block_size)
                for row in range(0,hi-lo,step):
                    ids=candidate_block_ids(scores[row:row+step],row_lens[row:row+step],indexer.candidate_topk_blocks,indexer.candidate_block_size)
                    packed[lo-start+row:lo-start+row+ids.shape[0],indexer.index_topk:indexer.index_topk+nblocks]=ids
            elif indexer.uses_candidates:
                scores.masked_fill_(~consume[request][lo-offset:hi-offset],-torch.inf)
        offset+=count
    assert offset==rows
    selected=torch.empty((end-start,indexer.index_topk),dtype=torch.int32,device=logits.device)
    topk(logits,local_lens,out_offsets=local_ks,out_indices=selected)
    if indexer.uses_candidates and not source:
        selected=mask_topk(logits,selected,local_ks)
    packed[:end-start,:indexer.index_topk]=selected
    return packed

@dataclass
class NativePrefillTPSplit:
    group:object
    rank:int
    world:int
    enabled:bool
    minimum_context:int=32768
    minimum_rows:int=1024

    def __post_init__(self):self._logged=set()

    @classmethod
    def from_parallel(cls,parallel,*,is_draft):
        group=parallel.attn_tp_group
        settings=(os.environ.get('SPARK_PREFILL_TP_SPLIT','0'),os.environ.get('SPARK_PREFILL_TP_MIN_CONTEXT','32768'),os.environ.get('SPARK_PREFILL_TP_MIN_ROWS','1024'))
        agreed=group.all_gather_object(settings)
        if any(x!=settings for x in agreed):raise RuntimeError(f'Inconsistent native prefill TP settings: {agreed}')
        if settings[0] not in ('0','1'):raise ValueError(settings)
        context,rows=map(int,settings[1:])
        if context<32768 or rows<1024:raise ValueError(settings)
        topology=(group.world_size==8 and parallel.attn_cp_size==parallel.attn_dp_size==parallel.attn_dcp_size==1)
        hardware=torch.cuda.is_available() and torch.cuda.get_device_capability()==(12,1)
        enabled=settings[0]=='1' and topology and hardware and not is_draft
        logger.info('SPARK_NATIVE_PREFILL_TP_INIT rank=%s enabled=%s context=%s rows=%s draft=%s',group.rank_in_group,enabled,context,rows,is_draft)
        return cls(group,group.rank_in_group,group.world_size,enabled,context,rows)

    def eligible(self,rows,width,*,prefill,device):
        return self.enabled and prefill and device.type=='cuda' and rows>=self.minimum_rows and width>=self.minimum_context and not torch.cuda.is_current_stream_capturing()

    def select(self,indexer,q_fp4,q_sf,k_fp4,k_sf,weights,ks,lens,lengths,q_lengths,
               *,scorer,topk,mask_topk,consume,layer_id,ratio):
        local=local_selections(indexer,q_fp4,q_sf,k_fp4,k_sf,weights,ks,lens,lengths,q_lengths,
                               self.world,self.rank,scorer,topk,mask_topk,consume)
        gathered=self.group.all_gather(local,dim=0)[:q_fp4.shape[0]]
        masks=None
        if indexer.is_candidate_source:
            masks=[];offset=0
            for width,count in zip(lengths,q_lengths):
                if not width or not count:masks.append(torch.zeros((0,0),dtype=torch.bool,device=gathered.device))
                else:
                    nblocks=min(indexer.candidate_topk_blocks,(width+indexer.candidate_block_size-1)//indexer.candidate_block_size)
                    ids=gathered[offset:offset+count,indexer.index_topk:indexer.index_topk+nblocks]
                    masks.append(candidate_mask(ids,width,indexer.candidate_block_size))
                offset+=count
        key=(ratio,bool(indexer.is_candidate_source),bool(indexer.uses_candidates))
        if key not in self._logged:
            self._logged.add(key);start,end,extent=partition(q_fp4.shape[0],self.world,self.rank)
            logger.info('SPARK_NATIVE_PREFILL_TP_DISPATCH rank=%s layer=%s ratio=%s source=%s consumer=%s rows=%s width=%s start=%s end=%s gather_shape=%s',self.rank,layer_id,ratio,indexer.is_candidate_source,indexer.uses_candidates,q_fp4.shape[0],max(lengths),start,end,tuple(local.shape))
        return gathered[:,:indexer.index_topk],masks
