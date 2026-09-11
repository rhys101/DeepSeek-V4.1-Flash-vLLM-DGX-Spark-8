"""Assert native TP8 resident storage without changing lookup arithmetic."""
import logging

def install(module):
    cls = module.EngramEmbedding
    original_init, original_owned = cls.__init__, cls._owned_rows
    def init(self, num_embeddings, dim, layer_id):
        original_init(self, num_embeddings, dim, layer_id)
        parallel = module.get_parallel()
        if parallel.tp_size != 8 or self.host_table is not None:
            raise RuntimeError('SGLang8 requires TP8 and native device-resident Engram shards')
        start = num_embeddings * parallel.tp_rank // 8
        end = num_embeddings * (parallel.tp_rank + 1) // 8
        if self.row_start != start or self.rows != end - start:
            raise RuntimeError('Unexpected Engram row partition')
        if tuple(self.weight.shape) != (end - start, dim) or tuple(self.scale.shape) != (end - start, dim // 32):
            raise RuntimeError('Engram allocation is not the expected owned-row shard')
        self._spark8_reported = False
        self._spark8_layer = layer_id
    def owned(self, indices):
        if not self._spark8_reported:
            if not self.weight.is_cuda or not self.scale.is_cuda:
                raise RuntimeError('Engram must be resident in CUDA memory before inference')
            nbytes = self.weight.numel() * self.weight.element_size() + self.scale.numel() * self.scale.element_size()
            logging.getLogger(__name__).warning(
                'SGLANG8_RESIDENT_ENGRAM layer=%s tp=8 rows=%s start=%s bytes=%s device=%s',
                self._spark8_layer, self.rows, self.row_start, nbytes, self.weight.device)
            self._spark8_reported = True
        return original_owned(self, indices)
    cls.__init__, cls._owned_rows = init, owned
