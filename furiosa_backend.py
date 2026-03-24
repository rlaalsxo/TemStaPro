"""
FuriosaAI RNGD NPU backend for ProtTrans T5 encoder inference.
Drop-in replacement for prottrans_models.get_embeddings()
using ONNX model on FuriosaAI NPU via furiosa.runtime.
"""

import numpy as np
import sys


SHAPE_BUCKETS = [128, 256, 512, 1024, 2000]


def _get_bucket_size(token_len):
	for b in SHAPE_BUCKETS:
		if token_len <= b:
			return b
	return token_len


class FuriosaT5Runner:
	def __init__(self, onnx_model_path):
		from furiosa.runtime.sync import create_runner
		self._runner = create_runner(onnx_model_path)

	def run(self, input_ids_np, attention_mask_np):
		outputs = self._runner.run([input_ids_np, attention_mask_np])
		return np.array(outputs[0])

	def close(self):
		if self._runner is not None:
			self._runner.close()
			self._runner = None


def get_embeddings(runner, tokenizer, seqs, per_residue, per_protein,
		max_residues=4000, max_seq_len=2000, max_batch=100):
	results = {
		'per_res_representations': dict(),
		'mean_representations': dict()
	}

	seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]),
		reverse=True)
	batch = list()

	for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
		seq_len = len(seq)
		seq_spaced = ' '.join(list(seq))
		batch.append((pdb_id, seq_spaced, seq_len))

		n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

		if (len(batch) >= max_batch or n_res_batch >= max_residues or
				seq_idx == len(seq_dict) or seq_len > max_seq_len):
			pdb_ids, seqs_batch, seq_lens = zip(*batch)
			batch = list()

			max_token_len = max(seq_lens) + 1
			bucket_size = _get_bucket_size(max_token_len)

			token_encoding = tokenizer.batch_encode_plus(
				seqs_batch, add_special_tokens=True,
				padding='max_length', max_length=bucket_size,
				truncation=True)

			input_ids = np.array(
				token_encoding['input_ids'], dtype=np.int64)
			attention_mask = np.array(
				token_encoding['attention_mask'], dtype=np.int64)

			try:
				embedding_output = runner.run(input_ids, attention_mask)
			except Exception as e:
				print(f"{sys.argv[0]}: NPU runtime error for batch "
					f"(L={max(seq_lens)}): {e}", file=sys.stderr)
				continue

			for batch_idx, identifier in enumerate(pdb_ids):
				s_len = seq_lens[batch_idx]
				emb = embedding_output[batch_idx, :s_len]
				if per_residue:
					results["per_res_representations"][identifier] = \
						emb.squeeze()
				if per_protein:
					protein_emb = emb.mean(axis=0)
					results["mean_representations"][identifier] = \
						protein_emb.squeeze()

	return results
