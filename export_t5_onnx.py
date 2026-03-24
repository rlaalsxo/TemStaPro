#!/usr/bin/env python3
"""
Utility to export ProtTrans T5 encoder to ONNX format.

Alternatively, use the pre-converted model from HuggingFace:
    huggingface-cli download Rostlab/prot-t5-xl-uniref50-enc-onnx --local-dir ./prot_t5_onnx
"""

import argparse
import os
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


def export_t5_encoder_to_onnx(pt_dir, output_path, opset_version=14):
	device = torch.device("cpu")

	if os.path.isfile(f"{pt_dir}/pytorch_model.bin"):
		model = T5EncoderModel.from_pretrained(
			f"{pt_dir}/pytorch_model.bin", config=f"{pt_dir}/config.json")
	else:
		model = T5EncoderModel.from_pretrained(pt_dir)

	model = model.eval().to(device)
	tokenizer = T5Tokenizer.from_pretrained(pt_dir, do_lower_case=False)

	dummy_seq = "M V L S P A D K T N V K A A W G K V G A"
	inputs = tokenizer(dummy_seq, return_tensors="pt", padding=True)
	input_ids = inputs["input_ids"]
	attention_mask = inputs["attention_mask"]

	torch.onnx.export(
		model,
		(input_ids, attention_mask),
		output_path,
		input_names=["input_ids", "attention_mask"],
		output_names=["last_hidden_state"],
		dynamic_axes={
			"input_ids": {0: "batch_size", 1: "sequence_length"},
			"attention_mask": {0: "batch_size", 1: "sequence_length"},
			"last_hidden_state": {0: "batch_size", 1: "sequence_length"},
		},
		opset_version=opset_version,
	)
	print(f"Exported ONNX model to {output_path}")


def verify_onnx_output(pt_dir, onnx_path,
		test_sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"):
	import onnxruntime as ort

	device = torch.device("cpu")

	if os.path.isfile(f"{pt_dir}/pytorch_model.bin"):
		model = T5EncoderModel.from_pretrained(
			f"{pt_dir}/pytorch_model.bin", config=f"{pt_dir}/config.json")
	else:
		model = T5EncoderModel.from_pretrained(pt_dir)
	model = model.eval().to(device)
	tokenizer = T5Tokenizer.from_pretrained(pt_dir, do_lower_case=False)

	seq_spaced = " ".join(list(test_sequence))
	inputs = tokenizer(seq_spaced, return_tensors="pt", padding=True)

	with torch.no_grad():
		pt_output = model(inputs["input_ids"],
			attention_mask=inputs["attention_mask"])
	pt_emb = pt_output.last_hidden_state[0, :len(test_sequence)].numpy()

	session = ort.InferenceSession(onnx_path)
	onnx_output = session.run(None, {
		"input_ids": inputs["input_ids"].numpy().astype(np.int64),
		"attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
	})
	onnx_emb = onnx_output[0][0, :len(test_sequence)]

	cos_sim = np.dot(pt_emb.flatten(), onnx_emb.flatten()) / (
		np.linalg.norm(pt_emb) * np.linalg.norm(onnx_emb))
	max_diff = np.max(np.abs(pt_emb - onnx_emb))

	print(f"Cosine similarity: {cos_sim:.6f}")
	print(f"Max absolute difference: {max_diff:.6f}")
	return cos_sim, max_diff


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Export ProtTrans T5 encoder to ONNX")
	parser.add_argument("--pt-dir", required=True,
		help="path to ProtTrans model directory")
	parser.add_argument("--output", required=True,
		help="output ONNX file path")
	parser.add_argument("--opset", type=int, default=14,
		help="ONNX opset version (default: 14)")
	parser.add_argument("--verify", action="store_true",
		help="verify ONNX output against PyTorch")
	args = parser.parse_args()

	export_t5_encoder_to_onnx(args.pt_dir, args.output, args.opset)

	if args.verify:
		verify_onnx_output(args.pt_dir, args.output)
