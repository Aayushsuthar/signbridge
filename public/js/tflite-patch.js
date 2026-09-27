// LiteRT.js can't resize a model's dynamic input ([-1, 543, 3]), so we do what TFLite's
// ResizeInputTensor does: write the real frame count into the input tensor's shape in the
// .tflite flatbuffer before compiling. Every op recomputes its output shape when the model is
// prepared, so the result is identical to running the model with that input length.
//
// Minimal FlatBuffers reader for the TFLite schema:
//   Model.subgraphs (field 2) → SubGraph.tensors (0), SubGraph.inputs (1)
//   Tensor.shape (0), Tensor.shape_signature (7)

function table(view, pos) {
  const vtable = pos - view.getInt32(pos, true);
  const vlen = view.getUint16(vtable, true);
  return {
    pos,
    field(i) {
      const o = 4 + i * 2;
      if (o >= vlen) return 0;
      const off = view.getUint16(vtable + o, true);
      return off ? pos + off : 0;
    },
  };
}
const deref = (view, p) => p + view.getUint32(p, true);
function vector(view, fieldPos) {
  const start = deref(view, fieldPos);
  return { len: view.getUint32(start, true), data: start + 4 };
}

export function inputShapeOffsets(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const model = table(view, deref(view, 0));
  const subgraphs = vector(view, model.field(2));
  const sg = table(view, deref(view, subgraphs.data));
  const tensors = vector(view, sg.field(0));
  const inputs = vector(view, sg.field(1));
  const inputIndex = view.getInt32(inputs.data, true);
  const tensor = table(view, deref(view, tensors.data + inputIndex * 4));
  const out = [];
  for (const f of [0, 7]) {
    const p = tensor.field(f);
    if (!p) continue;
    const v = vector(view, p);
    out.push({ field: f === 0 ? "shape" : "shape_signature", dims: Array.from({ length: v.len }, (_, i) => view.getInt32(v.data + i * 4, true)), pos: v.data });
  }
  return out;
}

// Returns a copy of the model whose first input dimension is fixed to `frames`.
export function withInputFrames(bytes, frames, offsets = inputShapeOffsets(bytes)) {
  const copy = bytes.slice();
  const view = new DataView(copy.buffer);
  for (const o of offsets) view.setInt32(o.pos, frames, true);
  return copy;
}
