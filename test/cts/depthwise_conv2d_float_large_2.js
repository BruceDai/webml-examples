describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Depthwise conv2d float large example/2', async function() {
    var model = await nn.createModel();
    var operandIndex = 0;

    let op1_value = [10, 21, 100, 0, 10, 22, 200, 0, 10, 23, 300, 0, 10, 24, 400, 0];
    let op4_expect = [600010, 700046, 830000, 900000];

    var type2 = {type: nn.INT32};
    var type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    var type3_length = product(type3.dimensions);
    var type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    var type0_length = product(type0.dimensions);
    var type1 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    var type1_length = product(type1.dimensions);

    var op1 = operandIndex++;
    model.addOperand(type0);
    var op2 = operandIndex++;
    model.addOperand(type0);
    var op3 = operandIndex++;
    model.addOperand(type1);
    var pad0 = operandIndex++;
    model.addOperand(type2);
    var act = operandIndex++;
    model.addOperand(type2);
    var stride = operandIndex++;
    model.addOperand(type2);
    var channelMultiplier = operandIndex++;
    model.addOperand(type2);
    var op4 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([0.25, 0, 10, 100, 0.25, 1, 20, 100, 0.25, 0, 30, 100, 0.25, 1, 40, 100]));
    model.setOperandValue(op3, new Float32Array([600000, 700000, 800000, 900000]));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.setOperandValue(channelMultiplier, new Int32Array([1]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, channelMultiplier, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
