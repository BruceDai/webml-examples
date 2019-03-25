describe('Unit Test/Model Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  let nn;

  async function assertThrowsAsync(fn, regExp) {
    let f = () => {};
    try {
      await fn();
    } catch(e) {
      f = () => {throw e};
    } finally {
      assert.throws(f, regExp);
    }
  }

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
  });

  afterEach(function(){
    nn = undefined;
  });

  describe('#addOperand API', function() {
    it('raise error when the length of inputs is 6 (not 7 or 10) for "CONV_2D" operation', async () => {
      let model = await nn.createModel(options);
      let operandIndex = 0;

      let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
      let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
      let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
      let type3 = {type: nn.INT32};
      let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};

      let op1 = operandIndex++;
      model.addOperand(type0);
      let op2 = operandIndex++;
      model.addOperand(type1);
      let bias = operandIndex++;
      model.addOperand(type2);
      let pad = operandIndex++;
      model.addOperand(type3);
      let rate_w = operandIndex++;
      model.addOperand(type3);
      let rate_h = operandIndex++;
      model.addOperand(type3);
      let act = operandIndex++;
      model.addOperand(type3);

      let op3 = operandIndex++;
      model.addOperand(type4);

      model.setOperandValue(bias, new Float32Array([0]));
      model.setOperandValue(pad, new Int32Array([1]));
      model.setOperandValue(rate_w, new Int32Array([1]));
      model.setOperandValue(rate_h, new Int32Array([1]));
      model.setOperandValue(act, new Int32Array([0]));

      model.addOperation(nn.CONV_2D, [op1, bias, pad, rate_w, rate_h, act], [op3]);
      model.identifyInputsAndOutputs([op1], [op3]);
      await model.finish();

      let compilation = await model.createCompilation();
      compilation.setPreference(getPreferenceCode(options.prefer));
      await compilation.finish();

      let execution = await compilation.createExecution();

      await assertThrowsAsync(async() => {
        await execution.startCompute();
      });
    });
  });
});
