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
    it('raise error when the length of inputs is 0 (not 3 or 4) for "RESIZE_BILINEAR" operation', async function() {
      let model = await nn.createModel(options);
      let operandIndex = 0;

      let type2 = {type: nn.INT32};
      let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
      let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};

      let op1 = operandIndex++;
      model.addOperand(type0);
      let op2 = operandIndex++;
      model.addOperand(type1);
      let height = operandIndex++;
      model.addOperand(type2);
      let width = operandIndex++;
      model.addOperand(type2);

      model.setOperandValue(height, new Int32Array([3]));
      model.setOperandValue(width, new Int32Array([3]));
      model.addOperation(nn.RESIZE_BILINEAR, [], [op2]);

      model.identifyInputsAndOutputs([op1], [op2]);
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
