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
    it('raise error when setting 0 output for "TRANSPOSE" operation', async function() {
      let model = await nn.createModel(options);
      let input0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
      let input1 = {type: nn.TENSOR_INT32, dimensions: [4]};

      model.addOperand(input0);
      model.addOperand(input1);
      model.setOperandValue(1, new Int32Array([0, 2, 1, 3]));
      model.addOperation(nn.TRANSPOSE, [0, 1], []);

      model.identifyInputsAndOutputs([0], []);
      await model.finish();

      let compilation = await model.createCompilation();
      compilation.setPreference(getPreferenceCode(options.prefer));
      await compilation.finish();

      let execution = await compilation.createExecution();
      execution.setInput(0, new Float32Array([1.0, 2.0, 3.0, 4.0]));
      await assertThrowsAsync(async() => {
        await execution.startCompute();
      });
    });
  });
});
