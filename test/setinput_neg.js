describe('Unit Test/Execution Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  let nn;

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
  });

  afterEach(function(){
    nn = undefined;
  });

  describe('#setInput API', function() {
    it('raise error when the value being set to \'index\' is equal or greater than the size of inputs', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            compilation.setPreference(nn.PREFER_LOW_POWER);
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Float32Array(product(op.dimensions));
                inputData.fill(1);
                assert.throws(()=>{
                  execution.setInput(1, inputData);
                });
              });
            });
          });
        });
      });
    });
  });
});
