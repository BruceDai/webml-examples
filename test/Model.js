describe('Unit Test/Model Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  let nn;

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
  });

  afterEach(function(){
    nn = undefined;
  });

  describe('#addOperand API', function() {
    it('raise error when attempting to add an operand into the finished model', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
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
          assert.throws(() => {
            model.addOperand(op);
          });
        });
      });
    });
  });

  describe('#setOperandValue API', function() {
    it('raise error when attempting to reset the value for an operand of the finished model', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
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
          assert.throws(() => {
            let updatData = new Float32Array(product(op.dimensions));
            data.fill(100);
            model.setOperandValue(1, data);
          });
        });
      });
    });
  });

  describe('#addOperation API', function() {
    it('raise error when attempting to reset the operation of the finished model', function() {
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
          assert.throws(() => {
            model.addOperation(nn.MUL, [0, 1, 2], [3]);
          });
        });
      });
    });
  });

  describe('#identifyInputsAndOutputs API', function() {
    it('raise error when attempting to modify inputs/outputs of the finished model', function() {
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
          assert.throws(() => {
            model.identifyInputsAndOutputs([3], [0]);
          });
        });
      });
    });
  });
});
