describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Add example', async function() {
    let operandIndex = 0;
    let model = await nn.createModel(options);
    let TENSOR_DIMENSIONS = [2, 2, 2, 2];
    let value0 = 0.4;
    let value1 = 0.5;
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);

    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(float32TensorType);
    let input0Data = new Float32Array(tensorLength);
    input0Data.fill(value0);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(float32TensorType);
    let output = operandIndex++;
    model.addOperand(float32TensorType);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();

    compilation.setPreference(getPreferenceCode(options.prefer));

    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1Data = new Float32Array(tensorLength);
    input1Data.fill(value1);

    execution.setInput(0, input1Data);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], input0Data[i] + input1Data[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/5', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/6', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120,
                                       130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
                                       250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
                                       370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480,
                                       490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 11,  12,  21,  22,  31,  32,  43,  44,  53,  54,  63,  64,
                         75,  76,  85,  86,  95,  96, 107, 108, 117, 118, 127, 128,
                        131, 132, 141, 142, 151, 152, 163, 164, 173, 174, 183, 184,
                        195, 196, 205, 206, 215, 216, 227, 228, 237, 238, 247, 248,
                        251, 252, 261, 262, 271, 272, 283, 284, 293, 294, 303, 304,
                        315, 316, 325, 326, 335, 336, 347, 348, 357, 358, 367, 368,
                        371, 372, 381, 382, 391, 392, 403, 404, 413, 414, 423, 424,
                        435, 436, 445, 446, 455, 456, 467, 468, 477, 478, 487, 488,
                        491, 492, 501, 502, 511, 512, 523, 524, 533, 534, 543, 544,
                        555, 556, 565, 566, 575, 576, 587, 588, 597, 598, 607, 608];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/5', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/6', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1,  2,  3,  4]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 21, 22, 23, 24,
                        31, 32, 33, 34, 41, 42, 43, 44];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding same example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [1.85284, -0.0393656, -0.127353, 1.43115, -0.302294, -1.0402, 0.655023, -0.587614, 1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -0.346357, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.104506, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, 1.42026, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, -0.343435, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, -1.46717, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494, 1.23741, -0.527402, -0.39954, -0.0128623, 1.3644, 0.985755, -0.718118, -0.1008, 1.24327];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type4_length = product(type4.dimensions);

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

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding same example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.93729883, 1.2219346, 1.63162, 1.8668158, 1.6269842, -1.9670266, -0.15441051, 0.5595218, -0.99790573, 2.3631613, -1.3033884, 1.2685156, -1.054666, 0.31054103, 1.3991811, -0.46040928, -0.5490349, 0.14452362, -0.3481132, 0.62236893, -0.83281666, -3.7738001, 0.5568896, 0.9274717, 0.48187765, -0.9098393, -2.0777307, 1.213712, -0.24457066, 0.14877218, -0.5466188, 0.9753277, -0.53815746, -0.21209812, 0.43179023, 3.625693, 0.18136086, -0.61304003, 0.0709098, 1.9279834, 1.5563309, 0.9073066, 2.7159054, -2.4034908, 0.37647444, -1.606053, 1.3484854, -0.9874026, 0.13162848, -2.3492568, -2.4371247, 1.1747775, 1.2780867, -1.0992509, -0.15879333, 0.62347984, -0.39933106, 0.2999848, -1.6485932, 0.12523836, -0.4088197, 0.7373756, -0.43234983, 0.1826737];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type4_length = product(type4.dimensions);

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

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding valid example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
    let type4_length = product(type4.dimensions);

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

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding valid example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.14452362, -0.3481132, 0.62236893, -0.83281666, -3.7738001, 0.5568896, -0.9098393, -2.0777307, 1.213712, -0.24457066, 0.14877218, -0.5466188, -0.21209812, 0.43179023, 3.625693, 0.18136086, -0.61304003, 0.0709098, 0.9073066, 2.7159054, -2.4034908, 0.37647444, -1.606053, 1.3484854];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4, 6, 1]};
    let type4_length = product(type4.dimensions);

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

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding same example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [-1.27853, 1.74987, -0.876718, 0.989692, 0.298548, 0.522103, -0.536896, -0.179382, -0.966914, 1.33708, 1.37042, -0.495494, 1.43859, -1.548, -0.430026, -0.662793, -0.0867897, -0.900658, -0.524396, 0.255731, -0.779081, 0.12666, 0.915651, -0.444765, -0.186842, -1.87308, 1.21135, -0.385009, 1.72032, -1.56036, -1.23059, 1.23694, 0.00200015, 0.359522, 1.60084, 0.434006, -0.282945, 2.37292, -1.28653, 0.0847837, -0.352093, -2.39659, 0.149246, 0.920351, -1.34346, 0.952311, -0.35811, 0.403449, 0.484796, -1.19989, -0.684298, -1.41301, 0.103177, -0.307039, 1.17741, 2.58936, -2.76237, -1.21565, -1.09619, 1.17432, 0.512143, 0.771379, 0.399879, -0.0533093, 0.290864, 0.95563, 1.16328, 1.80768, -1.52564, -0.126476, -0.185224, -0.114779, 1.2248, 0.237127, -0.213297, -0.619941, 0.497944, -1.68688, 1.59314, -0.127337, 0.111419, 1.13719, 1.68537, -0.479644, 1.18608, -2.52744, 1.34136, 0.548297, -2.0838, 2.64585, -0.993354, 0.128238, 1.26092, 0.318668, 0.893795, -0.0600559, -0.629126, -0.949229, 2.25828, -1.961, 0.00589599, -0.187854, -1.02403, 0.396121, 1.3704, 3.99355, 0.434221, 0.274464, -0.562438, -0.914871, 0.539129, -0.928687, 0.834954, 0.844178, -0.566053, -0.957341, 0.933336, 1.13613, -1.22109, 1.4649, -0.414666, -0.452821, -0.706006, -1.72657, -0.726574, -0.0979362, -0.478669, 1.78703, -0.639288, 1.48565, -0.179904, 1.01003, -0.317118, -0.675387, 1.90969, -1.38343, 0.697255, -0.292255, 1.81634, 0.717801, 0.862479, -0.407478, -0.343106, -0.0353232, -0.481893, -0.135565, -2.95941, 0.247846, 2.67757, -2.23999, -0.519673, 0.254447, 0.415283, -1.01065, 0.507911, 0.979926, -0.184304, -0.000950437, -0.734348, -0.196685, -0.713241, 0.594972, 0.0845042, 2.48496, 0.385019, -0.201145, 0.533332, -0.904872, -0.333518, -0.581063, -2.07065, 0.118687, -1.86708, -0.601987, 0.432037, 1.73923, 0.590007, 0.419788, 0.314198, 2.12817, 0.570793, -1.15998, -0.348587, -1.10231, -2.13091, 0.134467, -0.460382, 0.138338, 3.455, 0.679068, -0.190282, -0.0307461];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding same example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [-1.1709702, -0.15783468, 0.23686486, 0.18489599, 2.4330406, -0.83415174, -0.9379462, 2.3037708, -1.8530072, -0.6243031, -0.5570223, 0.0571152, -0.08371013, 1.2763771, -1.0637468, -0.034401, 0.2984934, -0.37836313, 0.0089688, 0.317119, -1.6136328, -0.04100603, -0.14438102, 0.42039546, 1.0826977, -1.4740779, 0.93527263, -1.6937343, 0.9933808, -0.00613139, 1.122337, -1.0007913, -0.19323519, -1.536109, 1.7444146, 0.07549044, 0.14941335, -1.0424318, 1.6394564, 0.05120939, 0.44861174, -1.0254043, 0.2891852, 1.0087392, -1.6427121, 0.22787222, 0.7912595, -1.0432496, 0.3554331, -1.2678999, 0.8390462, 2.43199, 0.7855995, -0.89247733, -0.8421999, -0.21892299, -1.4752842, 1.4666033, 1.2671402, -1.9113951, 2.8163433, 0.42375302, 1.384544, -0.05179526, -0.09704095, -1.0454197, -0.7849326, -1.0726444, -2.2269573, 0.38411486, -0.15826067, 1.7655121, -0.21607418, -0.220653, -1.0505613, -1.9059162, -1.3809854, -0.21753564, -0.6674532, 0.9924352, -1.3004371, 1.3581562, -0.50957847, 0.43931735, -0.30051446, 1.9288344, 1.3749437, 0.24674952, -1.3658104, -0.24712396, 1.8478253, 0.0548588, 0.5765619, 0.12883057, -0.9403651, 0.8154919, -0.38991603, 0.2580973, 0.27158368, -1.0782311, 2.7078195, 0.54151404, -1.2969424, -0.4957502, -0.8728107, 2.7895741, 0.764437, 1.8849254, -0.16873728, 0.36533558, 2.3231673, -1.0529735, -1.2732302, 0.87934554, 1.3826215, 0.24184477, 1.3531275, -0.28793597, 2.0084376, 1.4573742, -1.5291485, 0.31902915, -0.23054239, 0.62534, 1.8519323, -2.245485, -1.8446102, 0.66178447, -1.6817732, 0.43443537, -1.101484, 0.8291666, 0.7223018, -0.18338689, 1.9866216, -0.7683655, -1.1324087, -0.671756, -0.99642277, 1.714391, -0.30889648, -1.1144117, 0.58786345, -1.4462819, 0.5452746, -1.4152023, -0.51632243, -1.0784085, -1.8019311, 1.8430812, -0.77855986, 1.8445983, -1.4430277, 0.40093422, 1.7084532, 0.8918805, 0.36253592, -0.4176629, 0.91448, -0.92981076, -0.07481962, 0.8215766, 0.31338146, 0.26393196, -1.0675564, 0.70066214, 0.31446722, 0.87955433, 0.4141644, -0.8118956, -1.1245772, 1.742084, 0.3557291, -2.3003993, -0.01551861, -0.8920257, -0.8597669, -0.8725518, 0.6303311, 1.0367441, -0.6833122, -1.3960947, -0.16622084, 0.06926525, -1.2923226, 0.53113765, -0.04628024, 0.63293314, 2.2518039, -0.3721881, -1.2018218, 0.750074];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
 
  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding valid example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [-1.86841726e-01, -1.87308407e+00, 1.21135116e+00, -3.85009050e-01, 1.72032380e+00, -1.56035602e+00, -1.23059344e+00, 1.23694098e+00, 1.99985504e-03, 3.59522343e-01, 1.60084629e+00, 4.34007555e-01, -2.82945693e-01, 2.37292123e+00, -1.28653407e+00, 8.47842395e-02, -3.52093250e-01, -2.39659071e+00, 1.49246454e-01, 9.20351386e-01, -1.34345913e+00, 4.84796733e-01, -1.19989347e+00, -6.84298515e-01, -1.41301155e+00, 1.03178442e-01, -3.07042211e-01, 1.17741525e+00, 2.58936214e+00, -2.76237011e+00, -1.21565342e+00, -1.09619403e+00, 1.17431641e+00, 5.12142301e-01, 7.71379948e-01, 3.99879634e-01, -5.33092022e-02, 2.90863872e-01, 9.55634058e-01, 1.16327548e+00, 1.80768192e+00, -1.52564144e+00, 1.22480464e+00, 2.37127364e-01, -2.13295698e-01, -6.19941294e-01, 4.97942507e-01, -1.68688416e+00, 1.59314167e+00, -1.27335250e-01, 1.11420155e-01, 1.13719368e+00, 1.68536687e+00, -4.79643047e-01, 1.18607867e+00, -2.52744436e+00, 1.34135664e+00, 5.48298419e-01, -2.08380222e+00, 2.64585400e+00, -9.93354917e-01, 1.28238201e-01, 1.26091874e+00, -6.29126132e-01, -9.49230671e-01, 2.25827789e+00, -1.96100128e+00, 5.89534640e-03, -1.87852085e-01, -1.02403378e+00, 3.96120340e-01, 1.37040257e+00, 3.99355221e+00, 4.34221208e-01, 2.74464667e-01, -5.62437356e-01, -9.14871454e-01, 5.39128900e-01, -9.28685188e-01, 8.34952950e-01, 8.44179749e-01, -5.66052437e-01, -9.57342565e-01, 9.33336258e-01, -4.14666116e-01, -4.52821493e-01, -7.06006944e-01, -1.72656703e+00, -7.26575494e-01, -9.79378521e-02, -4.78667945e-01, 1.78702688e+00, -6.39287651e-01, 1.48564780e+00, -1.79904699e-01, 1.01003110e+00, -3.17118764e-01, -6.75386369e-01, 1.90969336e+00, -1.38342953e+00, 6.97255731e-01, -2.92255253e-01, 1.81634486e+00, 7.17801273e-01, 8.62478435e-01, -4.81892645e-01, -1.35565460e-01, -2.95940900e+00, 2.47845054e-01, 2.67756557e+00, -2.23998690e+00, -5.19674301e-01, 2.54447937e-01, 4.15283501e-01, -1.01065040e+00, 5.07912159e-01, 9.79926169e-01, -1.84304118e-01, -9.52005386e-04, -7.34347284e-01, -1.96684420e-01, -7.13242233e-01, 5.94973564e-01, 8.45057964e-02, 2.48496294e+00, 3.85019749e-01];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
    let type4_length = product(type4.dimensions);

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

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding valid example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [2.43199, 0.7855995, -0.89247733, -0.8421999, -0.21892299, -1.4752842, 1.4666033, 1.2671402, -1.9113951, 2.8163433, 0.42375302, 1.384544, -0.05179526, -0.09704095, -1.0454197, -0.7849326, -1.0726444, -2.2269573, -1.9059162, -1.3809854, -0.21753564, -0.6674532, 0.9924352, -1.3004371, 1.3581562, -0.50957847, 0.43931735, -0.30051446, 1.9288344, 1.3749437, 0.24674952, -1.3658104, -0.24712396, 1.8478253, 0.0548588, 0.5765619, -1.0782311, 2.7078195, 0.54151404, -1.2969424, -0.4957502, -0.8728107, 2.7895741, 0.764437, 1.8849254, -0.16873728, 0.36533558, 2.3231673, -1.0529735, -1.2732302, 0.87934554, 1.3826215, 0.24184477, 1.3531275, 0.62534, 1.8519323, -2.245485, -1.8446102, 0.66178447, -1.6817732, 0.43443537, -1.101484, 0.8291666, 0.7223018, -0.18338689, 1.9866216, -0.7683655, -1.1324087, -0.671756, -0.99642277, 1.714391, -0.30889648];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4, 6, 3]};
    let type4_length = product(type4.dimensions);

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

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.840539, -0.301347, 0.754947, -0.14848, -0.40603, 0.294432, 0.130372, 0.11573, -0.182277, 0.2504, 0.132901, 0.442306, -0.739693, -0.196274, 0.457246, -0.636246, -0.100205, 0.698864, 0.244348, 0.22389, -0.436108, 0.067699, 0.462205, 0.249125, -0.145748, -0.387964, -0.391573, -0.392801, 0.166114, -0.622396, 0.344322, -0.374205, 0.586815, -0.203372, 0.29652, -0.590411, 0.726629, -0.213891, 0.452749, 0.532555, -0.208851, 0.186981, -0.209039, 0.398664, 0.288932, -0.540171, 0.312503, 0.24948, 0.351473, 0.076122, -0.0576253, -0.73055, 0.0665069, -0.271043, 0.634142, 0.466579, 0.536743, 0.389538, 0.417773, -0.355728, -0.591672, 0.40651, 0.586329, 0.384641, 0.0198003, -0.358878, 0.894009, -0.0825318, -0.676451, -0.0935613, 0.138747, 0.351167, -0.0305845, 0.118962, -0.201319, -0.0916215, -0.300945, 0.743041, -0.34075, 0.421278, -0.218791, 0.935359, 0.287684, 0.319749, -0.907324, 0.054362, -0.0883874, 0.0563023, -0.203432, -0.275113, -0.444178, -0.335382, -0.408242, 0.657194, 0.194033, -0.279365, -0.488907, 0.157917, 0.0881365, 0.166668, -0.407001, -0.766027, 0.921655, -0.422149, -0.624807, -0.202641, 0.13341, 0.374139, -0.109369, -0.0353696, -0.0759913, 0.456887, -0.44906, 0.131841, 0.811082, -0.213681, -0.134277, -0.333215, 0.0615286, -0.566144, 0.522554, -0.267049, 0.785754, -0.489062, 0.0728509, -0.0649092, -0.731203, 0.3095, -0.199677, -0.445251, -0.0831503, 0.238257, 0.618959, -0.328446, 0.416281, 0.549062, 0.0333644, -0.340149, -0.154278, 0.142677, -0.110001, 0.15484, -0.368053, 0.619189, -0.580424, -0.123033, 0.133487, -0.461813, 0.328611, 0.600933, 0.907739, 0.245199, -0.767835, 0.49435, 0.235373, -0.0873295, 0.312748, -0.249839, 0.693584, -0.351866, -0.0173133, 0.13876, 0.39089, 0.380607, -0.754171, 0.322982, -0.312857, 0.658611, -0.151223, 0.200055, -0.311675, -0.790939, 0.303812, -0.351079, 0.566216, 0.261687, 0.68551, -0.0862257, 0.290419, -0.175771, -0.449781, -0.2199, -0.312586, -0.399111, -0.0845297, -0.142101, -0.575998, -0.385935, -0.544937, 0.680582, 0.139135, -0.573594];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(mul, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.840539, -0.301347, 0.754947, -0.14848, -0.40603, 0.294432, 0.130372, 0.11573, -0.182277, 0.2504, 0.132901, 0.442306, -0.739693, -0.196274, 0.457246, -0.636246, -0.100205, 0.698864, 0.244348, 0.22389, -0.436108, 0.067699, 0.462205, 0.249125, -0.145748, -0.387964, -0.391573, -0.392801, 0.166114, -0.622396, 0.344322, -0.374205, 0.586815, -0.203372, 0.29652, -0.590411, 0.726629, -0.213891, 0.452749, 0.532555, -0.208851, 0.186981, -0.209039, 0.398664, 0.288932, -0.540171, 0.312503, 0.24948, 0.351473, 0.076122, -0.0576253, -0.73055, 0.0665069, -0.271043, 0.634142, 0.466579, 0.536743, 0.389538, 0.417773, -0.355728, -0.591672, 0.40651, 0.586329, 0.384641, 0.0198003, -0.358878, 0.894009, -0.0825318, -0.676451, -0.0935613, 0.138747, 0.351167, -0.0305845, 0.118962, -0.201319, -0.0916215, -0.300945, 0.743041, -0.34075, 0.421278, -0.218791, 0.935359, 0.287684, 0.319749, -0.907324, 0.054362, -0.0883874, 0.0563023, -0.203432, -0.275113, -0.444178, -0.335382, -0.408242, 0.657194, 0.194033, -0.279365, -0.488907, 0.157917, 0.0881365, 0.166668, -0.407001, -0.766027, 0.921655, -0.422149, -0.624807, -0.202641, 0.13341, 0.374139, -0.109369, -0.0353696, -0.0759913, 0.456887, -0.44906, 0.131841, 0.811082, -0.213681, -0.134277, -0.333215, 0.0615286, -0.566144, 0.522554, -0.267049, 0.785754, -0.489062, 0.0728509, -0.0649092, -0.731203, 0.3095, -0.199677, -0.445251, -0.0831503, 0.238257, 0.618959, -0.328446, 0.416281, 0.549062, 0.0333644, -0.340149, -0.154278, 0.142677, -0.110001, 0.15484, -0.368053, 0.619189, -0.580424, -0.123033, 0.133487, -0.461813, 0.328611, 0.600933, 0.907739, 0.245199, -0.767835, 0.49435, 0.235373, -0.0873295, 0.312748, -0.249839, 0.693584, -0.351866, -0.0173133, 0.13876, 0.39089, 0.380607, -0.754171, 0.322982, -0.312857, 0.658611, -0.151223, 0.200055, -0.311675, -0.790939, 0.303812, -0.351079, 0.566216, 0.261687, 0.68551, -0.0862257, 0.290419, -0.175771, -0.449781, -0.2199, -0.312586, -0.399111, -0.0845297, -0.142101, -0.575998, -0.385935, -0.544937, 0.680582, 0.139135, -0.573594];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(mul, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D same example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [11.0, 3.0, 7.2, 10.6, 11.0, 3.0, 7.4, 10.9, 6.0, 2.0, 7.6, 4.0, 11.0, 3.0, 7.8, 11.5, 11.0, 3.0, 8.0, 11.8, 6.0, 2.0, 8.2, 4.0, 6.0, 2.0, 8.4, 12.4, 6.0, 2.0, 8.6, 12.7, 3.5, 2.0, 8.8, 4.0];
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 4]};
    let type4_length = product(type4.dimensions);

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
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([0.25, 0, 0.2, 0, 0.25, 0, 0, 0.3, 0.25, 0, 0, 0, 0.25, 0.1, 0, 0]));
    model.setOperandValue(bias, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(mul, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D same example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [3.5, 3.0, 3.0, 4.0, 6.0, 3.0, 3.0, 4.0, 3.5, 2.0, 3.0, 4.0, 6.0, 3.0, 3.0, 10.6, 11.0, 3.0, 7.2, 10.9, 6.0, 2.0, 7.4, 4.0, 3.5, 2.0, 3.0, 11.5, 6.0, 2.0, 7.8, 11.8, 3.5, 2.0, 8.0, 4.0];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 4]};
    let type4_length = product(type4.dimensions);

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
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([0.25, 0, 0.2, 0, 0.25, 0, 0, 0.3, 0.25, 0, 0, 0, 0.25, 0.1, 0, 0]));
    model.setOperandValue(bias, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(mul, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D vaild example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8];
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};

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
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type1);

    model.setOperandValue(op2, new Float32Array([0.25, 0, 0.2, 0, 0.25, 0, 0, 0.3, 0.25, 0, 0, 0, 0.25, 0.1, 0, 0]));
    model.setOperandValue(bias, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(mul, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type1_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D valid example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [11, 3, 7.2, 10.9];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    let type4_length = product(type4.dimensions);

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
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([0.25, 0, 0.2, 0, 0.25, 0, 0, 0.3, 0.25, 0, 0, 0, 0.25, 0.1, 0, 0]));
    model.setOperandValue(bias, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(mul, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for Concatenation axis 0 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 0;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                      109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0,
                      201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                      209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);

    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Concatenation axis 1 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 1;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                      201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                      109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0,
                      209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);

    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Concatenation axis 2 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 2;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 103.0, 104.0,
                      201.0, 202.0, 203.0, 204.0,
                      105.0, 106.0, 107.0, 108.0,
                      205.0, 206.0, 207.0, 208.0,
                      109.0, 110.0, 111.0, 112.0,
                      209.0, 210.0, 211.0, 212.0,
                      113.0, 114.0, 115.0, 116.0,
                      213.0, 214.0, 215.0, 216.0]);
    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Concatenation axis 3 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 3;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 4]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 201.0, 202.0,
                      103.0, 104.0, 203.0, 204.0,
                      105.0, 106.0, 205.0, 206.0,
                      107.0, 108.0, 207.0, 208.0,
                      109.0, 110.0, 209.0, 210.0,
                      111.0, 112.0, 211.0, 212.0,
                      113.0, 114.0, 213.0, 214.0,
                      115.0, 116.0, 215.0, 216.0]);

    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Fully connected float 3D input example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 32, 16];
    let op3_expect = [8, 68, 36];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type4);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([2]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for Fully connected float 3D input example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 10, 100];
    let op3_expect = [127];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type4);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([3, 2, 1]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for Fully connected float 4D input example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 32, 16];
    let op3_expect = [8, 68, 36];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 1]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type4);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([2]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for Mul example', async function() {
    let operandIndex = 0;
    let model = await nn.createModel(options);
    let TENSOR_DIMENSIONS = [2, 2, 2, 2];
    let value0 = 0.4;
    let value1 = 0.5;

    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);

    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(float32TensorType);
    let input0Data = new Float32Array(tensorLength);
    input0Data.fill(value0);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(float32TensorType);
    let output = operandIndex++;
    model.addOperand(float32TensorType);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();

    compilation.setPreference(getPreferenceCode(options.prefer));

    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1Data = new Float32Array(tensorLength);
    input1Data.fill(value1);

    execution.setInput(0, input1Data);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], input0Data[i] * input1Data[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/5', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  50,  60,  70,  80,
                         90, 100, 110, 120, 130, 140, 150, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/6', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]);

    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
                         7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12,
                        13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
                        19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
                        25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30,
                        31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36,
                        37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42,
                        43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48,
                        49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54,
                        55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60];


    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/5', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  50,  60,  70,  80,
                         90, 100, 110, 120, 130, 140, 150, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/6', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1,  2,  3,  4]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  20,  40,  60,  80,
                         30,  60,  90, 120,  40,  80, 120, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Reshape example', async function() {
    let model = await nn.createModel(options);

    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions:[1, 4]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
    model.setOperandValue(1, new Int32Array([2, 2]));
    model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
    model.addOperation(nn.RESHAPE, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData = new Float32Array(tensorLength);
    inputData.set([1.0, 2.0, 3.0, 4.0]);
    execution.setInput(0, inputData);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], inputData[i]));
    }
  });

  it('check result for Resize bilinear with inputs (without align_corners) distorted example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 4, 4.5, 7, 6, 10, 6, 10, 7, 8, 8.5, 11, 10, 14, 10, 14, 9, 10, 10.5, 13, 12, 16, 12, 16, 3, 4, 4.5, 7, 6, 10, 6, 10, 7, 8, 8.5, 11, 10, 14, 10, 14, 9, 10, 10.5, 13, 12, 16, 12, 16];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([4]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear by align_corners(FALSE) distorted example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 3.6666667, 5.666667, 5, 7, 6.333333, 8.333333, 9, 11, 10.333333, 12.333333, 6.333334, 8.333334, 9.000001, 11.000001, 10.333334, 12.333334, 1, 3, 3.6666667, 5.666667, 5, 7, 6.333333, 8.333333, 9, 11, 10.333333, 12.333333, 6.333334, 8.333334, 9.000001, 11.000001, 10.333334, 12.333334];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.setOperandValue(align_corners, new Int32Array([0]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear by align_corners(TRUE) distorted example/6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 3, 5, 5, 7, 5, 7, 7, 9, 9, 11, 9, 11, 11, 13, 13, 15, 1, 3, 3, 5, 5, 7, 5, 7, 7, 9, 9, 11, 9, 11, 11, 13, 13, 15];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.setOperandValue(align_corners, new Int32Array([1]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear with inputs (without align_corners) remain size example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear by align_corners(FALSE) remain size example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([0]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear by align_corners(TRUE) remain size example/6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([1]));

    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear with inputs (without align_corners) zoom in example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 3.6666667, 4, 5, 7, 8, 6, 8.6666667, 10, 9, 9.6666667, 10, 11, 13, 14, 12, 14.6666667, 16];
    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 1]};
    let type1_length = product(type1.dimensions);

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
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear by align_corners(TRUE) zoom in example/4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 4, 4.5, 7, 6, 10, 6, 7, 7.5, 10, 9, 13, 9, 10, 10.5, 13, 12, 16, 3, 4, 4.5, 7, 6, 10, 6, 7, 7.5, 10, 9, 13, 9, 10, 10.5, 13, 12, 16];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.setOperandValue(align_corners, new Int32Array([1]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear with inputs (without align_corners) zoom out example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 4, 10, 13];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear by align_corners(FALSE) zoom out example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 3, 7, 9, 10, 12, 11.5, 9, 1, 3, 7, 9, 10, 12, 11.5, 9];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([0]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Resize bilinear by align_corners(TRUE) zoom out example/6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 3, 9, 11, 7, 9, 15, 17, 1, 3, 9, 11, 7, 9, 15, 17];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([1]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax example', async function() {
    let model = await nn.createModel(options);
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.FLOAT32});
    model.setOperandValue(1, new Float32Array([1.0]));
    model.addOperand(float32TensorType);
    model.addOperation(nn.SOFTMAX, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([1.0, 1.0, 1.0, 1.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();
    let expectedData = new Float32Array(tensorLength);
    expectedData.set([0.5, 0.5, 0.5, 0.5]);

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });
});
