from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from onnx import numpy_helper

def gd_predict_create_graph(ft_bits, num_classes, feature_table):
    features = helper.make_tensor_value_info('Features', TensorProto.UINT32, None)#TensorProto.FLOAT, [2**ft_bits])#, num_classes])
    classes = helper.make_tensor_value_info('Valid_Classes', TensorProto.FLOAT, None)

    outputs = helper.make_tensor_value_info('Outputs', TensorProto.INT64, None)

    '''
    murmur_def = helper.make_node(
        'MurmurHash3', # node name
        ['Features'], # inputs
        ['Hashed'], # outputs
        domain="com.microsoft",
    )
    '''

    ft_bits_def = helper.make_node("Constant",inputs=[],outputs=['bits'],value=onnx.helper.make_tensor(name="bs",data_type=TensorProto.UINT32,dims=[1],
        vals=np.array([32 - ft_bits])))

    bit_right_def = helper.make_node(
        'BitShift', # node name
        ['Features', 'bits'], # inputs
        ['Features_Cut'], # outputs
        direction="RIGHT",
    )



    predicts = []
    for i in range(num_classes):
        predicts.append(helper.make_node(
            'GDPredict', # node name
            ['Features_Cut'], # inputs
            ['Predict'+str(i)], # outputs
            weights=' '.join(str(x) for x in feature_table[i]), # Attributes
            domain="com.microsoft",
        ))

    concat_def = helper.make_node(
        'Concat', # node name
        ['Predict'+str(i) for i in range(num_classes)], # inputs
        ['ClassScores'], # outputs
        axis=0
    )

    class_mask_def = helper.make_node(
        'Mul', # node name
        ['ClassScores', 'Valid_Classes'], # inputs
        ['ClassScores_Masked'] # outputs
    )


    argmax_def = helper.make_node(
        'ArgMax', # node name
        ['ClassScores_Masked'], # inputs
        ['Select_Class'], # outputs
        axis=0
    )

    class_shift_def = helper.make_node("Constant",inputs=[],outputs=['ClassShift'],value_int=1)

    add_1_def = helper.make_node(
    'Add',
    ['Select_Class', 'ClassShift'],
    ['Outputs']
    )

    fullNodes = []
    for p in predicts: fullNodes.append(p)
    fullNodes.append(concat_def)
    fullNodes.append(argmax_def)
    fullNodes.append(add_1_def)
    fullNodes.append(class_mask_def)
    fullNodes.append(class_shift_def)
    fullNodes.append(ft_bits_def)
    fullNodes.append(bit_right_def)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        fullNodes,
        "testgdpredict",
        [features, classes],
        [outputs]
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='vwtest')
    #print(model_def)

    meta = model_def.metadata_props.add()
    meta.key = "class_count"
    meta.value = str(num_classes)
    onnx.save(model_def,"model.onnx")
    '''

    features =  helper.make_tensor_value_info('Features', TensorProto.FLOAT, [2**ft_bits])
    classes = helper.make_tensor_value_info('Classes', TensorProto.INT64, [num_classes])

    outputs = helper.make_tensor_value_info('Outputs', TensorProto.INT64, [1])

    body = helper.make_graph(
    [
    helper.make_node("Gather", ['feature_table','iteration_num'],['class_weights'],axis=0),
    helper.make_node("GDPredict", ['Features','class_weights'],['class_score'], domain="com.microsoft"),
    helper.make_node("Concat",["class_scores","class_score"],["final_class_scores"],axis=0)
    ],
    "body",
    [helper.make_tensor_value_info('iteration_num', TensorProto.INT64, [1]),
    helper.make_tensor_value_info('loop_cond',TensorProto.BOOL,[1]),
    helper.make_tensor_value_info('class_scores',TensorProto.FLOAT,[]),
    helper.make_tensor_value_info('feature_table',TensorProto.FLOAT,feature_table.shape),
    features,
    ],
    [helper.make_tensor_value_info('final_class_scores',TensorProto.FLOAT,[])]
    )

    graph_def = helper.make_graph([
helper.make_node("Constant",inputs=[],outputs=['num_classes'],value=onnx.helper.make_tensor(name="nc",data_type=TensorProto.INT64,dims=[1],
    vals=np.array([num_classes]))),
    helper.make_node("Constant",inputs=[],outputs=['feature_table'],value=onnx.helper.make_tensor(name="ft",data_type=TensorProto.FLOAT,dims=feature_table.shape,
        vals=feature_table.flatten().astype(float))),
    helper.make_node("Constant",inputs=[],outputs=['class_scores'],value=onnx.helper.make_tensor(name="cs",data_type=TensorProto.INT64,dims=[num_classes],
        vals=np.zeros((num_classes)).astype(float))),
    helper.make_node("Loop",["num_classes","","class_scores", "feature_table", "Features"],["final_class_scores"],body=body),

        helper.make_node(
            'ArgMax', # node name
            ['final_class_scores'], # inputs
            ['Outputs'], # outputs
            axis=0
        )],
    "testgdpredict",
    [features],
    [outputs]
    )

    model_def = helper.make_model(graph_def, producer_name='vwtest')
    #print(model_def)
    onnx.save(model_def,"test_predict.onnx")
    onnx.checker.check_model(model_def)
    '''

    return model_def

def inv_hash_parse(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        ft_bits = int(lines[4].split(":")[-1])
        num_classes = int(lines[8].split(" ")[-1])
        stride = int(lines[7].split(" ")[0])+1
        feature_table = np.zeros((num_classes, 2**ft_bits))
        for line in lines[11:]:
            parts = line.split(":")
            isclasspart=False
            for i in range(num_classes-1):
                if "[{}]".format(i+1) in parts[0]:
                    feature_table[i+1, int(parts[1])-(i+1)*stride] = float(parts[2])
                    isclasspart = True
                    break
            if not isclasspart:
                feature_table[0,int(parts[1])] = float(parts[2])
    #print(feature_table.T)
    gd_predict_create_graph(ft_bits, num_classes, feature_table)
    print("Model Created!")
    '''
    with open(filepath, "r") as f:
        lines = f.readlines()
        ft_bits = int(lines[4].split(":")[-1])
        num_classes = int(lines[8].split(" ")[-1])
        feature_table = np.zeros((2**ft_bits))
        for line in lines[11:]:
            parts = line.split(":")
            feature_table[int(parts[1])] = float(parts[2])

    print(' '.join(str(x) for x in feature_table))
    '''
inv_hash_parse("invhash.txt")
